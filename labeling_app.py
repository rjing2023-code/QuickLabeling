import sys
import cv2
import os
import json
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QLineEdit, QMessageBox, QScrollArea, QSizePolicy, QTextEdit)
from PyQt5.QtCore import Qt, QTimer, QPoint
from PyQt5.QtGui import QImage, QPixmap, QIntValidator, QPainter, QPen, QColor, QBrush

class AnnotatedImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.mouse_pos = None
        self.selecting = False
        self.sel_start = None
        self.sel_end = None
        self.annotations = []
        self.parent_ref = parent

    def set_annotations(self, annots):
        self.annotations = annots or []
        self.update()

    def mouseMoveEvent(self, event):
        self.mouse_pos = event.pos()
        if self.selecting and self.sel_start is not None:
            self.sel_end = event.pos()
        self.update()
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.pixmap() is not None:
            self.selecting = True
            self.sel_start = event.pos()
            self.sel_end = event.pos()
            self.update()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.selecting and self.sel_start and self.sel_end:
            x1 = max(0, min(self.sel_start.x(), self.sel_end.x()))
            y1 = max(0, min(self.sel_start.y(), self.sel_end.y()))
            x2 = min(self.width()-1, max(self.sel_start.x(), self.sel_end.x()))
            y2 = min(self.height()-1, max(self.sel_start.y(), self.sel_end.y()))
            if x2 > x1 and y2 > y1 and hasattr(self.parent_ref, "add_annotation"):
                self.parent_ref.add_annotation([x1, y1, x2, y2])
            self.selecting = False
            self.sel_start = None
            self.sel_end = None
            self.update()
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.pixmap() is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        if self.mouse_pos is not None and 0 <= self.mouse_pos.x() < self.width() and 0 <= self.mouse_pos.y() < self.height():
            pen_cross = QPen(QColor(255, 255, 255))
            pen_cross.setWidth(2)
            painter.setPen(pen_cross)
            mx, my = self.mouse_pos.x(), self.mouse_pos.y()
            painter.drawLine(mx-5, my, mx+5, my)
            painter.drawLine(mx, my-5, mx, my+5)
            pen_dash = QPen(QColor(220, 220, 220))
            pen_dash.setStyle(Qt.DashLine)
            pen_dash.setWidth(1)
            painter.setPen(pen_dash)
            painter.drawLine(0, my, self.width(), my)
            painter.drawLine(mx, 0, mx, self.height())
        if self.annotations:
            pen_box = QPen(QColor(255, 0, 255))
            pen_box.setWidth(2)
            pen_box.setStyle(Qt.DashLine)
            painter.setPen(pen_box)
            painter.setBrush(Qt.NoBrush)
            for b in self.annotations:
                if len(b) >= 4:
                    x1, y1, x2, y2 = b[:4]
                    painter.drawRect(x1, y1, x2 - x1, y2 - y1)
        if self.selecting and self.sel_start and self.sel_end:
            pen_sel = QPen(QColor(0, 0, 255))
            pen_sel.setStyle(Qt.DashLine)
            painter.setPen(pen_sel)
            x1 = min(self.sel_start.x(), self.sel_end.x())
            y1 = min(self.sel_start.y(), self.sel_end.y())
            x2 = max(self.sel_start.x(), self.sel_end.x())
            y2 = max(self.sel_start.y(), self.sel_end.y())
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)
    
    def enterEvent(self, event):
        self.setCursor(Qt.BlankCursor)
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        self.setCursor(Qt.ArrowCursor)
        super().leaveEvent(event)

class VideoLabeler(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QuickLabeling - Video Annotator")
        self.resize(1600, 1200)  # 初始窗口大小，留出空间给控制栏

        # 状态变量
        self.video_cap = None
        self.video_path = None
        self.camera_name = None
        self.npy_data = None
        self.total_frames = 0
        self.current_frame_idx = 0
        self.image_width = 1440
        self.image_height = 1080
        
        # 直方图相关
        self.ax = None
        self.canvas = None
        self.current_frame_line = None
        self.annotations = {}
        self.annotations_path = os.path.join(os.getcwd(), "annotations.json")
        self.load_annotations()

        # 初始化 UI
        self.init_ui()

    def init_ui(self):
        # 主窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 1. 顶部控制栏
        control_layout = QHBoxLayout()
        
        # 加载按钮
        self.btn_load = QPushButton("加载视频 (Load Video)")
        self.btn_load.clicked.connect(self.load_video)
        control_layout.addWidget(self.btn_load)

        # 加载 NPY 按钮
        self.btn_load_npy = QPushButton("加载 NPY (Load NPY)")
        self.btn_load_npy.clicked.connect(self.load_npy)
        control_layout.addWidget(self.btn_load_npy)

        # 帧数显示和跳转
        control_layout.addStretch()
        
        control_layout.addWidget(QLabel("帧号:"))
        
        self.input_frame = QLineEdit()
        self.input_frame.setValidator(QIntValidator()) # 只能输入整数
        self.input_frame.setFixedWidth(80)
        self.input_frame.returnPressed.connect(self.jump_to_frame_from_input) # 回车跳转
        control_layout.addWidget(self.input_frame)

        self.label_total_frames = QLabel("/ 0")
        control_layout.addWidget(self.label_total_frames)

        self.btn_jump = QPushButton("跳转 (Go)")
        self.btn_jump.clicked.connect(self.jump_to_frame_from_input)
        control_layout.addWidget(self.btn_jump)

        control_layout.addStretch()

        # 说明标签
        tips_label = QLabel("快捷键: 'A' 上一帧, 'D' 下一帧")
        tips_label.setStyleSheet("color: gray;")
        control_layout.addWidget(tips_label)

        control_layout.addStretch()
        self.label_annotated = QLabel("已在 0 帧上标注")
        control_layout.addWidget(self.label_annotated)
        self.btn_prev_annot = QPushButton("上一个标注帧")
        self.btn_prev_annot.clicked.connect(self.goto_prev_annotated)
        control_layout.addWidget(self.btn_prev_annot)
        self.btn_next_annot = QPushButton("下一个标注帧")
        self.btn_next_annot.clicked.connect(self.goto_next_annotated)
        control_layout.addWidget(self.btn_next_annot)

        main_layout.addLayout(control_layout)

        # 1.5 直方图区域 (插入在控制栏和图片展示区之间)
        # 创建 matplotlib figure
        fig = Figure(figsize=(5, 1.5), dpi=100) # 高度较小
        self.canvas = FigureCanvas(fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.canvas.setFixedHeight(150) # 固定高度
        self.ax = fig.add_subplot(111)
        # 初始化空图
        self.ax.set_title("Detection Boxes Histogram")
        self.ax.set_xlabel("Frame")
        self.ax.set_ylabel("Count")
        fig.tight_layout()
        
        main_layout.addWidget(self.canvas)

        # 2. 图片展示区域
        # 使用 ScrollArea 以便在窗口较小时也能查看大图，或者直接展示大图
        self.scroll_area = QScrollArea()
        # setWidgetResizable(False) 确保 Label 根据内容调整大小，从而显示原分辨率图片并出现滚动条
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setAlignment(Qt.AlignCenter)
        
        self.image_label = AnnotatedImageLabel(self)
        self.image_label.setText("请加载视频文件")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        # self.image_label.setScaledContents(True) # 如果需要缩放可以开启，但用户要求原分辨率

        self.scroll_area.setWidget(self.image_label)
        content_layout = QHBoxLayout()
        content_layout.addWidget(self.scroll_area, 3)
        self.help_text = QTextEdit()
        self.help_text.setReadOnly(True)
        self.help_text.setMinimumWidth(320)
        self.help_text.setText(
            "用户帮助手册\n\n"
            "工作流程\n"
            "1. 运行 run.bat 启动程序\n"
            "2. 点击“加载视频”选择视频文件\n"
            "3. 可选：点击“加载 NPY”叠加检测框并生成直方图\n"
            "4. 使用 A/D 或输入帧号进行导航\n"
            "5. 在图像上拖拽鼠标左键框选矩形进行标注\n"
            "6. 按 Q 删除当前帧最后一次标注\n"
            "7. 使用“上一个标注帧”“下一个标注帧”快速跳转\n\n"
            "显示说明\n"
            "- NPY 框：绿色淡蒙版叠加\n"
            "- 用户标注：品红色虚线矩形\n"
            "- 鼠标指示：红色十字与延长虚线\n"
            "- 直方图：蓝柱为框数量，橙点为已标注帧，红虚线为当前帧\n\n"
            "快捷键\n"
            "- A：上一帧\n"
            "- D：下一帧\n"
            "- Q：删除当前帧最后一次标注\n"
            "- 回车：在帧号输入框中回车跳转\n"
        )
        content_layout.addWidget(self.help_text, 1)
        main_layout.addLayout(content_layout)

    def load_npy(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择 NPY 文件", "", "Numpy Files (*.npy)")
        if file_path:
            try:
                self.npy_data = np.load(file_path, allow_pickle=True)
                QMessageBox.information(self, "成功", "NPY 数据加载成功！")
                
                # 更新直方图
                self.update_histogram()

                # 刷新当前帧
                if self.video_cap:
                    self.show_frame(self.current_frame_idx)
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载 NPY 失败: {e}")

    def update_histogram(self):
        if self.npy_data is not None:
            counts = [len(x) for x in self.npy_data]
            frames = list(range(1, len(counts) + 1))
        else:
            if self.total_frames <= 0:
                return
            counts = [0] * self.total_frames
            frames = list(range(1, self.total_frames + 1))
        self.ax.clear()
        self.ax.bar(frames, counts, width=1.0, color='skyblue', edgecolor='none')
        self.ax.set_title("Detection Boxes Histogram")
        self.ax.set_xlabel("Frame Number")
        self.ax.set_ylabel("Box Count")
        self.ax.set_xlim(0, (frames[-1] if frames else 0) + 1)
        maxc = max(counts) if counts else 0
        ann_frames = self.get_annotated_frames()
        if ann_frames:
            ys = []
            xs = []
            for f in ann_frames:
                xs.append(f)
                y = counts[f - 1] if 0 <= f - 1 < len(counts) else 0
                ys.append(y)
            self.ax.scatter(xs, ys, color='orange', s=30, zorder=3)
            offset = max(1, maxc) * 0.05
            for f, y in zip(xs, ys):
                self.ax.text(f, y + offset, str(f), rotation=90, color='orange', fontsize=8, ha='center', va='bottom')
        self.current_frame_line = self.ax.axvline(x=self.current_frame_idx + 1, color='red', linewidth=2, linestyle='--')
        self.canvas.draw()

    def load_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "Video Files (*.mp4 *.avi *.mkv *.mov)")
        if file_path:
            self.video_path = file_path
            self.camera_name = os.path.basename(file_path)
            self.video_cap = cv2.VideoCapture(file_path)
            if not self.video_cap.isOpened():
                QMessageBox.critical(self, "错误", "无法打开视频文件！")
                return
            
            self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.label_total_frames.setText(f"/ {self.total_frames}")
            self.current_frame_idx = 0
            
            # 显示第一帧
            self.show_frame(self.current_frame_idx)
            self.update_frame_input_display()
            
            # 聚焦到主窗口以便接收键盘事件，而不是停留在按钮上
            self.setFocus()
            self.update_annotation_status()

    def show_frame(self, frame_idx):
        if self.video_cap is None or not self.video_cap.isOpened():
            return

        # 限制帧号范围
        if frame_idx < 0:
            frame_idx = 0
        elif frame_idx >= self.total_frames:
            frame_idx = self.total_frames - 1

        self.current_frame_idx = frame_idx
        
        # 设置读取位置
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        ret, frame = self.video_cap.read()
        
        if ret:
            # OpenCV BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 绘制 NPY 蒙版
            if self.npy_data is not None and self.current_frame_idx < len(self.npy_data):
                try:
                    boxes = self.npy_data[self.current_frame_idx]
                    if len(boxes) > 0:
                        overlay = frame.copy()
                        for box in boxes:
                            if len(box) >= 4:
                                x1, y1, x2, y2 = map(int, box[:4])
                                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 128, 0), -1)
                        
                        alpha = 0.35
                        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                except Exception as e:
                    print(f"Error drawing NPY mask: {e}")

            h, w, ch = frame.shape
            bytes_per_line = ch * w
            
            # 转换为 QImage
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # 转换为 QPixmap 并显示
            self.image_label.setPixmap(QPixmap.fromImage(qt_image))
            self.image_label.resize(w, h) # 确保 Label 大小适应图片
            
            self.update_frame_input_display()

            # 更新直方图指示器
            if self.current_frame_line:
                self.current_frame_line.set_xdata([self.current_frame_idx + 1]) # 帧号从1开始
                self.canvas.draw_idle() # 高效重绘
            self.image_label.set_annotations(self.get_current_annotations())
        else:
            self.image_label.setText(f"无法读取第 {frame_idx + 1} 帧")

    def update_frame_input_display(self):
        # 显示时 +1，符合人类习惯（1-based），内部逻辑用 0-based
        self.input_frame.setText(str(self.current_frame_idx + 1))

    def jump_to_frame_from_input(self):
        if not self.video_cap:
            return
            
        text = self.input_frame.text()
        if text.isdigit():
            target_frame = int(text) - 1 # 转换为 0-based
            self.show_frame(target_frame)
        
        # 恢复焦点到主窗口
        self.setFocus()

    def keyPressEvent(self, event):
        if not self.video_cap:
            return

        # A 键：上一帧
        if event.key() == Qt.Key_A:
            if self.current_frame_idx > 0:
                self.show_frame(self.current_frame_idx - 1)
        
        # D 键：下一帧
        elif event.key() == Qt.Key_D:
            if self.current_frame_idx < self.total_frames - 1:
                self.show_frame(self.current_frame_idx + 1)
        
        else:
            super().keyPressEvent(event)
        if event.key() == Qt.Key_Q:
            self.delete_last_annotation()
            if self.video_cap:
                self.show_frame(self.current_frame_idx)

    def get_annotated_frames(self):
        if not self.camera_name:
            return []
        cam = self.annotations.get(self.camera_name, {})
        frames = []
        for k, v in cam.items():
            if isinstance(v, list) and len(v) > 0:
                try:
                    frames.append(int(k))
                except Exception:
                    pass
        frames.sort()
        return frames

    def update_annotation_status(self):
        frames = self.get_annotated_frames()
        self.label_annotated.setText(f"已在 {len(frames)} 帧上标注")
        self.btn_prev_annot.setEnabled(len(frames) > 0)
        self.btn_next_annot.setEnabled(len(frames) > 0)
        self.update_histogram()

    def goto_prev_annotated(self):
        frames = self.get_annotated_frames()
        if not frames:
            return
        current = self.current_frame_idx + 1
        prev = None
        for f in frames:
            if f < current:
                prev = f
        if prev is None:
            return
        self.show_frame(prev - 1)
    
    def goto_next_annotated(self):
        frames = self.get_annotated_frames()
        if not frames:
            return
        current = self.current_frame_idx + 1
        next_f = None
        for f in frames:
            if f > current:
                next_f = f
                break
        if next_f is None:
            return
        self.show_frame(next_f - 1)

    def load_annotations(self):
        try:
            if os.path.exists(self.annotations_path):
                with open(self.annotations_path, "r", encoding="utf-8") as f:
                    self.annotations = json.load(f)
            else:
                self.annotations = {}
        except Exception:
            self.annotations = {}

    def save_annotations(self):
        try:
            with open(self.annotations_path, "w", encoding="utf-8") as f:
                json.dump(self.annotations, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def get_current_annotations(self):
        if not self.camera_name:
            return []
        cam = self.annotations.get(self.camera_name, {})
        return cam.get(str(self.current_frame_idx + 1), [])

    def add_annotation(self, box):
        if not self.camera_name:
            return
        cam = self.annotations.get(self.camera_name, {})
        frame_key = str(self.current_frame_idx + 1)
        lst = cam.get(frame_key, [])
        lst.append(box)
        cam[frame_key] = lst
        self.annotations[self.camera_name] = cam
        self.save_annotations()
        self.image_label.set_annotations(self.get_current_annotations())
        self.update_annotation_status()

    def delete_last_annotation(self):
        if not self.camera_name:
            return
        cam = self.annotations.get(self.camera_name, {})
        frame_key = str(self.current_frame_idx + 1)
        lst = cam.get(frame_key, [])
        if lst:
            lst.pop()
            if lst:
                cam[frame_key] = lst
            else:
                cam.pop(frame_key, None)
        self.annotations[self.camera_name] = cam
        self.save_annotations()
        self.image_label.set_annotations(self.get_current_annotations())
        self.update_annotation_status()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoLabeler()
    window.show()
    sys.exit(app.exec_())
