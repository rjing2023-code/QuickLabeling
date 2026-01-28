import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import math

def plot_all_histograms(npy_dir, output_file="all_histograms.png"):
    # 搜索目录下所有 npy 文件
    npy_files = glob.glob(os.path.join(npy_dir, "*.npy"))
    if not npy_files:
        print(f"在 {npy_dir} 中未找到 .npy 文件")
        return

    num_files = len(npy_files)
    print(f"找到 {num_files} 个 npy 文件，开始处理...")

    # 计算子图布局 (尽可能接近正方形，或者固定列数)
    cols = 3
    rows = math.ceil(num_files / cols)
    
    # 创建大图
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4), constrained_layout=True)
    
    # 展平 axes 数组以便遍历，处理只有一个子图的情况
    if num_files == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, file_path in enumerate(npy_files):
        ax = axes[i]
        file_name = os.path.basename(file_path)
        camera_name = os.path.splitext(file_name)[0]  # 假设文件名即相机名
        
        try:
            print(f"正在处理: {file_name}")
            # 加载数据
            npy_data = np.load(file_path, allow_pickle=True)
            
            # 复用 labeling_app.py 中的逻辑
            # 计算每帧框数量
            counts = [len(x) for x in npy_data]
            frames = list(range(1, len(counts) + 1)) # 帧号从 1 开始
            
            # 绘制直方图
            ax.bar(frames, counts, width=1.0, color='skyblue', edgecolor='none')
            
            # 设置标题和标签
            ax.set_title(f"Camera: {camera_name}\n({len(frames)} frames)", fontsize=10)
            ax.set_xlabel("Frame Number", fontsize=8)
            ax.set_ylabel("Box Count", fontsize=8)
            ax.set_xlim(0, len(counts) + 1)
            ax.grid(axis='y', linestyle='--', alpha=0.5)

        except Exception as e:
            print(f"处理 {file_name} 时出错: {e}")
            ax.text(0.5, 0.5, f"Error loading\n{file_name}", 
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='red')

    # 隐藏多余的子图
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # 保存图片
    print(f"正在保存结果到 {output_file} ...")
    plt.savefig(output_file, dpi=150)
    print("完成！")
    # plt.show() # 如果需要显示可以取消注释

if __name__ == "__main__":
    # 默认搜索当前目录，也可以修改为指定目录
    current_dir = os.getcwd()
    # 如果想指定子目录，可以修改这里，例如 os.path.join(current_dir, "npy_data")
    target_dir = os.path.join(current_dir, "npy_data") 
    
    output_path = os.path.join(target_dir, "combined_histograms.png")
    
    plot_all_histograms(target_dir, output_path)
