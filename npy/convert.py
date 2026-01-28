import os

# 设置文件所在的文件夹路径，'.' 表示当前目录
folder_path = './'

# 遍历 0 到 24
for i in range(25):
    # 构建原始文件名，例如 "0_result_boxes.npy"
    old_name = f"{i}_boxes.npy"
    
    # 检查文件是否存在，防止报错
    if os.path.exists(os.path.join(folder_path, old_name)):
        # 计算 Camera 编号 (1-5) 和 序列编号 (1-5)
        # i // 5 是整除，0-4得到0, 5-9得到1...
        camera_num = (i // 5) + 1
        # i % 5 是取余，得到 0,1,2,3,4
        sequence_num = (i % 5) + 1
        
        # 构建新文件名，例如 "Camera1-1.npy"
        new_name = f"Camera{camera_num}-{sequence_num}.npy"
        
        # 执行重命名
        old_file = os.path.join(folder_path, old_name)
        new_file = os.path.join(folder_path, new_name)
        
        os.rename(old_file, new_file)
        print(f"重命名成功: {old_name} -> {new_name}")
    else:
        print(f"未找到文件: {old_name}")

print("任务完成！")