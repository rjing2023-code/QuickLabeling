import os
import shutil
import random
import argparse
import yaml
from pathlib import Path

# 数据集划分脚本    
# 该脚本用于将数据集划分为训练集和验证集，用于YOLO模型的训练和评估

def main():
    parser = argparse.ArgumentParser(description="Split dataset into train and val sets for YOLO")
    parser.add_argument("--source_dir", type=str, default="dataset", help="Source dataset directory containing images and labels folders")
    parser.add_argument("--output_dir", type=str, default="yolo_split_dataset", help="Output directory for split dataset")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation set ratio (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    val_ratio = args.val_ratio
    
    # Verify source directories
    src_images_dir = source_dir / "images"
    src_labels_dir = source_dir / "labels"
    
    if not src_images_dir.exists():
        print(f"Error: Images directory not found at {src_images_dir}")
        return
    if not src_labels_dir.exists():
        print(f"Error: Labels directory not found at {src_labels_dir}")
        return

    # Create output directories
    for split in ['train', 'val']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    image_files = [f for f in os.listdir(src_images_dir) if os.path.splitext(f)[1].lower() in valid_extensions]
    
    if not image_files:
        print(f"No image files found in {src_images_dir}")
        return
        
    print(f"Found {len(image_files)} images.")
    
    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(image_files)
    
    split_idx = int(len(image_files) * (1 - val_ratio))
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"Splitting into {len(train_files)} training and {len(val_files)} validation samples.")
    
    def copy_files(files, split_name):
        print(f"Copying {split_name} set...")
        for img_file in files:
            # Copy image
            src_img_path = src_images_dir / img_file
            dst_img_path = output_dir / split_name / 'images' / img_file
            shutil.copy2(src_img_path, dst_img_path)
            
            # Copy label
            label_name = os.path.splitext(img_file)[0] + ".txt"
            src_label_path = src_labels_dir / label_name
            dst_label_path = output_dir / split_name / 'labels' / label_name
            
            if src_label_path.exists():
                shutil.copy2(src_label_path, dst_label_path)
            else:
                # If no label file exists, it's a background image (no objects), create empty label file
                # or just skip? YOLO usually expects a file or assumes empty if missing? 
                # Better to be explicit and check. 
                # Assuming previous export script always generates .txt if it processes the frame.
                # If missing, it might be an error or intentional.
                # Let's create an empty file just in case to avoid YOLO warnings if it expects one.
                # But standard YOLO dataset structure: image exists -> look for label. If label missing, no objects.
                # So we don't strictly need to create it if it's missing, but copying if exists is key.
                pass

    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    
    # Create dataset.yaml
    yaml_content = {
        'path': os.path.abspath(output_dir),
        'train': 'train/images',
        'val': 'val/images',
        'names': {
            0: 'object'
        }
    }
    
    yaml_path = output_dir / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
        
    print("-" * 30)
    print("Dataset split completed successfully.")
    print(f"Output directory: {output_dir}")
    print(f"Configuration file: {yaml_path}")
    print("You can use this yaml file path for YOLO training.")

if __name__ == "__main__":
    main()
