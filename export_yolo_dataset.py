import json
import os
import cv2
import argparse
import sys

# 导出YOLO数据集脚本
# 该脚本用于将标注好的视频数据集导出为YOLO格式的数据集，用于YOLO模型的训练和评估

def convert_to_yolo_format(box, img_width, img_height):
    """
    Convert [x1, y1, x2, y2] to [x_center, y_center, width, height] normalized.
    """
    x1, y1, x2, y2 = box
    
    # Calculate center, width, height
    w = x2 - x1
    h = y2 - y1
    x_center = x1 + w / 2
    y_center = y1 + h / 2
    
    # Normalize
    x_center /= img_width
    y_center /= img_height
    w /= img_width
    h /= img_height
    
    # Clamp to [0, 1] just in case
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))
    
    return x_center, y_center, w, h

def convert_from_yolo_format(yolo_box, img_width, img_height):
    """
    Convert [x_center, y_center, w, h] normalized back to [x1, y1, x2, y2].
    Used for debug visualization.
    """
    x_center, y_center, w, h = yolo_box
    
    # Denormalize
    w_pixel = w * img_width
    h_pixel = h * img_height
    x_center_pixel = x_center * img_width
    y_center_pixel = y_center * img_height
    
    # Calculate corners
    x1 = int(x_center_pixel - w_pixel / 2)
    y1 = int(y_center_pixel - h_pixel / 2)
    x2 = int(x_center_pixel + w_pixel / 2)
    y2 = int(y_center_pixel + h_pixel / 2)
    
    return x1, y1, x2, y2

def main():
    parser = argparse.ArgumentParser(description="Export annotations to YOLO format dataset")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing video files")
    parser.add_argument("--json_file", type=str, default="annotations.json", help="Path to annotations json file")
    parser.add_argument("--output_dir", type=str, default="dataset", help="Output directory for dataset")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode: draw boxes on images and save to 'debug' folder")
    
    args = parser.parse_args()
    
    video_dir = args.video_dir
    json_file = args.json_file
    output_dir = args.output_dir
    debug_mode = args.debug
    
    if not os.path.exists(json_file):
        print(f"Error: JSON file not found: {json_file}")
        return
        
    if not os.path.exists(video_dir):
        print(f"Error: Video directory not found: {video_dir}")
        return

    # Create output directories
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    debug_dir = None
    if debug_mode:
        debug_dir = os.path.join(output_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        print(f"Debug mode enabled. Debug images will be saved to: {debug_dir}")
    
    print(f"Loading annotations from {json_file}...")
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    print(f"Found annotations for {len(data)} videos.")
    
    total_images_saved = 0
    total_boxes_saved = 0
    
    for video_name, frames_data in data.items():
        video_path = os.path.join(video_dir, video_name)
        
        # Check if video exists
        if not os.path.exists(video_path):
            print(f"Warning: Video file not found: {video_path}. Skipping...")
            continue
            
        print(f"Processing video: {video_name}...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video: {video_path}")
            continue
            
        # Get video dimensions
        img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        for frame_key, boxes in frames_data.items():
            try:
                # JSON keys are 1-based strings, convert to 0-based integer
                frame_idx = int(frame_key) - 1
            except ValueError:
                print(f"Warning: Invalid frame key '{frame_key}' in video {video_name}")
                continue
                
            if frame_idx < 0 or frame_idx >= total_frames:
                print(f"Warning: Frame index {frame_idx+1} out of bounds for video {video_name} (Total: {total_frames})")
                continue
            
            if not boxes:
                continue
                
            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                print(f"Error: Could not read frame {frame_idx+1} from {video_name}")
                continue
                
            # Generate base filename
            # Remove extension from video name
            video_basename = os.path.splitext(video_name)[0]
            # Format: video_name_frame_000123
            file_basename = f"{video_basename}_frame_{frame_idx+1:06d}"
            
            image_path = os.path.join(images_dir, f"{file_basename}.png")
            label_path = os.path.join(labels_dir, f"{file_basename}.txt")
            
            # Save original image (PNG is lossless)
            cv2.imwrite(image_path, frame)
            
            # Prepare debug image if needed
            debug_frame = frame.copy() if debug_mode else None
            
            # Save labels
            with open(label_path, "w", encoding="utf-8") as lf:
                for box in boxes:
                    if len(box) < 4:
                        continue
                        
                    # box is [x1, y1, x2, y2]
                    x_center, y_center, w, h = convert_to_yolo_format(box, img_width, img_height)
                    
                    # YOLO format: class_id center_x center_y width height
                    # Assuming class_id is 0 for all boxes
                    class_id = 0
                    lf.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
                    total_boxes_saved += 1
                    
                    if debug_mode:
                        # Draw box on debug frame using the calculated YOLO values
                        # to verify the conversion is correct
                        dx1, dy1, dx2, dy2 = convert_from_yolo_format([x_center, y_center, w, h], img_width, img_height)
                        cv2.rectangle(debug_frame, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)
                        cv2.putText(debug_frame, str(class_id), (dx1, dy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            if debug_mode:
                debug_image_path = os.path.join(debug_dir, f"{file_basename}.jpg")
                cv2.imwrite(debug_image_path, debug_frame)
            
            total_images_saved += 1
            
        cap.release()
        
    print("-" * 30)
    print(f"Export completed.")
    print(f"Images saved: {total_images_saved}")
    print(f"Boxes saved: {total_boxes_saved}")
    print(f"Output directory: {output_dir}")
    if debug_mode:
        print(f"Debug images saved to: {debug_dir}")

if __name__ == "__main__":
    main()
