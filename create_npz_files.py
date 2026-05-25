import json
import numpy as np
import os
import argparse
import cv2
from collections import defaultdict

# --- CONFIGURATION ---
JSON_PATH = "annotations/person_keypoints_val2017_kpts.json"
VIDEO_DIR = "../../videos/input_20fps"
DEFAULT_OUTPUT_DIR = "npz_output" 
GT_PREFIX = "ground_truth"
FPS = 20 

def parse_filename(filename):
    """
    Parses 'video_030_005.jpg' -> ('video_030', 5)
    """
    base = os.path.splitext(filename)[0]
    parts = base.split('_')
    
    if len(parts) != 3:
        raise ValueError(f"CRITICAL ERROR: Unexpected filename format: '{filename}'")
    
    video_name = f"{parts[0]}_{parts[1]}"
    second_mark = int(parts[2])
        
    return video_name, second_mark

def get_video_frame_count(video_path):
    """
    Reads the exact frame count from the mp4 file metadata.
    """
    if not os.path.exists(video_path):
        return None
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
        
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return length

def main():
    parser = argparse.ArgumentParser(description="Convert 1fps COCO JSON to 20fps sparse NPZ.")
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUTPUT_DIR, 
                        help="Output directory for .npz files")
    parser.add_argument("--max-annotations", type=int, default=None,
                        help="Optional: Limit processing to the first N images.")
    args = parser.parse_args()

    if not os.path.exists(JSON_PATH):
        raise FileNotFoundError(f"CRITICAL ERROR: JSON file not found at {JSON_PATH}")

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    print(f"📖 Loading annotations from: {JSON_PATH}")
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)

    # 0. Determine global keypoint shape upfront
    if 'annotations' not in data or len(data['annotations']) == 0:
        raise ValueError("CRITICAL ERROR: No 'annotations' found in JSON.")
    
    # Extract shape from the very first annotation in the dataset
    sample_kpts = np.array(data['annotations'][0]['keypoints']).reshape(-1, 3)
    n_joints, n_dims = sample_kpts.shape

    # 1. Apply Optional Image Limit
    image_list = data['images']
    if args.max_annotations is not None:
        image_list = image_list[:args.max_annotations]
        print(f"⚠️  Limit applied: Processing only the first {args.max_annotations} images.")

    # 2. Map Images to Seconds
    video_groups = defaultdict(list)
    for img in image_list:
        vid_name, second_mark = parse_filename(img['file_name'])
        video_groups[vid_name].append({'second_mark': second_mark, 'id': img['id']})

    # 3. Map Annotations
    #ann_lookup = {ann['image_id']: np.array(ann['keypoints']).reshape(-1, 3) for ann in data['annotations']}

    ann_lookup = {
        ann['image_id']: {
            'keypoints': np.array(ann['keypoints']).reshape(-1, 3),
            'bbox': np.array(ann['bbox'])
        } 
        for ann in data['annotations']
    }

    print(f"🔍 Found {len(video_groups)} unique videos in the selected batch.")

    # 4. Process Videos
    saved_count = 0
    skipped_empty_count = 0
    
    for vid_name, frames in video_groups.items():
        vid_path = os.path.join(VIDEO_DIR, f"{vid_name}.mp4")
        total_frames = get_video_frame_count(vid_path)
        
        if total_frames is None:
            raise FileNotFoundError(f"CRITICAL ERROR: Missing source video file at '{vid_path}'.")

        # Initialize full-length timeline. By default, these are the "empty frames".
        full_timeline_kpts = np.zeros((total_frames, n_joints, n_dims))
        full_timeline_bboxes = np.zeros((total_frames, 4))
        annotated_count = 0
        
        for frame in frames:
            second_mark = frame['second_mark']
            img_id = frame['id']
            
            target_frame_idx = second_mark * FPS
            
            if img_id in ann_lookup:
                if target_frame_idx < total_frames:
                    full_timeline_kpts[target_frame_idx] = ann_lookup[img_id]['keypoints']
                    full_timeline_bboxes[target_frame_idx] = ann_lookup[img_id]['bbox']
                    annotated_count += 1
                else:
                    raise IndexError(f"CRITICAL BOUNDARY ERROR on '{vid_name}': Target frame {target_frame_idx} exceeds actual video length of {total_frames} frames.")

        # Do not create NPZ if there are no annotations for this video in the current slice
        if annotated_count == 0:
            print(f"⏭️  Skipped {vid_name:<15} | Reason: 0 annotations in this batch.")
            skipped_empty_count += 1
            continue

        out_filename = f"{GT_PREFIX}_{vid_name}.npz"
        out_path = os.path.join(args.out_dir, out_filename)
        
        np.savez_compressed(out_path, j2ds=full_timeline_kpts, bboxes=full_timeline_bboxes)
        print(f"✅ {vid_name:<15} | Target Length: {total_frames:<5} | Placed Kpts: {annotated_count:<3}")
        saved_count += 1

    print(f"\n🎉 Done! Created {saved_count} NPZ files. Skipped {skipped_empty_count} empty files.")

if __name__ == "__main__":
    main()