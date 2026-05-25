import numpy as np
import os
import pandas as pd
import sys
import glob
import warnings
import argparse
import cv2
from scipy.ndimage import median_filter
from scipy.integrate import trapezoid
from scipy.spatial.transform import Rotation as R_sci
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# --- CONFIGURATION ---
DATA_DIR = os.path.expanduser('~/data')
NPZ2D_SUBDIR = os.path.join(DATA_DIR, 'pose_estimation/npz_2D')
NPZ3D_SUBDIR = os.path.join(DATA_DIR, 'pose_estimation/npz_3D')
VIDEO_DIR = os.path.expanduser('~/data/input_vid')
REPORT_DIR = os.path.join(DATA_DIR, 'pose_estimation/evaluation_reports')
GT_DIR = os.path.join(DATA_DIR, 'pose_estimation', 'gt_data') # <-- NEW

# Evaluation Toggles
ENABLE_2D_EVAL = True
PCK_VALUE = 0.1
EXPECTED_VIDEO_COUNT = 100
VIS_EVAL_OPTIONS = [[1, 2, 3], [1], [1, 2], [2], [2, 3], [3]]

GT_MODEL_PREFIX = "ground_truth" 
MODEL_ORDER = ["mediapipe", "vit", "yolo", "hmr2", "sam3d", "BMPv2"]

OUTPUT_MODEL_NAMES = {
    "mediapipe": "MediaPipe",
    "vit": "ViTPose+",
    "yolo": "YOLOv8",
    "hmr2": "HMR 2.0",
    "sam3d": "SAM-3D",
    "bmpv2": "BMP v2"
}

COCO_JOINT_NAMES = [
    "Nose", "L_Eye", "R_Eye", "L_Ear", "R_Ear", 
    "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow", 
    "L_Wrist", "R_Wrist", "L_Hip", "R_Hip", 
    "L_Knee", "R_Knee", "L_Ankle", "R_Ankle"
]

# --- MAPPINGS: Target is COCO 17-Joint Format ---
COCO_IDENTITY_MAP = list(range(17))

MP_TO_COCO_MAP = [
    0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28
]

SAM3D_TO_COCO_MAP = [
    0,   # COCO 0: Nose       <- SAM 0
    1,   # COCO 1: L_Eye      <- SAM 1
    2,   # COCO 2: R_Eye      <- SAM 2
    3,   # COCO 3: L_Ear      <- SAM 3
    4,   # COCO 4: R_Ear      <- SAM 4
    5,   # COCO 5: L_Shoulder <- SAM 5
    6,   # COCO 6: R_Shoulder <- SAM 6
    7,   # COCO 7: L_Elbow    <- SAM 7
    8,   # COCO 8: R_Elbow    <- SAM 8
    62,  # COCO 9: L_Wrist    <- SAM 62
    41,  # COCO 10: R_Wrist   <- SAM 41
    9,   # COCO 11: L_Hip     <- SAM 9
    10,  # COCO 12: R_Hip     <- SAM 10
    11,  # COCO 13: L_Knee    <- SAM 11
    12,  # COCO 14: R_Knee    <- SAM 12
    13,  # COCO 15: L_Ankle   <- SAM 13
    14   # COCO 16: R_Ankle   <- SAM 14
]

HMR2_TO_COCO_MAP = [
    0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11
]

MODEL_MAPPINGS = {
    "sam3d": SAM3D_TO_COCO_MAP,
    "bmpv2": SAM3D_TO_COCO_MAP,
    "hmr2": HMR2_TO_COCO_MAP,
    "yolo": COCO_IDENTITY_MAP,
    "ViT": COCO_IDENTITY_MAP,
    "mediapipe": MP_TO_COCO_MAP, 
    "ground_truth": COCO_IDENTITY_MAP
}

MODEL_TYPES = {
    "sam3D": "3D", "BMPv2": "3D", "hmr2": "3D", "yolo": "2D",
    "mediapipe": "2D", "ViT": "2D", "ground_truth": "GT"
}

MODEL_NATIVE_JOINTS = {
    "sam3d": 70, "bmpv2": 70, "hmr2": 20, "yolo": 17,
    "vit": 17, "mediapipe": 33, "ground_truth": 17
}

# The Standard 13-Joint Benchmark Order
COMMON_JOINTS_3D = [
    "Neck", 
    "R_Shoulder", "R_Elbow", "R_Wrist", 
    "L_Shoulder", "L_Elbow", "L_Wrist",
    "R_Hip", "R_Knee", "R_Ankle", 
    "L_Hip", "L_Knee", "L_Ankle"
]


class KeypointConverter:
    @staticmethod
    def convert(points, model_prefix, allowed_vis_levels=None):
        mapping = MODEL_MAPPINGS.get(model_prefix.lower())
        if mapping is None:
            mapping = COCO_IDENTITY_MAP

        N_frames = points.shape[0]
        n_dims = points.shape[2]
        new_points = np.zeros((N_frames, 17, n_dims))
        valid_mask = np.zeros((N_frames, 17), dtype=bool)

        for coco_idx, source_idx in enumerate(mapping):
            if source_idx != -1 and source_idx < points.shape[1]:
                new_points[:, coco_idx, :] = points[:, source_idx, :]
                
                # Check spatial X, Y to prevent flags from causing false positives
                is_nonzero = np.any(points[:, source_idx, :2] != 0, axis=1)
                
                if model_prefix.lower() == "ground_truth" and allowed_vis_levels is not None:
                    if n_dims >= 3:
                        vis_flags = points[:, source_idx, 2]
                        valid_mask[:, coco_idx] = is_nonzero & np.isin(vis_flags, allowed_vis_levels)
                    else:
                        valid_mask[:, coco_idx] = is_nonzero
                else:
                    valid_mask[:, coco_idx] = is_nonzero

        return new_points, valid_mask

class CocoAPBridge:
    def __init__(self, coco_joint_names, report_dir, model_name):
        self.report_dir = report_dir
        self.model_name = model_name
        self.gt_dict = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "person", "supercategory": "person", "keypoints": coco_joint_names, "skeleton": []}]
        }
        self.pred_list = []
        self.annot_id_counter = 1

    def add_frame_data(self, image_id, gt_joints, pred_joints):
        # 1. Register Image
        self.gt_dict["images"].append({"id": image_id, "width": 1920, "height": 1080}) # Dimensions are required but arbitrary for OKS
        
        # 2. Format Ground Truth
        # Strip out zeros to find the bounding box for COCO Area
        valid_gt = gt_joints[np.any(gt_joints[:, :2] != 0, axis=1)]
        if len(valid_gt) > 0:
            min_vals = np.min(valid_gt[:, :2], axis=0)
            max_vals = np.max(valid_gt[:, :2], axis=0)
            w, h = max_vals - min_vals
            # Standard COCO approximation: Mask area is roughly 53% of BBox area
            area = float((w + 1e-4) * (h + 1e-4))
            bbox = [float(min_vals[0]), float(min_vals[1]), float(w), float(h)]
            
            coco_gt_kpts = []
            num_valid = 0
            for pt in gt_joints:
                x, y = float(pt[0]), float(pt[1])
                # COCO Visibility: 0 = missing, 1 = occluded, 2 = visible. 
                # We map any user flag > 0 (like 1 or 3) to 2 to ensure it is evaluated.
                v_raw = pt[2] 
                if v_raw == 0:
                    v = 0 # Not labeled
                elif v_raw == 2:
                    v = 2 # Visible
                elif v_raw == 1 or v_raw == 3:
                    v = 1 # Obscured/Occluded (Maps your 1 and 3 to COCO 1)
                else:
                    print("ERROR")
                if v > 0: num_valid += 1
                coco_gt_kpts.extend([x, y, v])
            
            self.gt_dict["annotations"].append({
                "id": self.annot_id_counter,
                "image_id": image_id,
                "category_id": 1,
                "bbox": bbox,
                "area": area,
                "keypoints": coco_gt_kpts,
                "num_keypoints": num_valid,
                "iscrowd": 0
            })
            self.annot_id_counter += 1
            
        # 3. Format Prediction
        valid_pred = pred_joints[np.any(pred_joints[:, :2] != 0, axis=1)]
        if len(valid_pred) > 0:
            coco_pred_kpts = []
            scores = []
            for pt in pred_joints:
                x, y = float(pt[0]), float(pt[1])
                conf = float(pt[2]) if len(pt) > 2 else 1.0 # Use extracted confidence
                coco_pred_kpts.extend([x, y, conf])
                if conf > 0: scores.append(conf)
            
            # Overall person score is the mean of their valid joint confidences
            overall_score = sum(scores)/len(scores) if scores else 0.0
            
            self.pred_list.append({
                "image_id": image_id,
                "category_id": 1,
                "keypoints": coco_pred_kpts,
                "score": overall_score
            })

    def evaluate(self):
        print(f"--- Running COCO AP Evaluation for {self.model_name.upper()} ---")
        gt_path = os.path.join(self.report_dir, f"temp_gt_{self.model_name}.json")
        pred_path = os.path.join(self.report_dir, f"temp_pred_{self.model_name}.json")
        
        with open(gt_path, 'w') as f: json.dump(self.gt_dict, f)
        with open(pred_path, 'w') as f: json.dump(self.pred_list, f)
        
        if not self.pred_list:
            print("No predictions found to evaluate.")
            return {"AP": np.nan, "AP@0.5": np.nan, "AP@0.75": np.nan, "AR": np.nan}
            
        try:
            coco_gt = COCO(gt_path)
            coco_dt = coco_gt.loadRes(pred_path)
            coco_eval = COCOeval(coco_gt, coco_dt, 'keypoints')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            stats = coco_eval.stats # stats[0]=AP, stats[1]=AP@0.5, stats[2]=AP@0.75, stats[5]=AR
            return {
                "AP (mAP)": stats[0] * 100,
                "AP@0.5": stats[1] * 100,
                "AP@0.75": stats[2] * 100,
                "AR (Recall)": stats[5] * 100
            }
        except Exception as e:
            print(f"COCO Eval Failed: {e}")
            return {"AP (mAP)": np.nan, "AP@0.5": np.nan, "AP@0.75": np.nan, "AR (Recall)": np.nan}
        
# 2D data --------------------------------------------------------------------

def safe_load_npz(filepath, alarms_list):
    try:
        return np.load(filepath, allow_pickle=True)
    except Exception as e:
        alarms_list.append(f"❌ File corrupted or unreadable: {filepath} | Details: {e}")
        return None

def extract_model_points(pred_data, model_nick, gt_frames_count, alarms_list, vid_name):
    native_joints = MODEL_NATIVE_JOINTS.get(model_nick.lower(), 17)
    
    # Extract Coordinates
    if 'j2ds' in pred_data:
        pred_raw = pred_data['j2ds']
    elif 'data' in pred_data:
        raw_objs = pred_data['data']
        pred_raw = np.array([f['results'][0]['pred_keypoints_2d'][:,:2] if f['results'] else np.zeros((native_joints, 2)) for f in raw_objs])
    else:
        alarms_list.append(f"[2D - {model_nick.upper()}] Pred missing 'j2ds' or 'data' key in {vid_name}")
        return None

    # --- EXTRACT AND STITCH CONFIDENCES ---
    # Look for the detached confidence arrays
    confs = None
    for conf_key in ['confs', 'confidences', 'vis', 'scores']:
        if conf_key in pred_data:
            confs = pred_data[conf_key]
            break
            
    if confs is not None:
        # If confs is 1D per joint (N, J), reshape it to (N, J, 1) and concatenate
        if len(confs.shape) == 2:
            confs = confs[..., np.newaxis]
        pred_raw = np.concatenate([pred_raw[..., :2], confs], axis=-1)
    else:
        # Fallback if no confidences exist: give a flat 1.0 to any non-zero coordinate
        fallback_confs = np.any(pred_raw[..., :2] != 0, axis=-1).astype(float)[..., np.newaxis]
        pred_raw = np.concatenate([pred_raw[..., :2], fallback_confs], axis=-1)

    # Length Mismatch Handling
    if pred_raw.shape[0] < gt_frames_count:
        alarms_list.append(f"[2D - {model_nick.upper()}] Missing {gt_frames_count - pred_raw.shape[0]} trailing frames in {vid_name}. Padding with zeros.")
        padding = np.full((gt_frames_count - pred_raw.shape[0], pred_raw.shape[1], pred_raw.shape[2]))
        pred_raw = np.concatenate([pred_raw, padding], axis=0)
    elif pred_raw.shape[0] > gt_frames_count:
        alarms_list.append(f"[2D - {model_nick.upper()}] Over-predicted {pred_raw.shape[0] - gt_frames_count} frames in {vid_name}. Truncating.")
        pred_raw = pred_raw[:gt_frames_count]

    return pred_raw

def discover_data(alarms_list):
    if not os.path.exists(REPORT_DIR):
        os.makedirs(REPORT_DIR)

    # 1. Discover Ground Truth Videos
    gt_search_pattern = os.path.join(NPZ2D_SUBDIR, f"{GT_MODEL_PREFIX}_video_*.npz")
    gt_files = glob.glob(gt_search_pattern)
    
    gt_videos = [os.path.basename(f).replace(f"{GT_MODEL_PREFIX}_", "").replace(".npz", "") for f in gt_files]
    
    gt_count = len(gt_videos)
    if gt_count != EXPECTED_VIDEO_COUNT:
        alarms_list.append(f"Expected {EXPECTED_VIDEO_COUNT} Ground Truth files, but found {gt_count}!")
    else:
        print(f"✅ Found all {EXPECTED_VIDEO_COUNT} Ground Truth videos.")

    if gt_count == 0:
        print("❌ CRITICAL: No ground truth files found. Exiting.")
        return

    all_files = os.listdir(NPZ2D_SUBDIR)
    discovered_models = set()
    for f in all_files:
        if f.endswith('.npz') and GT_MODEL_PREFIX not in f and "_video_" in f:
            discovered_models.add(f.split("_video_")[0])
            
    models = []
    known_models_lower = [m.lower() for m in MODEL_ORDER]
    for m in discovered_models:
        if m.lower() in known_models_lower:
            models.append(m)
        else:
            alarms_list.append(f"UNSTATED MODEL DETECTED: '{m}'. It has been excluded from output tables.")

    print(f"🔍 Discovered Models: {', '.join(models)}\n")
    print("-" * 50)

    if gt_count == 0:
        return [], [] # Return empty lists if no GT found

    return gt_videos, list(models)

def get_pck_bbox_thresholds(points, alarms_list=None, vid_name="", model_nick="GT"):
    """
    Calculates PCK threshold using the Maximum of the bbox dimensions.
    """
    N_frames = points.shape[0]
    thresholds = np.zeros(N_frames)
    
    for i in range(N_frames):
        frame_points = points[i, :, :2]
        valid_points = frame_points[np.any(frame_points != 0, axis=1)]
        
        if len(valid_points) < 2:
            thresholds[i] = 1.0 
            continue
            
        
        min_vals = np.min(valid_points, axis=0)
        max_vals = np.max(valid_points, axis=0)
        bbox_dims = max_vals - min_vals
        thresholds[i] = np.max(bbox_dims)
        
    has_points = np.any(points != 0, axis=(1, 2)) 
    is_collapsed = (thresholds == 0) & has_points
    if is_collapsed.any() and alarms_list is not None:
         alarms_list.append(f"[{model_nick}] BBox collapsed to 0.0 in {is_collapsed.sum()} predicted frames of {vid_name}.")
         
    thresholds[thresholds == 0] = 1.0 
    return thresholds

def calculate_auc(distances, thresholds, total_valid, max_multiplier=0.2, step=0.01):
    """
    Calculates Area Under the Curve (AUC) for PCK across discrete threshold steps.
    Standard evaluation protocol computes the trapezoidal integration of the PCK curve.
    """
    if total_valid == 0: 
        return np.nan
        
    # Generate discrete steps (e.g., 0.01, 0.02 ... 0.20)
    # Adding step/2 ensures the max_multiplier is inclusive against floating point errors
    t_steps = np.arange(0.0, max_multiplier + (step / 2), step)
    
    pck_scores = []
    for t in t_steps:
        # Calculate standard PCK at this specific threshold tier
        pck_t = (np.sum(distances < (thresholds * t)) / total_valid) * 100.0
        pck_scores.append(pck_t)
        
    # Normalize the x-axis so the domain is [0, 1]
    # This bounds the resulting AUC strictly between 0 and 100 (%)
    normalized_x = t_steps / max_multiplier 
    
    # Calculate the exact area under the discrete curve using standard trapezoidal rule
    auc_value = trapezoid(y=pck_scores, x=normalized_x)
    
    return auc_value
def calculate_pck(distances, thresholds, total_valid, multiplier):
    if total_valid == 0: return np.nan
    # Multiply the raw torso threshold by the target strictness (e.g., 0.2, 0.1, 0.05)
    return (np.sum(distances < (thresholds * multiplier)) / total_valid) * 100.0

def compile_model_metrics(model_nick, model_pck_dists, model_thresholds, model_masks, model_original_preds_attempted):
    total_gt_valid = model_masks.sum()
    total_original_pred_valid = model_original_preds_attempted.sum()
    
    if total_gt_valid == 0:
        return None, []

    m_type = MODEL_TYPES.get(model_nick, "Unknown")
    
    

    # PCK is evaluated on ALL valid GT frames (failed frames are hardcoded to 9999.0)
    flat_pck_dists = model_pck_dists[model_masks]
    
    overall_result = {
        "Model": model_nick, "Type": m_type, "Joint": "OVERALL",
        "Det. Rate (%)": (total_original_pred_valid / total_gt_valid) * 100.0,
        "PCK@0.10 (%)": calculate_pck(flat_pck_dists, model_thresholds[model_masks], total_gt_valid, 0.1),
        "PCK@0.05 (%)": calculate_pck(flat_pck_dists, model_thresholds[model_masks], total_gt_valid, 0.05),
        "AUC (%)": calculate_auc(flat_pck_dists, model_thresholds[model_masks], total_gt_valid, max_multiplier=PCK_VALUE, step=0.01)
    }

    joint_results = []
    for j in range(17):
        mask_j = model_masks[:, j]
        valid_gt_j = mask_j.sum()
        
        if valid_gt_j > 0:
            pck_dists_j = model_pck_dists[mask_j, j]
            thresh_j = model_thresholds[mask_j, j] 
            
            # Det rate uses the original mask
            original_preds_j_mask = model_original_preds_attempted[:, j]
            det_rate_j = (original_preds_j_mask[mask_j].sum() / valid_gt_j) * 100.0
            
            
            pck_10_j = calculate_pck(pck_dists_j, thresh_j, valid_gt_j, 0.1)
            pck_05_j = calculate_pck(pck_dists_j, thresh_j, valid_gt_j, 0.05)
            auc_j = calculate_auc(pck_dists_j, thresh_j, valid_gt_j, max_multiplier=PCK_VALUE, step=0.01)
            
        else:
            det_rate_j, auc_j, = np.nan, np.nan
            pck_10_j, pck_05_j = np.nan, np.nan

        joint_results.append({
            "Model": model_nick, "Type": m_type, "Joint": COCO_JOINT_NAMES[j],
            "Det. Rate (%)": det_rate_j, 
            "PCK@0.10 (%)": pck_10_j, 
            "PCK@0.05 (%)": pck_05_j, 
            "AUC (%)": auc_j
        })
        
    return overall_result, joint_results

def evaluate_models(gt_videos, models, alarms_list, vis_levels):
    results_table = []
    joint_results_table = []

    for model_nick in models:
        print(f"⚙️  Evaluating Model: [{model_nick.upper()}]")
        coco_bridge = CocoAPBridge(COCO_JOINT_NAMES, REPORT_DIR, model_nick) if vis_levels == [1, 2, 3] else None
        
        model_thresholds, model_masks, model_preds_attempted = [], [], []

        all_pck_dists, all_thresholds, all_masks, all_preds_attempted=[], [], [], []

        for vid_idx, vid_name in enumerate(gt_videos):
            gt_path = os.path.join(NPZ2D_SUBDIR, f"{GT_MODEL_PREFIX}_{vid_name}.npz")
            pred_path = os.path.join(NPZ2D_SUBDIR, f"{model_nick}_{vid_name}.npz")

            if not os.path.exists(pred_path):
                alarms_list.append(f"[2D - {model_nick.upper()}] Missing Pred file: {pred_path}. Skipping.")
                continue

            gt_data = safe_load_npz(gt_path, alarms_list)
            pred_data = safe_load_npz(pred_path, alarms_list)
            if gt_data is None or pred_data is None:
                alarms_list.append(f"[2D - {model_nick.upper()}] Missing Pred data or gt data: {pred_path}. Skipping.")
                continue

            gt_j2d_raw = gt_data.get('j2ds', gt_data.get('results'))
            if gt_j2d_raw is None:
                alarms_list.append(f"[2D - GT] Missing 'j2ds' or 'results' key in GT file for {vid_name}. Skipping.")
                continue

            
       
            gt_j2d_smpl, gt_valid_mask = KeypointConverter.convert(gt_j2d_raw, "ground_truth", allowed_vis_levels=vis_levels)
            

            if gt_valid_mask.sum() == 0:
                alarms_list.append(f"[2D - GT] No GT joints matching vis_levels {vis_levels} for {vid_name}.")
                continue

            gt_2d = gt_j2d_smpl[:,:,:2]

            gt_frames_count = gt_j2d_raw.shape[0]
            
            # Ensure PCK BBox uses the full skeleton shape to avoid artificial metric inflation


            pred_j2d_raw = extract_model_points(pred_data, model_nick, gt_frames_count, alarms_list, vid_name)
            if pred_j2d_raw is None: 
                alarms_list.append(f"[2D - Pred] Missing 'j2ds' or 'results' key in pred file for {vid_name}, model {model_nick}. Skipping.")                
                continue
            pred_j2d_smpl, original_pred_valid_mask = KeypointConverter.convert(pred_j2d_raw, model_nick)
            pred_2d = pred_j2d_smpl[:,:,:2]

            #AP
            if vis_levels == [1, 2, 3]:
                for frame_idx in range(gt_frames_count):
                    if gt_valid_mask[frame_idx].sum() > 0:
                        image_id = (vid_idx + 1) * 100000 + frame_idx
                        
                        # Pass the raw 3D array (which now contains X, Y, and the Confidence/Vis flags we stitched)
                        coco_bridge.add_frame_data(image_id, gt_j2d_smpl[frame_idx], pred_j2d_smpl[frame_idx])

            
            thresholds = get_pck_bbox_thresholds(gt_2d, alarms_list, vid_name, "GT")
            
            # 3. PCK: Strict 9999.0 failure penalty based on the RAW distances
            
            pck_dists = np.linalg.norm(gt_2d - pred_2d, axis=2)
            pck_dists[~original_pred_valid_mask] = 9999.0

            # ALARM: Check if the model completely failed the video
            if original_pred_valid_mask.sum() == 0:
                alarms_list.append(f"[2D - {model_nick.upper()}] Model output 0 valid joints across all frames in {vid_name}.")

            all_pck_dists.append(pck_dists)
            all_thresholds.append(thresholds.reshape(-1, 1) * np.ones((1, 17))) 
            all_masks.append(gt_valid_mask)
            
            # Det Rate relies strictly on where GT is valid AND the model originally attempted it
            all_preds_attempted.append(gt_valid_mask & original_pred_valid_mask)

        if not all_masks: 
            alarms_list.append(f"[2D - {model_nick.upper()}] CRITICAL: Model yielded no overlapping GT frames across the entire dataset. Skipping compilation.")
            continue

        model_pck_dists = np.concatenate(all_pck_dists, axis=0)
        model_thresholds = np.concatenate(all_thresholds, axis=0)
        model_masks = np.concatenate(all_masks, axis=0)
        model_preds_attempted = np.concatenate(all_preds_attempted, axis=0)

        ap_metrics = coco_bridge.evaluate() if vis_levels == [1, 2, 3] else {}
        overall_res, joint_res = compile_model_metrics(
            model_nick, model_pck_dists, model_thresholds, model_masks, model_preds_attempted
        )
        
        if overall_res:
            overall_res.update(ap_metrics)
            results_table.append(overall_res)
            joint_results_table.extend(joint_res)

    return results_table, joint_results_table

def load_timing_data():
    """Loads timing data assuming CSV model names match internal script names."""
    timing_path = os.path.join(REPORT_DIR, "timing_results_video_001.csv")
    if not os.path.exists(timing_path):
        return {}
    
    df_time = pd.read_csv(timing_path)
    # Convert model names to lowercase to ensure matching regardless of capitalization
    return {str(row['Model']).lower(): {'ms': row['Avg Time per Frame (ms)'], 'fps': row['Avg FPS']} 
            for _, row in df_time.iterrows()}

def apply_academic_bolding(df, lower_is_better_metrics=None):
    """
    Finds the best value in each column and wraps it in LaTeX \textbf{}.
    Higher is better by default, unless the column name matches 'lower_is_better_metrics'.
    """
    lower_is_better_metrics = lower_is_better_metrics or []
    df_out = df.copy()
    
    for col in df.columns:
        # Extract the string name (handles both standard and MultiIndex columns)
        col_label = col[-1] if isinstance(col, tuple) else col
        
        # Skip string identifier columns
        if col_label == 'Model' or col_label == '':
            continue
            
        # Determine if we are looking for the Minimum or Maximum
        is_lower_better = any(metric in col_label for metric in lower_is_better_metrics)
        
        # Convert column to float (ignoring '-' strings)
        s_num = pd.to_numeric(df[col], errors='coerce')
        if s_num.isna().all():
            df_out[col] = "-"
            continue
            
        target_val = s_num.min() if is_lower_better else s_num.max()
        
        # Format the column with bolding
        formatted_col = []
        for num in s_num:
            if pd.isna(num):
                formatted_col.append("-")
            elif num == target_val:
                formatted_col.append(f"\\textbf{{{num:.2f}}}")
            else:
                formatted_col.append(f"{num:.2f}")
                
        df_out[col] = formatted_col
        
    return df_out

def generate_reports(results_table, joint_results_table, alarms_list):
    if not results_table:
        return

    timing_data = load_timing_data()

    df_overall = pd.DataFrame(results_table)
    df_joints = pd.DataFrame(joint_results_table)
    
    # 1. Sort Enforcements & Fix Categorical Warning
    known_models = [m.lower() for m in MODEL_ORDER]
    df_overall['Model_Norm'] = df_overall['Model'].str.lower().replace('hrm2', 'hmr2')
    df_joints['Model_Norm'] = df_joints['Model'].str.lower().replace('hrm2', 'hmr2')

    for norm_name in df_overall['Model_Norm'].unique():
        if norm_name not in known_models:
            alarms_list.append(f"Model '{norm_name}' is not in MODEL_ORDER. Pushing to bottom of table.")
            known_models.append(norm_name)

    sort_cat = pd.CategoricalDtype(categories=known_models, ordered=True)
    df_overall['SortKey'] = df_overall['Model_Norm'].astype(sort_cat)
    df_overall = df_overall.sort_values('SortKey').drop(columns=['SortKey', 'Model_Norm'])

    df_joints['SortKey'] = df_joints['Model_Norm'].astype(sort_cat)
    df_joints = df_joints.sort_values(['SortKey', 'Joint']).drop(columns=['SortKey', 'Model_Norm'])
    
    # Drop AP metrics from joint table
    ap_cols = ["AP (mAP)", "AP@0.5", "AP@0.75", "AR (Recall)"]
    df_joints_clean = df_joints.drop(columns=[col for col in ap_cols if col in df_joints.columns], errors='ignore')

    tables_dir = os.path.join(REPORT_DIR, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    # --- Timing Injection Helper ---
    def get_timing(model_val, metric):
        m_lower = str(model_val).lower().replace('hrm2', 'hmr2')
        if m_lower not in timing_data:
            msg = f"[Timing 2D] Missing CSV data for model: '{model_val}'"
            if msg not in alarms_list:
                alarms_list.append(msg)
            return '-'
        return timing_data[m_lower].get(metric, '-')

    # --- 1. Overall 2D Performance ---
    df_o2d = df_overall[df_overall.get('Visibility', '1,2,3') == '1,2,3'].copy()
    df_o2d['FPS'] = df_o2d['Model'].apply(lambda x: get_timing(x, 'fps'))
    
    df_o2d = df_o2d.rename(columns={
        'Det. Rate (%)': r'Det. (\%)', 'AP (mAP)': 'mAP', 
        'PCK@0.05 (%)': 'PCK@0.05', 'AUC (%)': 'AUC', 'AR (Recall)': 'AR'
    })
    
    cols_2d = ['Model', r'Det. (\%)', 'PCK@0.05', 'AUC', 'mAP', 'AP@0.5', 'AP@0.75', 'AR', 'FPS']
    cols_2d = [c for c in cols_2d if c in df_o2d.columns]

    # Map output names
    df_o2d['Model'] = df_o2d['Model'].apply(lambda x: OUTPUT_MODEL_NAMES.get(str(x).lower(), x))
    # Apply bolding (all 2D metrics here are Higher=Better)
    df_o2d_styled = apply_academic_bolding(df_o2d[cols_2d])
    with open(os.path.join(tables_dir, "eval_2D_overall.tex"), "w") as f:
        f.write(df_o2d_styled.style.hide(axis="index").to_latex(hrules=True))

    # --- 2. 2D Visibility Degradation ---
    if 'Visibility' in df_overall.columns:
        df_vis_list = []
        # Reordered and renamed to drop the custom COCO flags: 
        # 2 = Visible, 1 = Easy Occlusion, 3 = Hard Occlusion
        visibility_mappings = [
            ('2', 'Visible'), 
            ('3', 'Easy Occlusion'), 
            ('1', 'Hard Occlusion')
        ]
        
        for vis, name in visibility_mappings:
            # Extract and rename columns to drop the (%) for the final table width
            sub = df_overall[df_overall['Visibility'] == vis][['Model', 'PCK@0.05 (%)', 'AUC (%)']].set_index('Model')
            sub = sub.rename(columns={'PCK@0.05 (%)': 'PCK@0.05', 'AUC (%)': 'AUC'})
            sub.columns = pd.MultiIndex.from_product([[name], ['PCK@0.05', 'AUC']])
            df_vis_list.append(sub)
        
        if df_vis_list:
            df_vis_combined = pd.concat(df_vis_list, axis=1).reset_index()
            df_vis_combined.columns = pd.MultiIndex.from_tuples([('Model', '')] + df_vis_combined.columns.tolist()[1:])
            
            # Map output names for multi-index
            df_vis_combined[('Model', '')] = df_vis_combined[('Model', '')].apply(lambda x: OUTPUT_MODEL_NAMES.get(str(x).lower(), x))
            
            # Apply bolding
            df_vis_styled = apply_academic_bolding(df_vis_combined)
            with open(os.path.join(tables_dir, "eval_2D_visibility.tex"), "w") as f:
                f.write(df_vis_styled.style.hide(axis="index").to_latex(hrules=True, multicol_align="c"))

    # --- 3. 2D Kinematic Joints ---
    df_j2d = df_joints_clean[df_joints_clean.get('Visibility', '1,2,3') == '1,2,3'].copy()
    joint_pairs = {'Shoulders': ['L_Shoulder', 'R_Shoulder'], 'Elbows': ['L_Elbow', 'R_Elbow'], 
                   'Wrists': ['L_Wrist', 'R_Wrist'], 'Hips': ['L_Hip', 'R_Hip'], 
                   'Knees': ['L_Knee', 'R_Knee'], 'Ankles': ['L_Ankle', 'R_Ankle']}
    
    j2d_rows = []
    for model in df_j2d['Model'].unique():
        row = {'Model': model}
        m_df = df_j2d[df_j2d['Model'] == model]
        for paired_name, j_list in joint_pairs.items():
            vals = m_df[m_df['Joint'].isin(j_list)]['PCK@0.05 (%)']
            row[paired_name] = vals.mean() if not vals.empty else float('nan')
        j2d_rows.append(row)
    
    if j2d_rows:
        df_j2d_matrix = pd.DataFrame(j2d_rows)
        
        # Map output names
        df_j2d_matrix['Model'] = df_j2d_matrix['Model'].apply(lambda x: OUTPUT_MODEL_NAMES.get(str(x).lower(), x))
        
        # Apply bolding
        df_j2d_styled = apply_academic_bolding(df_j2d_matrix)
        with open(os.path.join(tables_dir, "eval_2D_joints.tex"), "w") as f:
            f.write(df_j2d_styled.style.hide(axis="index").to_latex(hrules=True))

    print(f"\n📁 2D LaTeX tables strictly saved to: {tables_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate 2D and 3D pose estimation models on Climbing dataset.")
    
    # Evaluation Toggles (Default to ON, flag turns them OFF)
    parser.add_argument('--skip_2d', action='store_true', help='Skip the 2D evaluation pipeline.')
    
    return parser.parse_args()

def main():
    
    args = parse_args()
    if args.skip_2d and args.skip_3d:
        print("🛑 ERROR: Both 2D and 3D evaluations were skipped. Nothing to do.")
        return
    
    alarms_list = []

    if not os.path.exists(REPORT_DIR):
        os.makedirs(REPORT_DIR)

    if not args.skip_2d:
        print("\n--- STARTING 2D EVALUATION ---")
        gt_videos, models = discover_data(alarms_list)
        
        if not gt_videos:
            print("❌ CRITICAL: No ground truth files found. Exiting.")
        else:
            all_results = []
            all_joint_results = []
            
            for vis_opt in VIS_EVAL_OPTIONS:
                print(f"\n>> Evaluating subset for Visibility Levels: {vis_opt}")
                res, j_res = evaluate_models(gt_videos, models, alarms_list, vis_levels=vis_opt)
                
                # Tag the results with the visibility subset so it appears in the final CSV/LaTeX
                vis_str = ",".join(map(str, vis_opt))
                for r in res: r['Visibility'] = vis_str
                for r in j_res: r['Visibility'] = vis_str
                
                all_results.extend(res)
                all_joint_results.extend(j_res)
                
            generate_reports(all_results, all_joint_results, alarms_list)


    # --- UNIFIED ALARM OUTPUT ---
    if alarms_list:
        RED = '\033[91m'
        RESET = '\033[0m'
        print("\n" + RED + "!"*80)
        print("🚨 🚨 🚨 CRITICAL ALARMS DETECTED DURING EVALUATION 🚨 🚨 🚨".center(80))
        print("!"*80)
        # Convert to a set to remove duplicate missing file warnings if they repeat
        for alarm in sorted(set(alarms_list)):
            print(f"❌  {alarm}")
        print("!"*80 + RESET + "\n")
    

if __name__ == "__main__":
    main()
    
