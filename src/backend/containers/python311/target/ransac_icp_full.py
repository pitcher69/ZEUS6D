
import os
import json
import re
from ransac_icp import run_ransac_icp_single, angle_errors, trans_errors 
# === Folder paths ===
query_ply = "/content/data/query_5000_scaled.ply"
query_feat_npy = "/content/output/query_128.npy"
target_ply_dir = "/content/output/point_cloud"
target_feat_dir = "/content/fused_feature"
scene_gt_path = "/content/data/scene_gt.json"
save_dir = "/content/output/ransac"
def ransac_icp_full():    
    # === Collect all valid frame files ===
    frame_files = []
    for fname in os.listdir(target_ply_dir):
        if fname.startswith("target_") and fname.endswith(".ply"):
            match = re.search(r"target_(\d+)\.ply", fname)
            if match:
                fid = int(match.group(1))  # remove leading zeros
                frame_files.append((fid, fname))
    
    # === Sort by frame id numerically ===
    frame_files.sort()
    total = len(frame_files)
    
    # === Process each frame ===
    for idx, (fid, fname) in enumerate(frame_files, start=1):
        target_ply = os.path.join(target_ply_dir, fname)
        frame_id_str = f"{fid:06d}"  # format with leading zeros for filename
        target_feat_npy = os.path.join(target_feat_dir, f"target_{frame_id_str}_128.npy")
    
        if not os.path.exists(target_feat_npy):
            print(f"❌ Skipping frame {fid}: feature file not found.")
            continue
    
        try:
            run_ransac_icp_single(
                query_ply=query_ply,
                query_feat_npy=query_feat_npy,
                target_ply=target_ply,
                target_feat_npy=target_feat_npy,
                scene_gt_path=scene_gt_path,
                save_dir=save_dir
            )
            print(f"✓ [{idx}/{total}] Frame {fid} done")
        except Exception as e:
            print(f"⚠️ [{idx}/{total}] Frame {fid} failed: {e}")
    
    # === Final Average Error ===
    if angle_errors:
        avg_angle = sum(angle_errors) / len(angle_errors)
        avg_trans = sum(trans_errors) / len(trans_errors)
        print(f"\n✅ Processed {len(angle_errors)} valid frames")
        print(f"📊 Average Rotation Error: {avg_angle:.2f}°")
        print(f"📊 Average Translation Error: {avg_trans:.4f} m")
    else:
        print("⚠️ No valid frames processed.")
