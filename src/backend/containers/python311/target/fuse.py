import os
import numpy as np

def fuse_features_from_paths(gedi_path, dino_pca_path,
                             output_fused_dir="output/fused_feature"):
    os.makedirs(output_fused_dir, exist_ok=True)

    # === Load features
    gedi = np.load(gedi_path)
    dino = np.load(dino_pca_path)

    # === Sanity check
    assert gedi.shape == (1000, 64), f"{gedi_path} shape mismatch"
    assert dino.shape == (1000, 64), f"{dino_pca_path} shape mismatch"

    # === Extract padded frame ID from file name
    frame_id = os.path.basename(gedi_path).split("_")[1]  # "000620" from "target_000620_gedi.npy"

    # === Concatenate
    fused = np.concatenate([gedi, dino], axis=1)  # → (1000, 128)

    # === Save
    save_path = os.path.join(output_fused_dir, f"target_{frame_id}_128.npy")
    np.save(save_path, fused)
    print(f"✅ Saved {os.path.basename(save_path)} with shape:", fused.shape)

    return save_path

import os
from glob import glob

def fuse(
    gedi_root="output/gedi",
    dino_root="output/dino_feature_pca",
    output_dir="fused_feature"
):
    from tqdm import tqdm
    os.makedirs(output_dir, exist_ok=True)

    gedi_files = sorted(glob(os.path.join(gedi_root, "target_*_gedi.npy")))
    total = len(gedi_files)

    for i, gedi_path in enumerate(gedi_files, 1):
        frame_id = os.path.basename(gedi_path).split("_")[1]  # e.g., "000620"
        dino_path = os.path.join(dino_root, f"target_{frame_id}_pca64.npy")

        if not os.path.exists(dino_path):
            print(f"❌ Missing DINO file for frame {frame_id}, skipping...")
            continue

        fuse_features_from_paths(gedi_path, dino_path, output_dir)
        print(f"✅ [{i}/{total}] Fused frame {frame_id}")
