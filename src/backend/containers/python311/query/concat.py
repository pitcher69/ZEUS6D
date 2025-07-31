import numpy as np

def concat():
    # === Load features ===
    gedi_0 = np.load("/content/fused_query_gedi (3).npy")
    dino_64 = np.load("/content/output/query_pca64.npy")
    # === Sanity check ===
    assert gedi_0.shape == (5000, 64)
    assert dino_64.shape == (5000, 64)
    
    # === Concatenate ===
    fused_128 = np.concatenate([gedi_0, dino_64], axis=1)
    np.save("/content/output/query_128.npy", fused_128)
    print("✅ Saved query_128.npy with shape:", fused_128.shape)
