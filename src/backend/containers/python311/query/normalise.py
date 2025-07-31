import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
def query_normalise():
    # Load original DINO features
    query_dino = np.load("output/query_5000_dino.npy")
    print("Shape of feature vector", query_dino.shape)
    
    # Apply PCA to reduce to 64 dimensions
    pca = PCA(n_components=64)
    reduced_query = pca.fit_transform(query_dino)
    
    # ✅ Apply L2 normalization (row-wise)
    normalized_query = normalize(reduced_query, norm='l2', axis=1)
    
    # Save the normalized, reduced feature
    np.save("output/query_pca64.npy", normalized_query)
    
    print("✅ Reduced and normalized feature shape:", normalized_query.shape)
