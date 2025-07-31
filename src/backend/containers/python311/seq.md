# starts with
- input
- download.py
- !wget -O data/model.ply "https://github.com/pitcher69/IITISOC/raw/refs/heads/main/DATA/mustard/model.ply"
- !wget -O data/obj_000005.png "https://raw.githubusercontent.com/pitcher69/IITISOC/main/DATA/mustard/obj_000005.png"

- query
- python /content/cnos/src/poses/generate_views.py /content/data/obj_000015.ply /content/cnos/src/poses/predefined_poses/obj_poses_level0.npy /content/output/renders 0 False 1 0.35
- views.py
- genpc.py
- features.py
- normalise.py
- concat.py


- target
- genpc.py
- dino.py
- dino_pca.py

!zip -r /content/point_clouds.zip /content/output/point_cloud
!zip -r /content/dino_feature_pca.zip /content/output/dino_feature_pca
# from google.colab import files
# files.download("/content/point_clouds.zip")
# files.download("/content/dino_feature_pca.zip")
- fuse.py
# !zip -r /content/fused_feature.zip /content/fused_feature
# from google.colab import files
# files.download("/content/fused_feature.zip")
- ransac_icp.py
- ransac_icp_full.py
- process.py
# !zip -r /content/matrix.zip /content/output/ransac
# from google.colab import files
# files.download("/content/matrix.zip")
