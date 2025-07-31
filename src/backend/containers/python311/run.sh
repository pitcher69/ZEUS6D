#!/usr/bin/env bash

echo SETTING UP ENV
mkdir -p env
python3 -m venv env
source env/bin/activate
pip install pyrender trimesh Pillow tqdm numpy matplotlib open3d




echo DOWNLOADING DATA
cd input
./download.sh


cd ..
cur_dir="$(pwd)"

cd ../../cnos/src/poses
mv "$cur_dir/input/data" .
wget -O ./data/model.ply "https://github.com/pitcher69/IITISOC/raw/refs/heads/main/DATA/mustard/model.ply"
wget -O ./data/obj_000005.png "https://raw.githubusercontent.com/pitcher69/IITISOC/main/DATA/mustard/obj_000005.png"

mkdir -p output


echo RUNNING CNOS
python ./generate_views.py ./data/model.ply ./predefined_poses/obj_poses_level0.npy ./output/renders 0 False 1 0.35


echo QUERY PIPELINE
cur_dir="./$cur_dir/query"
mv ./data "./$cur_dir"
mv ./output "./$cur_dir"
cd "./$cur_dir"
python3 ./main.py




echo TARGET PIPELINE
mv ./data ../target
cd ../target
python3 ./main.py



echo POSE ESTIMATION
mv ./data ../pose
cd ../pose
python3 ./main.py
