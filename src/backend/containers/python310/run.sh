#!/usr/bin/env bash

echo preparing env
python -m venv venv
source venv/bin/activate
git clone https://github.com/IRVLUTD/NIDS-Net
cd NIDS-Net
mkdir -p ckpts/sam_weights

echo curling weights
CHECKPOINT_PATH="ckpts/sam_weights/sam_vit_h_4b8939.pth"
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "downloading SAM ViT-H checkpoint..."
    curl -L https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -o "$CHECKPOINT_PATH"
    echo "download complete."
else
    echo "checkpoint already exists at $CHECKPOINT_PATH — skipping download."
fi

echo pip installs
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/detectron2.git
python setup.py install
pip install einops

echo running code
cp ../mask_match.py .
python mask_match.py
deactivate