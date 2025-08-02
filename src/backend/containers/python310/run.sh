#!/usr/bin/env bash

echo preparing env
pyenv local 3.10.13
python -m venv venv
source venv/bin/activate
git clone https://github.com/IRVLUTD/NIDS-Net
mkdir -p NIDS-Net/ckpts/sam_weights

echo curling weights
CHECKPOINT_PATH="NIDS-Net/ckpts/sam_weights/sam_vit_h_4b8939.pth"
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "downloading SAM ViT-H checkpoint..."
    curl -L https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -o "$CHECKPOINT_PATH"
    echo "download complete."
else
    echo "checkpoint already exists at $CHECKPOINT_PATH — skipping download."
fi

echo pip installs
pip install -r NIDS-Net/requirements.txt
pip install git+https://github.com/facebookresearch/detectron2.git
pip install torch opencv-python einops
cd NIDS-Net
python setup.py install

echo running code
mv ../mask_match.py .
python mask_match.py
cd ..
deactivate