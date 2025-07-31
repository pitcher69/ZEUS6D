#!/usr/bin/env bash

echo preparing env
python -m venv .
source bin/activate
git clone https://github.com/IRVLUTD/NIDS-Net
cd NIDS-Net
mkdir -p ckpts/sam_weights

echo curling weights
curl https://huggingface.co/HCMUE-Research/SAM-vit-h/resolve/main/sam_vit_h_4b8939.pth?download=true --output ckpts/sam_weights/sam_vit_h_4b8939.pth

echo pip installs
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/detectron2.git
python setup.py install

echo running code
cp ../mask_match.py .
python mask_match.py

