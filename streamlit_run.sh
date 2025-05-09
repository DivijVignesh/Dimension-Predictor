#!/bin/bash
echo "Installing MobileSAM dependencies..."
pip install --upgrade pip
pip install torch==2.0.1 torchvision==0.15.2
pip install timm==0.9.2 PyYAML==6.0
pip install opencv-python-headless==4.8.0.74
pip install protobuf==3.20.3
pip install git+https://github.com/ChaoningZhang/MobileSAM.git

echo "Downloading model checkpoint..."
wget -q https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt

echo "Starting Streamlit app..."
streamlit run app.py