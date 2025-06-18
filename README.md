# Object Dimension Prediction

##Overview

The goal of this project is to create a computer vision system that can predict the real-world dimensions (height, width, and length) of a object from a single  image. This system uses segmentation techniques and deep learning models without needing any additional depth information. The key innovation of this approach is the dual-input which processes both the original image and the segmented image of the primary object, allowing the model to learn from the background ques which will be available in the original image.

## Business Objectives

Many industries require accurate object dimension estimation for inventory management, quality control, and automated sorting processes. Traditional systems uses sophisticated setup consisting of multiple cameras and maybe some complicated setup. These kind of setups might be difficult to manage and sometimes costly. Smaller businesses might not be able to afford such systems. To address this particular issue I have created a dimension prediction system that can estimate the real world dimensions upto to good level of accuracy.

## Tools Used

- MobileSAM (Segment Anything Model variant for object segmentation)
- MobileNetV3 Large (lightweight CNN backbone for feature extraction)
- PyTorch
- OpenCV
- Streamlit (web interface)


## Methodology

This entire network has two major components working together. The heart of this system is the dual-input neural network architecture which processes both original and segmented images simultaneously. MobileSAM is used for automatically identifying and isolating the primary object in the image without requiring any user input. General assumption here is that the image consists of a single object in the foreground.

Since the goal is not only to create an accurate dimension predictor but also an efficient one that can run on mobile devices, the system uses MobileNetV3 as the primary feature extractor. This network should be lightweight enough for real-time inference while maintaining high accuracy. The segmentation module generates multiple potential object masks based on confidence scores and selects the largest coherent mask as the primary object of interest.

The dual input approach is important because if we only use the segment object, then we will loose the background contextual information like scale cues. The same MobileNetV3 processes both images seperately to generate extracted features. These extracted features are concatenated to preserve both object and contextual information.

The training process involves several optimization techniques to achieve better performance. The model uses Mean Squared Error(MSE) loss for dimension regression ,AdamW optimizer with weight decay to prevent overfitting of the model and OneCycleLR scheduling for achieving faster convergence.

## Architecture

- The network processes images of size 224×224 and creates 2560-dimension combined feature representation through feature concatenation
- Memory optimization includes dataset caching, GPU memory clearing between epochs and gradient optimization has improved training efficiency
- The regression head architecture used in this architecture (2560 to 256 to 128 to 3) provided stable convergence during training


## Impact

This project has given me a significant learning experience in training a complex neural network combining multiple state of the art models. 


## Technologies Used

Python, PyTorch, OpenCV, NumPy, Matplotlib, timm, MobileSAM, Streamlit

## Files

main.py - main streamlit file

prep_dataset.py - Creates dataset for training

train.py - Contains training Code

