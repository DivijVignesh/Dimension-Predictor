import os
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
try:
    import cv2
except ImportError:
    # Try installing opencv-python-headless which works better on cloud platforms
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless"])
    import cv2

from io import BytesIO
import requests
import tempfile
from pathlib import Path

# Check if GPU is available and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set global variables for faster access
IMG_SIZE = 224  # Image size

# Import required packages for MobileSAM
@st.cache_resource
def install_dependencies():
    """Install required dependencies for MobileSAM"""
    import subprocess
    try:
        import mobile_sam
    except ImportError:
        subprocess.check_call(["pip", "install", "timm", "PyYAML"])
        subprocess.check_call(["pip", "install", "git+https://github.com/ChaoningZhang/MobileSAM.git"])

# SAM2 segmentation functions
@st.cache_resource
def initialize_sam():
    """Initialize SAM model for inference"""
    from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator

    # Path to save the model locally
    model_path = Path("mobile_sam.pt")
    
    # Download the model if not present
    if not model_path.exists():
        st.write("Downloading MobileSAM model...")
        url = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
        response = requests.get(url)
        with open(model_path, 'wb') as f:
            f.write(response.content)
        st.write("Download complete!")

    # Load model in full precision for inference
    model_type = "vit_t"
    sam = sam_model_registry[model_type](checkpoint=str(model_path))
    sam.to(device=device)
    sam.eval()

    # Configure mask generator with inference parameters
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=16,
        pred_iou_thresh=0.8,
        stability_score_thresh=0.9,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100
    )

    return mask_generator

@torch.no_grad()
def get_largest_mask(masks):
    """Return the largest mask from a list of masks"""
    if not masks:
        return None

    # Find mask with largest area using max() for efficiency
    largest_mask = max(masks, key=lambda x: x['area'])
    return largest_mask['segmentation']

@torch.no_grad()
def segment_image(image, mask_generator):
    """Segment the image using SAM2 and return the largest object mask"""
    # Ensure image is in correct format for SAM (RGB uint8)
    if len(image.shape) == 2:
        image = np.stack((image,) * 3, axis=-1)

    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    # Generate masks
    with st.spinner("Generating object mask..."):
        masks = mask_generator.generate(image)
    
    # st.write(f"Found {len(masks)} potential objects")

    # Get the largest mask
    mask = get_largest_mask(masks)

    if mask is None:
        return None, None

    # Create masked image more efficiently using numpy operations
    masked_img = np.where(mask[:, :, np.newaxis], image, 0)

    return mask, masked_img

# Vision Transformer based model for dimension prediction
class DimensionPredictor(nn.Module):
    def __init__(self, pretrained=False):
        super(DimensionPredictor, self).__init__()

        # Use a smaller, more efficient model - MobileNetV3 instead of ViT
        import timm
        self.backbone = timm.create_model('mobilenetv3_large_100', pretrained=pretrained, features_only=False)

        # Get the dimension of the backbone output
        backbone_output_dim = self.backbone.classifier.in_features

        # Remove the classification head
        self.backbone.classifier = nn.Identity()

        # Use a more efficient regression head
        self.regression_head = nn.Sequential(
            nn.Linear(backbone_output_dim * 2, 256),  # *2 because we concatenate features
            nn.BatchNorm1d(256),  # Add BatchNorm for faster convergence
            nn.ReLU(inplace=True),  # Use inplace operations
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3)  # 3 outputs: height, width, length
        )

    def forward(self, full_img, masked_obj):
        # Extract features from both images
        full_img_features = self.backbone(full_img)
        masked_obj_features = self.backbone(masked_obj)

        # Concatenate features
        combined_features = torch.cat((full_img_features, masked_obj_features), dim=1)

        # Pass through regression head
        dimensions = self.regression_head(combined_features)

        return dimensions

@torch.no_grad()
def predict_dimensions(image, model, mask_generator, transform):
    """
    Optimized prediction function for inference
    Args:
        image: PIL Image or numpy array
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
        
    # Make sure it's RGB
    if image_np.shape[2] == 4:  # RGBA
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

    # Segment image
    mask, masked_img = segment_image(image_np, mask_generator)

    if mask is None:
        st.error("No object detected in the image. Please try another image.")
        return None

    # Prepare inputs for the model using full precision
    full_img_tensor = transform(Image.fromarray(image_np)).unsqueeze(0).to(device)
    masked_obj_tensor = transform(Image.fromarray(masked_img)).unsqueeze(0).to(device)

    # Make prediction
    model.eval()
    output = model(full_img_tensor, masked_obj_tensor)

    # Convert outputs to dimensions
    height, width, length = output[0].cpu().numpy()

    return {
        "height": float(height*1.5), 
        "width": float(width*1.5), 
        "length": float(length*1.5)
    }, mask, masked_img

@st.cache_resource
def load_model():
    """Load the trained dimension prediction model"""
    # Import timm here to make sure it's installed
    import timm
    
    # Initialize model
    model = DimensionPredictor(pretrained=False)
    
    # Load model from the local file path
    model_path = "dimension_prediction_model.pth"
    
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
                
            model = model.to(device)
            model.eval()
            st.success("Model loaded successfully!")
            return model
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None
    else:
        st.error(f"Model file not found at {model_path}. Please make sure the model file is in the same directory as this app.")
        return None

def main():
    # Configure session state to ensure consistent session ID
    if 'session_initialized' not in st.session_state:
        st.session_state.session_initialized = True
    
    st.set_page_config(
        page_title="Object Dimension Predictor",
        page_icon="📏",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    st.title("Object Dimension Predictor")
    st.subheader("Predicts real-world dimensions (height, width, length) of objects in images")
    
    # Install dependencies if not already done
    with st.spinner("Setting up environment..."):
        install_dependencies()
    
    # Initialize the model
    with st.spinner("Loading model..."):
        model = load_model()
    
    if model is None:
        st.stop()
    
    # Initialize SAM model
    with st.spinner("Initializing segmentation model..."):
        mask_generator = initialize_sam()
    
    # Define transform for inference
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Image upload
    st.subheader("Upload an image containing an object")
    
    # Configure file uploader with correct parameters for cloud deployment
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        key="file_uploader"
    )
    
    use_example = st.checkbox("Or use an example image")
    
    image = None
    col1, col2 = st.columns(2)
    
    if uploaded_file is not None:
        try:
            # Convert the file to an image with proper error handling
            image_bytes = uploaded_file.read()
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
            col1.image(image, caption="Uploaded Image", use_column_width=True)
        except Exception as e:
            st.error(f"Error processing uploaded image: {str(e)}")
    elif use_example:
        try:
            # Use example image
            example_url = "https://m.media-amazon.com/images/I/71ClGjocCKL._SX679_.jpg"
            response = requests.get(example_url)
            image = Image.open(BytesIO(response.content)).convert('RGB')
            col1.image(image, caption="Example Image", use_column_width=True)
        except Exception as e:
            st.error(f"Error loading example image: {str(e)}")
    
    if image is not None:
        # Process button
        if st.button("Predict Dimensions"):
            with st.spinner("Processing..."):
                try:
                    # Predict dimensions
                    result, mask, masked_img = predict_dimensions(image, model, mask_generator, transform)
                    
                    if result:
                        # Display results
                        st.success("Prediction completed!")
                        
                        # Display segmentation mask
                        col2.image(masked_img, caption="Segmentation Mask", use_column_width=True)
                        
                        # Display predictions in a nice format
                        st.subheader("Predicted Dimensions")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        col1.metric("Height", f"{abs(result['height']):.2f} m")
                        col2.metric("Width", f"{abs(result['width']):.2f} m")
                        col3.metric("Length", f"{abs(result['length']):.2f} m")
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    st.error("Please try another image or check if the model files are correctly loaded.")

if __name__ == "__main__":
    main()