import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from google.colab import drive
from tqdm import tqdm
import timm
import gc

# Check if GPU is available and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Mount Google Drive for saving/loading models
drive.mount('/content/drive')

# Install required packages efficiently with minimal output
# !pip install -q timm PyYAML
# !pip install -q git+https://github.com/ChaoningZhang/MobileSAM.git

# Set global variables for faster access
IMG_SIZE = 224  # Image size for ViT
BATCH_SIZE = 16  # Increased batch size for better GPU utilization

# 1. Optimized SAM2 segmentation functions
def initialize_sam():
    """Initialize SAM model with optimized settings"""
    from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

    # Download the model checkpoint if not already present
    if not os.path.exists("mobile_sam.pt"):
        print("Uncomment when run on google colab or Run the following command: wget -q https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt")
        # !wget -q https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt 

    sam_checkpoint = "mobile_sam.pt"
    model_type = "vit_t"

    # Load model with half precision for faster inference
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    # Optimize SAM for inference
    if device.type == 'cuda':
        sam.eval()
        sam = sam.half()  # Use half precision

    # Configure mask generator with optimized parameters
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=16,  # Reduced from 32 for speed
        pred_iou_thresh=0.8,
        stability_score_thresh=0.9,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100
    )

    return mask_generator

@torch.no_grad()  # Ensure no gradients are calculated during segmentation
def get_largest_mask(masks):
    """Return the largest mask from a list of masks"""
    if not masks:
        return None

    # Find mask with largest area using max() for efficiency
    largest_mask = max(masks, key=lambda x: x['area'])
    return largest_mask['segmentation']

@torch.no_grad()  # Decorator to prevent gradient calculation
def segment_image(image, mask_generator):
    """Segment the image using SAM2 and return the largest object mask"""
    # Ensure image is in correct format for SAM (RGB uint8)
    if len(image.shape) == 2:
        image = np.stack((image,) * 3, axis=-1)

    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    # Generate masks
    masks = mask_generator.generate(image)

    # Get the largest mask
    mask = get_largest_mask(masks)

    if mask is None:
        return None, None

    # Create masked image more efficiently using numpy operations
    masked_img = np.where(mask[:, :, np.newaxis], image, 0)

    return mask, masked_img

# 2. Optimized Vision Transformer based model for dimension prediction
class DimensionPredictor(nn.Module):
    def __init__(self, pretrained=True):
        super(DimensionPredictor, self).__init__()

        # Use a smaller, more efficient model - MobileNetV3 instead of ViT
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

# 3. Memory-efficient dataset class with caching
class ObjectDimensionDataset(Dataset):
    def __init__(self, image_dir, dimension_file, transform=None, cache_size=100):
        """
        Memory-efficient dataset with LRU caching
        """
        self.image_dir = image_dir
        self.transform = transform
        self.dimensions = {}
        self.cache = {}  # Image cache
        self.cache_size = cache_size

        # Load dimensions from CSV file
        with open(dimension_file, 'r') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split(',')
                img_name = parts[0]
                dims = [float(parts[1]), float(parts[2]), float(parts[3])]
                self.dimensions[img_name] = dims

        self.image_names = list(self.dimensions.keys())

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Check if image is in cache
        if img_name in self.cache:
            image = self.cache[img_name]
        else:
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Failed to load image at path: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Add to cache
            if len(self.cache) >= self.cache_size:
                # Remove a random item if cache is full
                self.cache.pop(list(self.cache.keys())[0])
            self.cache[img_name] = image

        # Get dimensions
        dimensions = torch.tensor(self.dimensions[img_name], dtype=torch.float32)

        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image)

        # Apply transformations
        if self.transform:
            image = self.transform(pil_image)
        else:
            image = transforms.ToTensor()(pil_image)

        return image, dimensions, img_name

# 4. Optimized batch processing functions with prefetching
class PrefetchLoader:
    def __init__(self, loader):
        self.loader = loader
        self.stream = torch.cuda.Stream()

    def __iter__(self):
        loader_iter = iter(self.loader)
        self.preload(loader_iter)
        while self.batch is not None:
            yield self.batch
            self.preload(loader_iter)

    def preload(self, loader_iter):
        try:
            self.batch = next(loader_iter)
        except StopIteration:
            self.batch = None
            return

        with torch.cuda.stream(self.stream):
            for i, item in enumerate(self.batch):
                if isinstance(item, torch.Tensor):
                    self.batch[i] = item.to(device, non_blocking=True)

@torch.no_grad()  # No gradients needed for processing
def process_batch_with_sam(images, mask_generator, transform):
    """Process a batch of images with SAM - optimized version"""
    full_images = []
    masked_objects = []

    for img_tensor in images:
        # Convert tensor to numpy for SAM
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)

        # Generate mask
        mask, masked_img = segment_image(img_np, mask_generator)

        if mask is None:
            # If segmentation fails, use the whole image
            masked_img = img_np

        # Apply transformations - directly convert to tensor for efficiency
        full_img_tensor = transform(Image.fromarray(img_np))
        masked_obj_tensor = transform(Image.fromarray(masked_img))

        full_images.append(full_img_tensor)
        masked_objects.append(masked_obj_tensor)

    # Stack tensors
    full_images = torch.stack(full_images)
    masked_objects = torch.stack(masked_objects)

    return full_images, masked_objects

# 5. Training and validation functions with mixed precision
def train_one_epoch(model, mask_generator, train_loader, criterion, optimizer, device, transform, scaler):
    model.train()
    running_loss = 0.0

    # Get precomputed segmentation if available or compute on the fly
    for images, target_dims, _ in tqdm(train_loader):
        # Process images with SAM
        with torch.cuda.amp.autocast():  # Use mixed precision
            full_images, masked_objects = process_batch_with_sam(images, mask_generator, transform)

            # Move data to device
            full_images = full_images.to(device)
            masked_objects = masked_objects.to(device)
            target_dims = target_dims.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

            # Forward pass
            outputs = model(full_images, masked_objects)
            loss = criterion(outputs, target_dims)

        # Backward pass with gradient scaling for mixed precision
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

    # Clear cache to prevent memory leaks
    torch.cuda.empty_cache()
    gc.collect()

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

@torch.no_grad()  # No gradients needed for validation
def validate(model, mask_generator, val_loader, criterion, device, transform):
    model.eval()
    running_loss = 0.0

    for images, target_dims, _ in tqdm(val_loader):
        # Process images with SAM using mixed precision
        with torch.cuda.amp.autocast():
            full_images, masked_objects = process_batch_with_sam(images, mask_generator, transform)

            # Move data to device
            full_images = full_images.to(device)
            masked_objects = masked_objects.to(device)
            target_dims = target_dims.to(device)

            # Forward pass
            outputs = model(full_images, masked_objects)
            loss = criterion(outputs, target_dims)

        running_loss += loss.item() * images.size(0)

    # Clear cache
    torch.cuda.empty_cache()
    gc.collect()

    val_loss = running_loss / len(val_loader.dataset)
    return val_loss

# 6. Main training pipeline with optimizations
def main():
    # Set seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Define efficient data transformations
    data_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Initialize SAM model
    print("Initializing SAM model...")
    mask_generator = initialize_sam()

    # Create datasets and data loaders with optimized parameters
    train_dataset = ObjectDimensionDataset(
        image_dir='/content/drive/MyDrive/dimension_dataset/train/images',
        dimension_file='/content/drive/MyDrive/dimension_dataset/train/dimensions.csv',
        transform=data_transforms,
        cache_size=50  # Cache last 50 images
    )

    val_dataset = ObjectDimensionDataset(
        image_dir='/content/drive/MyDrive/dimension_dataset/val/images',
        dimension_file='/content/drive/MyDrive/dimension_dataset/val/dimensions.csv',
        transform=data_transforms,
        cache_size=50
    )

    # Use optimized DataLoader settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,  # Speed up data transfer to GPU
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=2  # Prefetch batches
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    # Initialize model
    model = DimensionPredictor(pretrained=True)
    model = model.to(device)

    # Initialize mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Use OneCycleLR scheduler for faster convergence
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        steps_per_epoch=len(train_loader),
        epochs=10,
        pct_start=0.3
    )

    # Training loop
    num_epochs = 10
    best_val_loss = float('inf')

    # Early stopping
    patience = 3
    no_improve_epochs = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Train and validate
        train_loss = train_one_epoch(model, mask_generator, train_loader, criterion, optimizer, device, data_transforms, scaler)
        val_loss = validate(model, mask_generator, val_loader, criterion, device, data_transforms)

        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, '/content/drive/MyDrive/dimension_prediction_model.pth')
            print("✅ Saved best model!")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        # Early stopping
        if no_improve_epochs >= patience:
            print(f"No improvement for {patience} epochs. Stopping training.")
            break

        # Clear memory between epochs
        torch.cuda.empty_cache()
        gc.collect()

    print("Training complete!")

def initialize_sam_for_inference():
    """Initialize SAM model specifically for inference with proper precision"""
    from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

    # Download the model checkpoint if not already present
    if not os.path.exists("mobile_sam.pt"):
        print("Uncomment when run on google colab or Run the following command: wget -q https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt")
        # !wget -q https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt

    sam_checkpoint = "mobile_sam.pt"
    model_type = "vit_t"

    # Load model in full precision for inference
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    # Do NOT convert to half precision for inference
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

@torch.no_grad()  # No gradients needed for inference
def predict_dimensions(image_path, model, mask_generator, transform):
    """Optimized prediction function for inference"""
    # Load image with error handling
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image at {image_path}. Please check if the path is correct and the image file is valid.")
        return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Segment image
    mask, masked_img = segment_image(image, mask_generator)

    if mask is None:
        print("No object detected in the image.")
        return None

    # Visualize segmentation
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Segmentation Mask")

    plt.subplot(1, 3, 3)
    plt.imshow(masked_img)
    plt.title("Segmented Object")
    plt.show()

    # Prepare inputs for the model using full precision
    full_img_tensor = transform(Image.fromarray(image)).unsqueeze(0).to(device)
    masked_obj_tensor = transform(Image.fromarray(masked_img)).unsqueeze(0).to(device)

    # Make prediction
    model.eval()
    output = model(full_img_tensor, masked_obj_tensor)

    # Convert outputs to dimensions
    height, width, length = output[0].cpu().numpy()

    return {"height": height, "width": width, "length": length}

def test_on_image(test_image_path=None):
    """
    Function to test the model on a new image
    Args:
        test_image_path: Path to test image. If None, will prompt for a path.
    """
    if test_image_path is None:
        test_image_path = input("Enter the full path to your test image: ")

    # Check if file exists
    if not os.path.isfile(test_image_path):
        print(f"Error: File not found at path: {test_image_path}")
        print("Please check the path and try again.")
        return

    try:
        # Load trained model
        checkpoint_path = '/content/drive/MyDrive/dimension_prediction_model.pth'
        if not os.path.isfile(checkpoint_path):
            print(f"Error: Model checkpoint not found at {checkpoint_path}")
            return

        checkpoint = torch.load(checkpoint_path)
        model = DimensionPredictor(pretrained=False)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(device)
        model.eval()

        # Initialize SAM specifically for inference
        mask_generator = initialize_sam_for_inference()

        # Define transform
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Predict dimensions
        dimensions = predict_dimensions(test_image_path, model, mask_generator, transform)

        if dimensions:
            print(f"Predicted dimensions: Height = {dimensions['height']}m, Width = {dimensions['width']}m, Length = {dimensions['length']}m")

    except Exception as e:
        print(f"An error occurred during testing: {str(e)}")
        import traceback
        traceback.print_exc()


# 8. Example usage
if __name__ == "__main__":
    # main()  # Train the model

    # After training, you can test on new images:
    # Uncomment the line below ONLY when you have a test image available
    test_on_image("/content/71ClGjocCKL._SX679_.jpg")