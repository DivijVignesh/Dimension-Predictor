# This script helps prepare and organize datasets for training the object dimension prediction model

import os
import shutil
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from google.colab import drive
import trimesh

# Mount Google Drive
drive.mount('/content/drive')

# Create necessary directories
def create_directory_structure():
    """Create the directory structure for the dataset"""
    base_dir = '/content/drive/MyDrive/dimension_dataset'

    # Create main directories
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'train/images'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'val/images'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'test/images'), exist_ok=True)

    return base_dir

# Sample dataset preparation function for Pix3D dataset
def prepare_pix3d_dataset():
    """
    Prepare the Pix3D dataset for training
    Download from: http://pix3d.csail.mit.edu/
    """
    # Install the dataset if not already downloaded
    if not os.path.exists('/content/pix3d.zip'):
        print("uncomment when run on google colab or run these commands: 'wget http://pix3d.csail.mit.edu/data/pix3d.zip' and 'unzip pix3d.zip -d /content/'")
        # !wget http://pix3d.csail.mit.edu/data/pix3d.zip
        # !unzip pix3d.zip -d /content/

    # Load annotations
    annotations_path = '/content/pix3d.json'
    import json
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    # Create directory structure
    base_dir = create_directory_structure()

    # Split data into train/val/test (70/15/15)
    random.seed(42)
    random.shuffle(annotations)
    n_samples = len(annotations)
    print("Total Samples:", n_samples)

    train_split = int(0.7 * n_samples)
    val_split = int(0.85 * n_samples)

    train_data = annotations[:train_split]
    val_data = annotations[train_split:val_split]
    test_data = annotations[val_split:]

    print(train_data)
    # Process and copy the images
    splits = [
        ('train', train_data),
        ('val', val_data),
        ('test', test_data)
    ]

    for split_name, split_data in splits:
        # Create CSV file for dimensions
        dimension_data = []

        for item in tqdm(split_data, desc=f"Processing {split_name} split"):
            # Skip if mask is not available
            if not item.get('mask', ''):
                continue

            # Get image path
            img_path = os.path.join('/content/', item['img'])

            # Skip if image doesn't exist
            if not os.path.exists(img_path):
                continue

            # Get model path for dimensions
            model_path = os.path.join('/content/', item['model'])

            # Skip if model doesn't exist
            if not os.path.exists(model_path):
                continue

            # Get dimensions (typically in the 3D model metadata)
            # For Pix3D, the dimensions are in the "bbox" field in meters
            bbox = item.get('bbox', None)
            if bbox is None:
                continue


            # Step 1: Load the Pix3D 3D model (.obj)
            mesh = trimesh.load(model_path)

            # Step 2: Get the axis-aligned bounding box
            min_bounds, max_bounds = mesh.bounds

            # Step 3: Compute the dimensions
            dimensions = max_bounds - min_bounds
            width, height, length = dimensions  # In meters, assuming standard unit

            # Step 4: Display
            print(f"Width (X-axis):  {width:.3f} meters")
            print(f"Height (Y-axis): {height:.3f} meters")
            print(f"Length (Z-axis): {length:.3f} meters")

            print("bbox exists")
            # Extract dimensions from bbox
            # height, width, length = bbox

            # Copy image to destination
            img_filename = os.path.basename(item['img'])
            dest_path = os.path.join(base_dir, split_name, 'images', img_filename)

            try:
                shutil.copy2(img_path, dest_path)
                print("Image path:",img_path," Destination path:",dest_path)
                # Add dimension data
                dimension_data.append({
                    'image_name': img_filename,
                    'height': height,
                    'width': width,
                    'length': length
                })
            except Exception as e:
                print(f"Error copying {img_path}: {e}")

        # Save dimension data to CSV
        df = pd.DataFrame(dimension_data)
        csv_path = os.path.join(base_dir, split_name, 'dimensions.csv')
        df.to_csv(csv_path, index=False)

        print(f"Processed {len(dimension_data)} samples for {split_name} split")

# Sample dataset preparation function for SUN RGB-D dataset
def prepare_sunrgbd_dataset():
    """
    Prepare the SUN RGB-D dataset for training
    Download from: https://rgbd.cs.princeton.edu/
    """
    # This would typically involve downloading and extracting the dataset
    # For this example, we'll assume it's already downloaded to /content/SUNRGBD

    # Check if dataset exists
    if not os.path.exists('/content/SUNRGBD'):
        print("Please download the SUN RGB-D dataset to /content/SUNRGBD")
        return

    # Create directory structure
    base_dir = create_directory_structure()

    # SUN RGB-D has annotations for furniture with dimensions
    # For this example, we'll create a synthetic version

    # Get all image paths
    image_dir = '/content/SUNRGBD/images'
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]

    # Split data
    random.seed(42)
    random.shuffle(image_paths)
    n_samples = len(image_paths)

    train_split = int(0.7 * n_samples)
    val_split = int(0.85 * n_samples)

    train_paths = image_paths[:train_split]
    val_paths = image_paths[train_split:val_split]
    test_paths = image_paths[val_split:]

    # Process splits
    splits = [
        ('train', train_paths),
        ('val', val_paths),
        ('test', test_paths)
    ]

    for split_name, split_paths in splits:
        # Create CSV file for dimensions
        dimension_data = []

        for img_path in tqdm(split_paths, desc=f"Processing {split_name} split"):
            img_filename = os.path.basename(img_path)
            dest_path = os.path.join(base_dir, split_name, 'images', img_filename)

            try:
                # Copy image
                shutil.copy2(img_path, dest_path)

                # In a real implementation, we would load the actual annotations
                # For this example, we'll create synthetic dimensions
                # In a real scenario, you would parse the annotation files

                # Synthetic dimensions (in meters)
                height = random.uniform(0.5, 2.0)
                width = random.uniform(0.5, 2.0)
                length = random.uniform(0.5, 2.0)

                dimension_data.append({
                    'image_name': img_filename,
                    'height': height,
                    'width': width,
                    'length': length
                })
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        # Save dimension data to CSV
        df = pd.DataFrame(dimension_data)
        csv_path = os.path.join(base_dir, split_name, 'dimensions.csv')
        df.to_csv(csv_path, index=False)

        print(f"Processed {len(dimension_data)} samples for {split_name} split")

# Function to create synthetic data for testing
def create_synthetic_dataset(num_samples=1000):
    """Create a synthetic dataset for testing"""
    base_dir = create_directory_structure()

    # Split data
    n_train = int(0.7 * num_samples)
    n_val = int(0.15 * num_samples)
    n_test = num_samples - n_train - n_val

    # Generate synthetic data
    for split_name, n_samples in [('train', n_train), ('val', n_val), ('test', n_test)]:
        # Create CSV file for dimensions
        dimension_data = []

        for i in tqdm(range(n_samples), desc=f"Creating {split_name} split"):
            # Generate a synthetic image with a shape
            img = np.ones((256, 256, 3), dtype=np.uint8) * 255

            # Choose a random shape
            shape_type = random.choice(['rectangle', 'circle', 'triangle'])

            # Random dimensions
            height = random.uniform(0.5, 2.0)
            width = random.uniform(0.5, 2.0)
            length = random.uniform(0.5, 2.0)

            # Draw shape
            if shape_type == 'rectangle':
                pt1 = (random.randint(50, 100), random.randint(50, 100))
                pt2 = (pt1[0] + random.randint(50, 100), pt1[1] + random.randint(50, 100))
                cv2.rectangle(img, pt1, pt2, (0, 0, 255), -1)
            elif shape_type == 'circle':
                center = (random.randint(100, 150), random.randint(100, 150))
                radius = random.randint(30, 70)
                cv2.circle(img, center, radius, (0, 255, 0), -1)
            else:  # triangle
                pts = np.array([
                    [random.randint(50, 200), random.randint(50, 100)],
                    [random.randint(50, 200), random.randint(150, 200)],
                    [random.randint(50, 200), random.randint(100, 150)]
                ], np.int32)
                cv2.fillPoly(img, [pts], (255, 0, 0))

            # Save the image
            img_filename = f"{shape_type}_{i}.jpg"
            img_path = os.path.join(base_dir, split_name, 'images', img_filename)
            cv2.imwrite(img_path, img)

            # Add dimension data
            dimension_data.append({
                'image_name': img_filename,
                'height': height,
                'width': width,
                'length': length
            })

        # Save dimension data to CSV
        df = pd.DataFrame(dimension_data)
        csv_path = os.path.join(base_dir, split_name, 'dimensions.csv')
        df.to_csv(csv_path, index=False)

        print(f"Created {len(dimension_data)} samples for {split_name} split")

# Function to visualize some examples from the dataset
def visualize_dataset_samples(num_samples=5):
    """Visualize some examples from the dataset"""
    base_dir = '/content/drive/MyDrive/dimension_dataset'

    # Load train data
    train_img_dir = os.path.join(base_dir, 'train/images')
    train_csv = os.path.join(base_dir, 'train/dimensions.csv')

    if not os.path.exists(train_csv):
        print(f"CSV file not found: {train_csv}")
        return

    df = pd.read_csv(train_csv)

    # Randomly select samples
    samples = df.sample(num_samples)

    # Display samples
    plt.figure(figsize=(15, 3*num_samples))

    for i, (_, row) in enumerate(samples.iterrows()):
        img_path = os.path.join(train_img_dir, row['image_name'])

        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.subplot(num_samples, 1, i+1)
        plt.imshow(img)
        plt.title(f"Image: {row['image_name']}\nDimensions: H={row['height']:.2f}m, W={row['width']:.2f}m, L={row['length']:.2f}m")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Function to check if the dataset is correctly set up
def check_dataset():
    """Check if the dataset is correctly set up"""
    base_dir = '/content/drive/MyDrive/dimension_dataset'

    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(base_dir, f'{split}/images')
        csv_path = os.path.join(base_dir, f'{split}/dimensions.csv')

        if not os.path.exists(img_dir):
            print(f"⚠️ {img_dir} directory not found")
        else:
            num_images = len([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
            print(f"✓ {split} images: {num_images}")

        if not os.path.exists(csv_path):
            print(f"⚠️ {csv_path} file not found")
        else:
            df = pd.read_csv(csv_path)
            print(f"✓ {split} annotations: {len(df)} rows")

            # Check if all images have annotations
            img_files = set([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
            ann_files = set(df['image_name'])

            if img_files != ann_files:
                print(f"⚠️ Mismatch between images and annotations")
                print(f"  - Images without annotations: {len(img_files - ann_files)}")
                print(f"  - Annotations without images: {len(ann_files - img_files)}")
            else:
                print(f"✓ All images have annotations")

# Main function to prepare the dataset
def main():
    print("Dataset Preparation Options:")
    print("1. Prepare Pix3D dataset")
    print("2. Prepare SUN RGB-D dataset")
    print("3. Create synthetic dataset")
    print("4. Visualize dataset samples")
    print("5. Check dataset setup")

    choice = input("Enter your choice (1-5): ")

    if choice == '1':
        prepare_pix3d_dataset()
    elif choice == '2':
        prepare_sunrgbd_dataset()
    elif choice == '3':
        num_samples = int(input("Enter number of synthetic samples to generate: "))
        create_synthetic_dataset(num_samples)
    elif choice == '4':
        num_samples = int(input("Enter number of samples to visualize: "))
        visualize_dataset_samples(num_samples)
    elif choice == '5':
        check_dataset()
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()