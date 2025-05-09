# This script fixes potential issues with MobileSAM integration in Streamlit
# Save as mobile_sam_fix.py and run as a separate step before starting your app

import os
import sys
import subprocess
import importlib.util

def is_package_installed(package_name):
    """Check if a package is installed"""
    return importlib.util.find_spec(package_name) is not None

def install_package(package):
    """Install a package using pip"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    print("Setting up environment for MobileSAM...")
    
    # Install base dependencies
    dependencies = [
        "torch==2.0.1",
        "torchvision==0.15.2", 
        "timm==0.9.2",
        "PyYAML==6.0",
        "opencv-python-headless==4.8.0.74",
        "protobuf==3.20.3"
    ]
    
    for dep in dependencies:
        print(f"Installing {dep}...")
        try:
            install_package(dep)
        except Exception as e:
            print(f"Warning: Could not install {dep}: {e}")
    
    # Ensure MobileSAM is installed correctly
    if not is_package_installed("mobile_sam"):
        print("Installing MobileSAM...")
        try:
            install_package("git+https://github.com/ChaoningZhang/MobileSAM.git")
        except Exception as e:
            print(f"Warning: Could not install MobileSAM: {e}")
    
    # Download model checkpoint if needed
    if not os.path.exists("mobile_sam.pt"):
        print("Downloading model checkpoint...")
        try:
            import requests
            url = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
            response = requests.get(url)
            with open("mobile_sam.pt", 'wb') as f:
                f.write(response.content)
            print("Model checkpoint downloaded successfully.")
        except Exception as e:
            print(f"Warning: Could not download model checkpoint: {e}")
    
    print("Setup complete! You can now run your Streamlit app.")
    print("Use: streamlit run app.py")

if __name__ == "__main__":
    main()