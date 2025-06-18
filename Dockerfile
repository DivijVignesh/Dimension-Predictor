# Use Python 3.10 as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and PyTorch
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    git \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create a directory for temp files with appropriate permissions
RUN mkdir -p /tmp/streamlit_uploads && chmod 777 /tmp/streamlit_uploads

# Environment variables for Streamlit
ENV PORT=8081
ENV STREAMLIT_SERVER_PORT=8081
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=true
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV TEMP=/tmp/streamlit_uploads
ENV TMPDIR=/tmp/streamlit_uploads
ENV STREAMLIT_CLIENT_TOOLING=false
ENV STREAMLIT_THEME_BASE="light"
ENV STREAMLIT_SERVER_COOKIE_SECRET="9f57cc5bab5c4d33837b694eba384deb"

# Expose port 8081 (matching what we configured)
EXPOSE 8081

# Command to run the application with increased upload limit and proper CORS settings
CMD ["streamlit", "run", "main.py", "--server.port=8081", "--server.enableCORS=true", "--server.enableXsrfProtection=false", "--server.maxUploadSize=10", "--server.address=0.0.0.0", "--server.fileWatcherType=none"]