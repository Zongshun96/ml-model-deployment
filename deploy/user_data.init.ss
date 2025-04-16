#!/bin/bash

# Enable error handling
set -e

# Set up logging to home directory
exec 1> >(tee -a ~/user_data.log) 2>&1

# Update and install packages for Ubuntu
sudo apt-get update -y
sudo apt-get install -y python3-pip git zip unzip awscli

# Clone the repository
REPO_URL="https://github.com/Zongshun96/ml-model-deployment.git"
REPO_PATH="/home/ubuntu/ml-model-deployment"

echo "Cloning repository..."
git clone "$REPO_URL" "$REPO_PATH"

# Download the ZIP files from S3
echo "Downloading models from S3..."
# aws s3 cp s3://your-bucket-name/path/to/True25_1000submodel_verpak.zip "$REPO_PATH/src_ss/models/True25_1000submodel_verpak.zip"
# aws s3 cp s3://your-bucket-name/path/to/True25_1submodel_verpak.zip "$REPO_PATH/src_ss/models/True25_1submodel_verpak.zip"
aws s3 cp s3://praxi-model-xgb-02/True25_1000submodel_verpak.zip "$REPO_PATH/src_ss/models/True25_1000submodel_verpak.zip"
aws s3 cp s3://praxi-model-xgb-02/True25_1submodel_verpak.zip "$REPO_PATH/src_ss/models/True25_1submodel_verpak.zip"

# Unzip the downloaded files
echo "Unzipping model files..."
unzip -o "$REPO_PATH/src_ss/models/True25_1000submodel_verpak.zip" -d "$REPO_PATH/src_ss/models/"
unzip -o "$REPO_PATH/src_ss/models/True25_1submodel_verpak.zip" -d "$REPO_PATH/src_ss/models/"

# Install Python packages
echo "Installing Python requirements..."
cd "$REPO_PATH"
sudo pip3 install -r requirements.ss

# Pull the latest changes
echo "Pulling the changes..."
git pull origin main

# Run the application
echo "Starting the application..."
python3 "src_ss/app.py" &
