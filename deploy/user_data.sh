#!/bin/bash

# Enable error handling
set -e

# Log file
exec 1> >(logger -s -t $(basename $0)) 2>&1

# Update and install packages with sudo
sudo apt-get update
sudo apt-get install -y python3-pip git

# Clone the repository (adjust the URL as needed)
REPO_URL="https://github.com/your-username/ml-model-deployment.git"
REPO_PATH="/home/ubuntu/ml-model-deployment"

echo "Cloning repository..."
git clone "$REPO_URL" "$REPO_PATH"

# Install Python packages
echo "Installing Python requirements..."
cd "$REPO_PATH"
sudo pip3 install -r requirements.txt

# Run the application
echo "Starting the application..."
python3 src/app.py