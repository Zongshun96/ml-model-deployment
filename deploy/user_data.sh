#!/bin/bash

# Enable error handling
set -e

touch ~/user_data_script_executed

# Set up logging to home directory
exec 1> >(tee -a ~/user_data.log) 2>&1

# Update and install packages for Ubuntu
sudo apt-get update -y
sudo apt-get install -y python3-pip git

# Clone the repository
REPO_URL="https://github.com/Zongshun96/ml-model-deployment.git"
REPO_PATH="/home/ubuntu/ml-model-deployment"

echo "Cloning repository..."
git clone "$REPO_URL" "$REPO_PATH"

# Install Python packages
echo "Installing Python requirements..."
cd "$REPO_PATH"
sudo pip3 install -r requirements.txt

# Run the application
echo "Starting the application..."
python3 "src/app.py" &
