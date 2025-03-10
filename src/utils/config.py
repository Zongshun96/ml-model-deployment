# File: /ml-model-deployment/ml-model-deployment/src/utils/config.py

# Configuration settings for the application

import os

# Define the base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model configuration
MODEL_PATH = os.path.join(BASE_DIR, 'model/networks/{}'.format(8))  # Set the path to the model file
CONFIDENCE_THRESHOLD = 0.85  # Set the confidence threshold for predictions
SDN_NAME = "cifar10_vgg16bn_sdn_sdn_training"
CUT_OUTPUT_IDX = 15  # Set the index to cut the model output
DEVICE = 'cpu'  # Set the device to

# Logging configuration
LOGGING_LEVEL = 'INFO'  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

# Other constants
BATCH_SIZE = 32  # Set the batch size for inference
NUM_CLASSES = 10  # Update with the number of classes in your model

# Environment variables
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')  # Default to development if not set

# Add any other configuration settings as needed