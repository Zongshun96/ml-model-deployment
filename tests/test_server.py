import requests
import numpy as np
import json
import time
import torch
import torchvision

def test_model_endpoint():
    # Server URL
    url = "http://localhost:5000/predict"
    
    # Create dummy input data (adjust shape/values based on your model's requirements)
    import torchvision.transforms as transforms
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load CIFAR10 dataset and get first 4 images
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transform)
    images, _ = next(iter(torch.utils.data.DataLoader(testset, batch_size=4,
                                                     shuffle=False)))
    
    # Prepare input format
    dummy_input = {
        "input": {
            "data": images.numpy().tolist(),
            "batch_size": 4
        }
    }

    try:
        # Send POST request to the server
        start_time = time.time()
        response = requests.post(url, json=dummy_input)
        end_time = time.time()
        
        # Check if request was successful
        if response.status_code == 200:
            result = response.json()
            print("Request successful!")
            print(f"Response time: {end_time - start_time:.3f} seconds")
            print("\nPrediction details:")
            print(f"High confidence mask: {result['high_conf_mask']}")
            print(f"Output: {result['output']}")
            print(f"Output ID: {result['output_id']}")
            print(f"Processed output ID: {result['processed_output_id']}")
            print(f"Early exit: {result['is_early']}")
            print(f"IaaS metrics: {result['IaaS_metrics']}")
            print(f"FaaS metrics: {result['FaaS_metrics']}")
        else:
            print(f"Request failed with status code: {response.status_code}")
            print(f"Error message: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to server. Make sure the Flask server is running.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    test_model_endpoint()