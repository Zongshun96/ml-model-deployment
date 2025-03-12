#!/usr/bin/env python3
import requests
import time
import base64
import numpy as np
from PIL import Image
import io

# Constant: Replace with your actual ALB endpoint (e.g. "http://your-alb-dns")
ALB_URL = "http://your-alb-dns"

def create_dummy_imagenet_image():
    """
    Create a dummy image with ImageNet-like dimensions (224x224 RGB)
    and encode it as a base64 string.
    """
    # Create a random image (values between 0 and 255)
    dummy_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    dummy_image = Image.fromarray(dummy_array)
    
    # Save the image to a bytes buffer in JPEG format
    buffer = io.BytesIO()
    dummy_image.save(buffer, format="JPEG")
    byte_data = buffer.getvalue()
    
    # Encode the byte data to a base64 string
    base64_str = base64.b64encode(byte_data).decode('utf-8')
    return base64_str

def send_classification_request(url, base64_image):
    """
    Sends an HTTP POST request to the specified URL with a JSON payload.
    The payload contains the base64 encoded image under the 'image' key.
    """
    payload = {"image": base64_image}
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, headers=headers, json=payload)
    return response

def main():
    print("Creating dummy ImageNet-style image...")
    base64_image = create_dummy_imagenet_image()

    print("Sending dummy image to ALB...")
    try:
        start = time.time()
        response = send_classification_request(ALB_URL, base64_image)
        elapsed = time.time() - start
        if response.status_code == 200:
            print(f"Response received in {elapsed:.2f} seconds:")
            try:
                print(response.json())
            except Exception:
                print("Response is not in JSON format:", response.text)
        else:
            print(f"Request failed with status code {response.status_code}:")
            print(response.text)
    except Exception as e:
        print("Error sending request:", e)

if __name__ == '__main__':
    main()
