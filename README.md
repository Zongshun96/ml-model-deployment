# ml-model-deployment/ml-model-deployment/README.md

# ML Model Deployment

This project provides a framework for deploying machine learning models in a cloud environment. It includes scripts for auto-scaling, environment setup, and model inference.

## Project Structure

```
ml-model-deployment
├── deploy
│   ├── user_data.sh        # Script to set up the environment and run the application
│   └── autoscaling.py      # Script to manage auto-scaling of VM instances
├── src
│   ├── app.py              # Main entry point of the application
│   ├── model
│   │   ├── __init__.py     # Marks the model directory as a package
│   │   ├── model_loader.py  # Functions to load the machine learning model
│   │   └── early_exit.py    # Logic for early exits in model inference
│   ├── utils
│   │   └── config.py       # Configuration settings for the application
│   └── wsgi.py             # Entry point for WSGI-compatible web servers
├── requirements.txt         # Python dependencies for the project
└── README.md                # Documentation for the project
```

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ml-model-deployment
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Provision the VM:**
   - Use the `deploy/autoscaling.py` script to create and configure the auto-scaling group and load balancer.

4. **Run the application:**
   - The application will automatically start when the VM is provisioned using the `deploy/user_data.sh` script.

## Usage

- The application handles incoming requests for model inference. You can send requests to the load balancer's DNS name once the setup is complete.

## License

This project is licensed under the MIT License. See the LICENSE file for details.