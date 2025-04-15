from .network_architectures import load_model as arc_load_model
from utils.config import MODEL_PATH, CONFIDENCE_THRESHOLD, SDN_NAME, CUT_OUTPUT_IDX, DEVICE

def load_model():
    sdn_model, sdn_params = arc_load_model(MODEL_PATH, SDN_NAME, epoch=-1)
    sdn_model.to(DEVICE)
    sdn_model.forward = sdn_model.early_exit
    sdn_model.cut_output_idx = CUT_OUTPUT_IDX
    sdn_model.confidence_threshold = CONFIDENCE_THRESHOLD
    sdn_model.eval()
    return sdn_model

def prepare_input(input_data):
    import numpy as np

    # Preprocess the input data as required by the model
    processed_data = np.array(input_data)  # Example preprocessing
    return processed_data

def predict(model, input_data):
    import torch

    # Prepare the input data
    processed_data = prepare_input(input_data)

    # Convert to tensor
    input_tensor = torch.tensor(processed_data, dtype=torch.float32)

    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)

    return output.numpy()  # Return the output as a numpy array