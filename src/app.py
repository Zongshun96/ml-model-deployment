from flask import Flask, request, jsonify
from model.model_loader import load_model
from model.early_exit import sdn_test_early_exits
import torch
import io
import base64
import lz4.frame

app = Flask(__name__)

# Load the model when the application starts
model = load_model()

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'input' not in data:
        return jsonify({'error': 'Invalid input'}), 400

    # Expect the input data to be a base64 encoded, lz4-compressed blob
    encoded_input = data['input']['data']
    CONFIDENCE_THRESHOLD = data['input']['confidence_threshold']
    
    try:
        # Decode from base64
        compressed_input = base64.b64decode(encoded_input)
        # Decompress using lz4.frame.decompress
        decompressed_input = lz4.frame.decompress(compressed_input)
        # Load the tensor (or input data) from the decompressed bytes
        buffer = io.BytesIO(decompressed_input)
        loaded_input = torch.load(buffer)
    except Exception as e:
        return jsonify({'error': f'Failed to decompress and deserialize input: {str(e)}'}), 400

    # If the loaded input is not already a tensor, convert it
    input_tensor = loaded_input if isinstance(loaded_input, torch.Tensor) else torch.tensor(loaded_input)

    # Call the early exit function to get predictions
    high_conf_mask_l, output_l, output_id_l, processed_output_id_l, is_early_l, IaaS_metrics_d_l, FaaS_metrics_d_l = sdn_test_early_exits(model, input_tensor, CONFIDENCE_THRESHOLD)
    
    # Package predictions in a dictionary
    predictions = {
        'high_conf_mask': high_conf_mask_l,
        'output': output_l,
        'output_id': output_id_l,
        'processed_output_id': processed_output_id_l,
        'is_early': is_early_l,
        'IaaS_metrics': IaaS_metrics_d_l,
        'FaaS_metrics': FaaS_metrics_d_l
    }
    
    # Serialize the predictions using torch.save into a BytesIO buffer
    buffer = io.BytesIO()
    torch.save(predictions, buffer)
    buffer.seek(0)
    serialized_data = buffer.read()
    
    # Compress the serialized data using lz4.frame.compress
    compressed_output = lz4.frame.compress(serialized_data)
    # Encode to base64 for safe JSON transmission
    encoded_output = base64.b64encode(compressed_output).decode('utf-8')
    
    response_payload = {'data': encoded_output}
    return jsonify(response_payload)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
