from flask import Flask, request, jsonify
from model.model_loader import load_model
from model.early_exit import sdn_test_early_exits
import torch

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

    input_data = data['input']['data']
    input_batchsize = data['input']['batch_size']
    # Call the early exit function to get predictions
    # Convert input data to tensor if needed
    input_tensor = torch.tensor(input_data) if not isinstance(input_data, torch.Tensor) else input_data
    high_conf_mask_l, output_l, output_id_l, processed_output_id_l, is_early_l, IaaS_metrics_d_l, FaaS_metrics_d_l = sdn_test_early_exits(model, input_tensor)
    

    predictions = {
        'high_conf_mask': tensor_to_list(high_conf_mask_l),
        'output': tensor_to_list(output_l),
        'output_id': tensor_to_list(output_id_l),
        'processed_output_id': tensor_to_list(processed_output_id_l),
        'is_early': tensor_to_list(is_early_l),
        'IaaS_metrics': IaaS_metrics_d_l,
        'FaaS_metrics': FaaS_metrics_d_l
    }
    return jsonify(predictions)

def tensor_to_list(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, list):
        return [tensor_to_list(item) for item in obj]
    return obj

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)