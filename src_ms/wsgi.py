from flask import Flask
from model.model_loader import load_model
from model.early_exit import sdn_test_early_exits

app = Flask(__name__)

# Load the model when the application starts
model = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    # Logic to handle incoming requests and call the model for predictions
    data = request.get_json()
    predictions = sdn_test_early_exits(model, data)
    return jsonify(predictions)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)