import os
import pickle
from flask import Flask, render_template, request, jsonify

# Import custom text preprocessing module
from preprocess import preprocess_text

app = Flask(__name__)

# Global variables for model and vectorizer
model = None
vectorizer = None
model_load_error = None

def load_ml_assets():
    """
    Loads the trained model and TF-IDF vectorizer pickles using absolute paths.
    """
    global model, vectorizer, model_load_error
    
    # Resolve paths relative to the directory containing this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model.pkl")
    vectorizer_path = os.path.join(base_dir, "vectorizer.pkl")
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(vectorizer_path, 'rb') as f:
                vectorizer = pickle.load(f)
            print("Successfully loaded model.pkl and vectorizer.pkl")
            model_load_error = None
            return True
        except Exception as e:
            model_load_error = str(e)
            print(f"Error loading pickle files: {e}")
            print("Troubleshooting steps:")
            print("1. Ensure 'train_model.py' ran successfully using the same Python environment.")
            print("2. If you are running 'app.py' globally, check if scikit-learn is installed.")
            print("   We recommend running app.py using '.venv\\Scripts\\python app.py'")
            return False
    else:
        model_load_error = "Pickle files not found on disk."
        print(f"Pickle files not found at:\n - {model_path}\n - {vectorizer_path}\nMake sure to run 'train_model.py' first.")
        return False

# Load assets when the server starts
assets_loaded = load_ml_assets()

@app.route('/')
def home():
    """
    Renders the homepage.
    """
    # Check if assets are loaded. Pass a status variable to the frontend.
    # Note: re-check dynamically in case training completed after startup
    current_assets_loaded = assets_loaded or (model is not None and vectorizer is not None)
    if not current_assets_loaded:
        current_assets_loaded = load_ml_assets()
        
    return render_template('index.html', assets_loaded=current_assets_loaded)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives email text from POST request, runs the preprocessor,
    transforms it using TF-IDF, runs prediction using the saved model,
    and returns prediction label and confidence percentage in JSON format.
    """
    global model, vectorizer
    
    # Reload models if they weren't loaded initially (just in case training completed in parallel)
    if model is None or vectorizer is None:
        if not load_ml_assets():
            error_message = "Trained model and vectorizer are not available."
            if model_load_error:
                error_message += f" ({model_load_error})"
            return jsonify({
                "success": False,
                "error": error_message
            }), 500

    try:
        # Get data from request
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({
                "success": False,
                "error": "No email text provided in the request."
            }), 400

        raw_message = data['message']
        
        # Check for empty string input
        if not raw_message.strip():
            return jsonify({
                "success": False,
                "error": "The input text cannot be empty."
            }), 400

        # 1. Preprocess the raw email text
        processed_message = preprocess_text(raw_message)
        
        # 2. Transform the preprocessed text using TF-IDF vectorizer
        # Note: Vectorizer expects a list of documents, so wrap it in a list
        features = vectorizer.transform([processed_message])
        
        # 3. Predict the category
        prediction = model.predict(features)[0] # Returns 0 (Ham) or 1 (Spam)
        
        # 4. Get the prediction probability (confidence)
        probabilities = model.predict_proba(features)[0]
        # confidence is the probability score of the predicted class
        confidence = float(probabilities[prediction]) * 100
        
        # Convert numeric prediction back to user-friendly text
        prediction_label = "Spam" if prediction == 1 else "Ham"
        
        return jsonify({
            "success": True,
            "prediction": prediction_label,
            "confidence": round(confidence, 2),
            "processed_text": processed_message
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"An error occurred during prediction: {str(e)}"
        }), 500

@app.route('/status', methods=['GET'])
def get_status():
    """
    Endpoint to check training and model availability status.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    is_trained = os.path.exists(os.path.join(base_dir, "model.pkl")) and os.path.exists(os.path.join(base_dir, "vectorizer.pkl"))
    return jsonify({
        "model_trained": is_trained,
        "model_type": str(type(model).__name__) if model is not None else "None"
    })

if __name__ == '__main__':
    # Start the Flask app locally on port 5000 in debug mode
    app.run(host='127.0.0.1', port=5000, debug=True)
