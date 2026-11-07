import os
import pickle
from flask import Flask, render_template, request, jsonify

# Import custom text preprocessing module
from preprocess import preprocess_text

# Resolve absolute base directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Configure Flask with explicit paths for templates and static files
app = Flask(
    __name__,
    template_folder=os.path.join(base_dir, 'templates'),
    static_folder=os.path.join(base_dir, 'static')
)

# Global variables for model and vectorizer
model = None
vectorizer = None
model_load_error = None

def load_ml_assets():
    """
    Loads the trained model and TF-IDF vectorizer pickles using absolute paths.
    Includes fallback search for serverless deployment environments.
    """
    global model, vectorizer, model_load_error
    
    # Primary paths relative to this script
    model_path = os.path.join(base_dir, "model.pkl")
    vectorizer_path = os.path.join(base_dir, "vectorizer.pkl")
    
    # Fallback to parent directory if executing from a nested folder (e.g., api/)
    if not (os.path.exists(model_path) and os.path.exists(vectorizer_path)):
        parent_dir = os.path.dirname(base_dir)
        alt_model_path = os.path.join(parent_dir, "model.pkl")
        alt_vec_path = os.path.join(parent_dir, "vectorizer.pkl")
        if os.path.exists(alt_model_path) and os.path.exists(alt_vec_path):
            model_path = alt_model_path
            vectorizer_path = alt_vec_path

    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(vectorizer_path, 'rb') as f:
                vectorizer = pickle.load(f)
            print(f"Successfully loaded model and vectorizer from: {model_path}")
            model_load_error = None
            return True
        except Exception as e:
            model_load_error = str(e)
            print(f"Error loading pickle files: {e}")
            return False
    else:
        model_load_error = f"Pickle files not found on disk at: {model_path} and {vectorizer_path}"
        print(model_load_error)
        return False

# Attempt initial asset load on startup
assets_loaded = load_ml_assets()

@app.route('/')
def home():
    """
    Renders the homepage dashboard.
    """
    global model, vectorizer
    if model is None or vectorizer is None:
        load_ml_assets()
    current_assets_loaded = (model is not None and vectorizer is not None)
    return render_template('index.html', assets_loaded=current_assets_loaded)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives email text from POST request, runs the preprocessor,
    transforms it using TF-IDF, runs prediction using the saved model,
    and returns prediction label and confidence percentage in JSON format.
    """
    global model, vectorizer
    
    # Lazy-load models if cold-started without initial load
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
        # Get JSON data from request
        data = request.get_json(silent=True)
        if not data or 'message' not in data:
            return jsonify({
                "success": False,
                "error": "No email text provided in the request."
            }), 400

        raw_message = data['message']
        
        # Check for empty or non-string input
        if not isinstance(raw_message, str) or not raw_message.strip():
            return jsonify({
                "success": False,
                "error": "The input text cannot be empty."
            }), 400

        # 1. Preprocess the raw email text using Scikit-Learn stopwords + PorterStemmer
        processed_message = preprocess_text(raw_message)
        
        # 2. Transform the preprocessed text using TF-IDF vectorizer
        features = vectorizer.transform([processed_message])
        
        # 3. Predict the category
        prediction = int(model.predict(features)[0]) # 0 (Ham) or 1 (Spam)
        
        # 4. Get the prediction probability (confidence)
        probabilities = model.predict_proba(features)[0]
        confidence = float(probabilities[prediction]) * 100
        
        # Convert numeric prediction to user-friendly label
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
    global model, vectorizer
    if model is None or vectorizer is None:
        load_ml_assets()
    model_path = os.path.join(base_dir, "model.pkl")
    vectorizer_path = os.path.join(base_dir, "vectorizer.pkl")
    is_trained = os.path.exists(model_path) and os.path.exists(vectorizer_path)
    return jsonify({
        "model_trained": is_trained and (model is not None and vectorizer is not None),
        "model_type": str(type(model).__name__) if model is not None else "None"
    })

# Expose WSGI application for serverless platforms like Vercel
application = app

if __name__ == '__main__':
    # Start the Flask app locally on port 5000 in debug mode
    app.run(host='127.0.0.1', port=5000, debug=True)
