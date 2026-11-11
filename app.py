import os
import re
import pickle
from urllib.parse import parse_qs, unquote
from flask import Flask, render_template, request, jsonify

# Import custom text preprocessing module
from preprocess import preprocess_text

# Resolve absolute base directory of app.py
base_dir = os.path.dirname(os.path.abspath(__file__))

# Resolve templates and static folders with multi-directory fallback
template_dir = None
static_dir = None

for d in [base_dir, os.path.dirname(base_dir), os.getcwd()]:
    t_cand = os.path.join(d, 'templates')
    s_cand = os.path.join(d, 'static')
    if not template_dir and os.path.isdir(t_cand):
        template_dir = t_cand
    if not static_dir and os.path.isdir(s_cand):
        static_dir = s_cand

if not template_dir:
    template_dir = os.path.join(base_dir, 'templates')
if not static_dir:
    static_dir = os.path.join(base_dir, 'static')

# Configure Flask with explicit paths
app = Flask(
    __name__,
    template_folder=template_dir,
    static_folder=static_dir
)

# Global variables for model and vectorizer
model = None
vectorizer = None
model_load_error = None

def load_ml_assets():
    """
    Loads the trained model and TF-IDF vectorizer pickles using absolute paths.
    Searches multiple potential locations to ensure reliability in serverless environments.
    """
    global model, vectorizer, model_load_error
    
    search_dirs = [
        base_dir,                           # Directory containing app.py
        os.path.dirname(base_dir),          # Parent directory
        os.getcwd(),                        # Current working directory (/var/task on Vercel)
        os.path.join(os.getcwd(), 'api'),   # api subdirectory
    ]

    model_path = None
    vectorizer_path = None

    for directory in search_dirs:
        m_p = os.path.join(directory, "model.pkl")
        v_p = os.path.join(directory, "vectorizer.pkl")
        if os.path.exists(m_p) and os.path.exists(v_p):
            model_path = m_p
            vectorizer_path = v_p
            break

    if model_path and vectorizer_path:
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
        model_load_error = f"Pickle files not found on disk in any search directory: {search_dirs}"
        print(model_load_error)
        return False

# Attempt initial asset load on startup
assets_loaded = load_ml_assets()

# WSGI Middleware to normalize PATH_INFO for Vercel serverless rewrites
class VercelPathRewriteMiddleware:
    """
    WSGI middleware that normalizes PATH_INFO when deployed behind Vercel serverless rewrites.
    Resolves the original request path from query parameters, routing headers, or HTTP method.
    """
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        raw_path = environ.get('PATH_INFO', '')
        query_string = environ.get('QUERY_STRING', '')
        request_method = environ.get('REQUEST_METHOD', 'GET').upper()
        
        resolved_path = None
        
        # 1. Check if route was forwarded via __route__ query parameter in vercel.json
        if '__route__=' in query_string:
            params = parse_qs(query_string, keep_blank_values=True)
            if '__route__' in params and params['__route__']:
                val = params['__route__'][0]
                if val:
                    resolved_path = val if val.startswith('/') else f"/{val}"
                    # Remove __route__ from QUERY_STRING so application receives clean query parameters
                    clean_params = {k: v for k, v in params.items() if k != '__route__'}
                    new_qs = []
                    for k, v_list in clean_params.items():
                        for v_item in v_list:
                            new_qs.append(f"{k}={v_item}")
                    environ['QUERY_STRING'] = '&'.join(new_qs)

        # 2. Check x-now-route-matches header if present
        if not resolved_path:
            route_matches = environ.get('HTTP_X_NOW_ROUTE_MATCHES', '')
            if route_matches:
                decoded = unquote(route_matches)
                matches = dict(re.findall(r"([^&=]+)=([^&]*)", decoded))
                if '1' in matches and matches['1']:
                    val = matches['1']
                    resolved_path = val if val.startswith('/') else f"/{val}"

        # 3. Check standard URI headers
        if not resolved_path:
            for header in ('REQUEST_URI', 'RAW_URI'):
                val = environ.get(header, '')
                if val:
                    clean = val.split('?')[0]
                    if clean and clean not in ('/api', '/api/', '/api/index', '/api/index.py'):
                        resolved_path = clean
                        break

        # 4. Apply resolved path or provide robust serverless defaults
        if resolved_path:
            environ['PATH_INFO'] = resolved_path
        elif not raw_path or raw_path in ('/api', '/api/', '/api/index', '/api/index.py'):
            # If path resolved to the serverless function root:
            # POST requests default to prediction endpoint; GET requests default to home
            if request_method == 'POST':
                environ['PATH_INFO'] = '/predict'
            else:
                environ['PATH_INFO'] = '/'

        return self.wsgi_app(environ, start_response)

# Apply middleware to Flask WSGI application
app.wsgi_app = VercelPathRewriteMiddleware(app.wsgi_app)

# JSON Error Handlers (Ensures API requests always receive JSON, never raw HTML)
@app.errorhandler(400)
def handle_bad_request(e):
    return jsonify({
        "success": False,
        "error": getattr(e, 'description', 'Bad Request')
    }), 400

@app.errorhandler(404)
def handle_not_found(e):
    if request.path.startswith('/api') or request.path.startswith('/predict') or request.method == 'POST' or request.is_json:
        return jsonify({
            "success": False,
            "error": f"API endpoint not found: {request.path}"
        }), 404
    return render_template('index.html', assets_loaded=(model is not None and vectorizer is not None)), 200

@app.errorhandler(405)
def handle_method_not_allowed(e):
    return jsonify({
        "success": False,
        "error": f"Method {request.method} not allowed for {request.path}"
    }), 405

@app.errorhandler(500)
def handle_server_error(e):
    return jsonify({
        "success": False,
        "error": "Internal server error occurred."
    }), 500

@app.errorhandler(Exception)
def handle_unhandled_exception(e):
    return jsonify({
        "success": False,
        "error": f"Server error: {str(e)}"
    }), 500

@app.route('/', methods=['GET'])
@app.route('/index', methods=['GET'])
@app.route('/api', methods=['GET'])
@app.route('/api/index', methods=['GET'])
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
@app.route('/api/predict', methods=['POST'])
@app.route('/api', methods=['POST'])
@app.route('/api/index', methods=['POST'])
@app.route('/', methods=['POST'])
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
@app.route('/api/status', methods=['GET'])
def get_status():
    """
    Endpoint to check training and model availability status.
    """
    global model, vectorizer
    if model is None or vectorizer is None:
        load_ml_assets()
    
    has_model = (model is not None and vectorizer is not None)
    return jsonify({
        "model_trained": has_model,
        "model_type": str(type(model).__name__) if model is not None else "None"
    })

# Expose WSGI application for serverless platforms like Vercel
application = app
handler = app

if __name__ == '__main__':
    # Start the Flask app locally on port 5000 in debug mode
    app.run(host='127.0.0.1', port=5000, debug=True)
