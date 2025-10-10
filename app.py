from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import pickle
import logging
import os
import re
from datetime import datetime
import numpy as np
from functools import wraps
import uuid

# Setup logging with UTF-8 encoding for Unicode support
import sys

# Configure file handler with UTF-8 encoding
file_handler = logging.FileHandler('app.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Configure console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# Setup root logger
logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')
CORS(app)  # Enable CORS for API endpoints

class SpamDetector:
    """Enhanced Spam Detector with improved functionality"""
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.metadata = None
        self.load_models()
    
    def load_models(self):
        """Load trained models with error handling"""
        try:
            # Load model (pipeline or standalone)
            if os.path.exists('model.pkl'):
                with open('model.pkl', 'rb') as f:
                    self.model = pickle.load(f)
                logger.info("Model loaded successfully")
            else:
                logger.error("Model file not found. Please train the model first.")
                raise FileNotFoundError("Model not found")
            
            # Load vectorizer (for compatibility)
            if os.path.exists('vectorizer.pkl'):
                with open('vectorizer.pkl', 'rb') as f:
                    self.vectorizer = pickle.load(f)
                logger.info("Vectorizer loaded successfully")
            
            # Load metadata if available
            if os.path.exists('model_metadata.pkl'):
                with open('model_metadata.pkl', 'rb') as f:
                    self.metadata = pickle.load(f)
                logger.info(f"Model metadata loaded: {self.metadata}")
                    
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
    
    def preprocess_text(self, text):
        """Preprocess text (same as training)"""
        if not text or text.strip() == "":
            return ""
        
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\b\d{10,15}\b', 'PHONE_NUMBER', text)
        text = re.sub(r'\b0\d{9}\b', 'PHONE_NUMBER', text)
        text = re.sub(r'\b\d+\b', 'NUMBER', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.,!?]', '', text)
        return text.strip()
    
    def predict(self, message):
        """Make prediction with confidence score"""
        try:
            # Validate input
            if not message or message.strip() == "":
                raise ValueError("Message cannot be empty")
            
            if len(message) > 1000:
                raise ValueError("Message too long (max 1000 characters)")
            
            # Preprocess message
            processed_message = self.preprocess_text(message)
            
            # Ensure we have some text to work with
            if not processed_message or processed_message.strip() == "":
                processed_message = "empty message"
            
            # Check if model is a pipeline or separate components
            if hasattr(self.model, 'predict') and hasattr(self.model, 'named_steps'):
                # Pipeline model
                prediction = self.model.predict([processed_message])[0]
                if hasattr(self.model, 'predict_proba'):
                    confidence = self.model.predict_proba([processed_message])[0]
                    spam_confidence = confidence[1] if len(confidence) > 1 else confidence[0]
                else:
                    spam_confidence = 0.5  # Default when no probability available
            else:
                # Separate vectorizer and model
                message_vector = self.vectorizer.transform([processed_message])
                prediction = self.model.predict(message_vector)[0]
                if hasattr(self.model, 'predict_proba'):
                    confidence = self.model.predict_proba(message_vector)[0]
                    spam_confidence = confidence[1] if len(confidence) > 1 else confidence[0]
                else:
                    spam_confidence = 0.5
            
            result = {
                'prediction': 'spam' if prediction == 1 else 'ham',
                'confidence': float(spam_confidence),
                'is_spam': bool(prediction == 1),
                'processed_message': processed_message,
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            # Safely handle Unicode characters in error messages
            try:
                error_msg = str(e).encode('ascii', errors='replace').decode('ascii')
                logger.error(f"Prediction error: {error_msg}")
                if 'processed_message' in locals():
                    logger.error(f"Message length: {len(processed_message)}")
                    logger.error(f"Message preview: {processed_message[:50]}...")
            except Exception:
                # Fallback if even logging fails
                logger.error("Prediction error occurred but could not log details")
            raise

# Initialize detector
detector = SpamDetector()

def log_request(f):
    """Decorator to log requests"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        request_id = str(uuid.uuid4())[:8]
        logger.info(f"Request {request_id}: {request.method} {request.endpoint}")
        try:
            result = f(*args, **kwargs)
            logger.info(f"Request {request_id}: Success")
            return result
        except Exception as e:
            logger.error(f"Request {request_id}: Error - {str(e)}")
            raise
    return decorated_function

@app.route('/')
@log_request
def home():
    """Home page"""
    # Initialize session for message history
    if 'message_history' not in session:
        session['message_history'] = []
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@log_request
def predict():
    """Web form prediction endpoint"""
    try:
        message = request.form.get('message', '').strip()
        
        if not message:
            return render_template('index.html', 
                                 error="Please enter a message to analyze")
        
        # Make prediction
        result = detector.predict(message)
        
        # Add to session history
        if 'message_history' not in session:
            session['message_history'] = []
        
        session['message_history'].insert(0, {
            'message': message,
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'timestamp': result['timestamp']
        })
        
        # Keep only last 10 messages
        session['message_history'] = session['message_history'][:10]
        session.modified = True
        
        # Format for web display
        prediction_text = "Spam" if result['is_spam'] else "Not Spam"
        
        return render_template('index.html', 
                             prediction=prediction_text,
                             message=message,
                             confidence=result['confidence'],
                             history=session['message_history'])
        
    except ValueError as e:
        return render_template('index.html', error=str(e))
    except Exception as e:
        logger.error(f"Web prediction error: {str(e)}")
        return render_template('index.html', 
                             error="An error occurred during prediction. Please try again.")

@app.route('/api/predict', methods=['POST'])
@log_request
def api_predict():
    """REST API prediction endpoint"""
    try:
        # Check content type
        if not request.is_json:
            return jsonify({
                'error': 'Content-Type must be application/json',
                'status': 'error'
            }), 400
        
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({
                'error': 'Message field is required',
                'status': 'error'
            }), 400
        
        # Make prediction
        result = detector.predict(message)
        result['status'] = 'success'
        
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400
    except Exception as e:
        logger.error(f"API prediction error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

@app.route('/api/batch', methods=['POST'])
@log_request
def api_batch_predict():
    """Batch prediction API endpoint"""
    try:
        if not request.is_json:
            return jsonify({
                'error': 'Content-Type must be application/json',
                'status': 'error'
            }), 400
        
        data = request.get_json()
        messages = data.get('messages', [])
        
        if not isinstance(messages, list):
            return jsonify({
                'error': 'Messages must be a list',
                'status': 'error'
            }), 400
        
        if len(messages) == 0:
            return jsonify({
                'error': 'At least one message is required',
                'status': 'error'
            }), 400
        
        if len(messages) > 100:
            return jsonify({
                'error': 'Maximum 100 messages allowed per batch',
                'status': 'error'
            }), 400
        
        results = []
        for i, message in enumerate(messages):
            try:
                if not isinstance(message, str):
                    results.append({
                        'index': i,
                        'error': 'Message must be a string',
                        'status': 'error'
                    })
                    continue
                
                result = detector.predict(message)
                result['index'] = i
                result['status'] = 'success'
                results.append(result)
                
            except Exception as e:
                results.append({
                    'index': i,
                    'error': str(e),
                    'status': 'error'
                })
        
        return jsonify({
            'results': results,
            'total': len(messages),
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

@app.route('/api/model/info')
@log_request
def api_model_info():
    """Get model information"""
    try:
        info = {
            'status': 'loaded',
            'model_type': str(type(detector.model).__name__),
            'has_probability': hasattr(detector.model, 'predict_proba'),
            'metadata': detector.metadata
        }
        
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"Model info error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

@app.route('/api/health')
@log_request
def api_health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': detector.model is not None
    })

@app.route('/history')
@log_request
def history():
    """View message history"""
    history = session.get('message_history', [])
    return render_template('index.html', history=history, show_history=True)

@app.route('/clear_history', methods=['POST'])
@log_request
def clear_history():
    """Clear message history"""
    session['message_history'] = []
    session.modified = True
    return render_template('index.html', message="History cleared successfully")

@app.errorhandler(404)
def not_found(error):
    """404 error handler"""
    return jsonify({
        'error': 'Endpoint not found',
        'status': 'error'
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    """405 error handler"""
    return jsonify({
        'error': 'Method not allowed',
        'status': 'error'
    }), 405

@app.errorhandler(500)
def internal_error(error):
    """500 error handler"""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        'error': 'Internal server error',
        'status': 'error'
    }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'True').lower() == 'true'
    
    logger.info(f"Starting spam detector app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
