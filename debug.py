#!/usr/bin/env python3
"""
Debug script to identify and fix the prediction issues
"""

import pickle
import sys
import traceback

def test_model_loading():
    """Test loading the model and vectorizer"""
    print("Testing model loading...")
    
    try:
        # Load model
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Model loaded: {type(model)}")
        
        # Load vectorizer
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        print(f"✅ Vectorizer loaded: {type(vectorizer)}")
        
        return model, vectorizer
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return None, None

def test_text_preprocessing():
    """Test text preprocessing with Sinhala characters"""
    print("\nTesting text preprocessing...")
    
    test_messages = [
        "Hello world",
        "අද ඔයාගේ class එකට late වෙලාද?",
        "FREE MONEY! Call 123-456-7890",
        "",
        "   ",
    ]
    
    import re
    
    def preprocess_text(text):
        """Same preprocessing as in app"""
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
    
    for msg in test_messages:
        try:
            processed = preprocess_text(msg)
            print(f"Original: '{msg}' -> Processed: '{processed}'")
        except Exception as e:
            print(f"❌ Preprocessing failed for '{msg}': {e}")

def test_predictions(model, vectorizer):
    """Test making predictions"""
    print("\nTesting predictions...")
    
    if not model or not vectorizer:
        print("❌ Cannot test predictions - models not loaded")
        return
    
    test_messages = [
        "hello world",
        "free money call now",
        "meeting at 3pm",
        "empty message"  # fallback case
    ]
    
    for msg in test_messages:
        try:
            print(f"\nTesting: '{msg}'")
            
            # Check if it's a pipeline
            if hasattr(model, 'named_steps'):
                print("  Using pipeline model")
                prediction = model.predict([msg])[0]
                if hasattr(model, 'predict_proba'):
                    confidence = model.predict_proba([msg])[0]
                    spam_confidence = confidence[1] if len(confidence) > 1 else confidence[0]
                else:
                    spam_confidence = 0.5
            else:
                print("  Using separate vectorizer and model")
                message_vector = vectorizer.transform([msg])
                print(f"  Vector shape: {message_vector.shape}")
                prediction = model.predict(message_vector)[0]
                if hasattr(model, 'predict_proba'):
                    confidence = model.predict_proba(message_vector)[0]
                    spam_confidence = confidence[1] if len(confidence) > 1 else confidence[0]
                else:
                    spam_confidence = 0.5
            
            result = 'spam' if prediction == 1 else 'ham'
            print(f"  ✅ Prediction: {result} (confidence: {spam_confidence:.3f})")
            
        except Exception as e:
            print(f"  ❌ Prediction failed: {e}")
            traceback.print_exc()

def main():
    """Main debug function"""
    print("🔍 Spam Detector Debug Script")
    print("=" * 50)
    
    # Test model loading
    model, vectorizer = test_model_loading()
    
    # Test text preprocessing
    test_text_preprocessing()
    
    # Test predictions
    test_predictions(model, vectorizer)
    
    print("\n" + "=" * 50)
    print("Debug complete!")

if __name__ == "__main__":
    main()