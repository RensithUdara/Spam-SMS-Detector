#!/usr/bin/env python3
"""
Quick start script for the Spam Detector application
Handles initial setup, model training, and application launch
"""

import os
import sys
import subprocess
import pickle
import argparse
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    try:
        import flask
        import sklearn
        import pandas
        import numpy
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def check_model_files():
    """Check if model files exist"""
    model_files = ['model.pkl', 'vectorizer.pkl']
    missing_files = [f for f in model_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"⚠️ Missing model files: {', '.join(missing_files)}")
        return False
    else:
        print("✅ Model files found")
        return True

def check_data_files():
    """Check if training data files exist"""
    data_files = ['SMSSpamCollection.tsv', 'SinhalaSpamCollection.tsv']
    missing_files = [f for f in data_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing data files: {', '.join(missing_files)}")
        return False
    else:
        print("✅ Training data files found")
        return True

def train_model():
    """Train the spam detection model"""
    print("🚀 Training spam detection model...")
    try:
        result = subprocess.run([sys.executable, 'train_model.py'], 
                              capture_output=True, text=True, check=True)
        print("✅ Model training completed successfully")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Model training failed: {e}")
        print(e.stderr)
        return False

def evaluate_model():
    """Evaluate the trained model"""
    print("📊 Evaluating model performance...")
    try:
        result = subprocess.run([sys.executable, 'evaluate_model.py'], 
                              capture_output=True, text=True, check=True)
        print("✅ Model evaluation completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Model evaluation failed: {e}")
        return False

def setup_directories():
    """Create necessary directories"""
    directories = ['logs', 'models', 'data']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    print("✅ Directories created")

def create_env_file():
    """Create .env file if it doesn't exist"""
    if not os.path.exists('.env'):
        if os.path.exists('.env.example'):
            import shutil
            shutil.copy('.env.example', '.env')
            print("✅ Created .env file from template")
            print("🔧 Please customize .env file for your environment")
        else:
            print("⚠️ .env.example not found, skipping .env creation")

def run_app():
    """Run the Flask application"""
    print("🌟 Starting Spam Detector application...")
    print("🌐 Application will be available at: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop the application")
    
    try:
        subprocess.run([sys.executable, 'app.py'], check=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Application failed to start: {e}")

def main():
    parser = argparse.ArgumentParser(description='Spam Detector Quick Start')
    parser.add_argument('--train-only', action='store_true', 
                       help='Only train the model, don\'t run the app')
    parser.add_argument('--evaluate-only', action='store_true', 
                       help='Only evaluate the model, don\'t run the app')
    parser.add_argument('--skip-checks', action='store_true', 
                       help='Skip initial checks and setup')
    args = parser.parse_args()

    print("🛡️ Spam Detector Quick Start")
    print("=" * 50)
    
    if not args.skip_checks:
        # Initial checks
        if not check_requirements():
            sys.exit(1)
        
        setup_directories()
        create_env_file()
        
        # Check for training data
        if not check_data_files():
            print("\n📥 Please ensure training data files are present:")
            print("  - SMSSpamCollection.tsv")
            print("  - SinhalaSpamCollection.tsv")
            sys.exit(1)
        
        # Check for model files
        if not check_model_files():
            print("\n🤖 Model files not found. Training new model...")
            if not train_model():
                sys.exit(1)
    
    # Handle specific operations
    if args.train_only:
        train_model()
        return
    
    if args.evaluate_only:
        if check_model_files():
            evaluate_model()
        else:
            print("❌ Model files not found. Please train the model first.")
        return
    
    # Evaluate model if files exist
    if check_model_files():
        evaluate_model()
    
    # Run the application
    print("\n🚀 All checks passed! Starting application...")
    run_app()

if __name__ == "__main__":
    main()