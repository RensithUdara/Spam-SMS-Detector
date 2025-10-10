import pandas as pd
import numpy as np
import re
import pickle
import logging
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SpamDetectorTrainer:
    """Enhanced Spam Detector Training with multiple algorithms and preprocessing"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_vectorizer = None
        self.best_score = 0
        
    def preprocess_text(self, text):
        """Advanced text preprocessing"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove phone numbers (both local and international)
        text = re.sub(r'\b\d{10,15}\b', 'PHONE_NUMBER', text)
        text = re.sub(r'\b0\d{9}\b', 'PHONE_NUMBER', text)
        
        # Replace numbers with placeholder
        text = re.sub(r'\b\d+\b', 'NUMBER', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.,!?]', '', text)
        
        return text.strip()
    
    def load_and_prepare_data(self):
        """Load and prepare datasets with validation"""
        logger.info("Loading datasets...")
        
        try:
            # Load datasets
            df1 = pd.read_csv('SMSSpamCollection.tsv', sep='\t', names=['label', 'message'])
            df2 = pd.read_csv('SinhalaSpamCollection.tsv', sep='\t', names=['label', 'message'])
            
            logger.info(f"English dataset: {len(df1)} samples")
            logger.info(f"Sinhala dataset: {len(df2)} samples")
            
            # Combine datasets
            df = pd.concat([df1, df2], ignore_index=True)
            
            # Data validation and cleaning
            initial_size = len(df)
            df = df.dropna()
            df = df[df['message'].str.len() > 0]  # Remove empty messages
            
            logger.info(f"Removed {initial_size - len(df)} invalid samples")
            
            # Preprocess messages
            logger.info("Preprocessing text data...")
            df['message'] = df['message'].apply(self.preprocess_text)
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['message'])
            
            # Shuffle data
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Convert labels
            df['label'] = df['label'].map({'ham': 0, 'spam': 1})
            
            # Check for class imbalance
            class_counts = df['label'].value_counts()
            logger.info(f"Class distribution - Ham: {class_counts[0]}, Spam: {class_counts[1]}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def create_models(self):
        """Create different model configurations"""
        
        # Define vectorizer configurations
        vectorizers = {
            'tfidf_basic': TfidfVectorizer(max_features=5000, stop_words='english'),
            'tfidf_ngram': TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english'),
            'tfidf_char': TfidfVectorizer(max_features=8000, analyzer='char', ngram_range=(2, 4))
        }
        
        # Define classifiers
        classifiers = {
            'naive_bayes': MultinomialNB(alpha=0.1),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': SVC(kernel='linear', probability=True, random_state=42)
        }
        
        # Create model combinations
        models = {}
        for vec_name, vectorizer in vectorizers.items():
            for clf_name, classifier in classifiers.items():
                model_name = f"{vec_name}_{clf_name}"
                models[model_name] = Pipeline([
                    ('vectorizer', vectorizer),
                    ('classifier', classifier)
                ])
        
        return models
    
    def train_and_evaluate(self, df):
        """Train and evaluate all models"""
        logger.info("Training and evaluating models...")
        
        X = df['message']
        y = df['label']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        models = self.create_models()
        results = {}
        
        for model_name, model in models.items():
            logger.info(f"Training {model_name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                
                # Test predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
                
                results[model_name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba
                }
                
                logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, CV: {cv_scores.mean():.4f}")
                
                # Update best model
                if accuracy > self.best_score:
                    self.best_score = accuracy
                    self.best_model = model
                    self.best_vectorizer = model.named_steps['vectorizer']
                    
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
        
        return results, X_test, y_test
    
    def save_models(self):
        """Save the best model and vectorizer"""
        logger.info(f"Saving best model with accuracy: {self.best_score:.4f}")
        
        # Save full pipeline (best model)
        with open('model.pkl', 'wb') as f:
            pickle.dump(self.best_model, f)
        
        # Save vectorizer separately for compatibility
        with open('vectorizer.pkl', 'wb') as f:
            pickle.dump(self.best_vectorizer, f)
        
        # Save model metadata
        metadata = {
            'accuracy': self.best_score,
            'training_date': datetime.now().isoformat(),
            'model_type': str(type(self.best_model.named_steps['classifier']).__name__)
        }
        
        with open('model_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
    
    def generate_report(self, results, X_test, y_test):
        """Generate detailed training report"""
        logger.info("Generating training report...")
        
        report = []
        report.append("="*80)
        report.append("SPAM DETECTOR TRAINING REPORT")
        report.append("="*80)
        report.append(f"Training Date: {datetime.now()}")
        report.append(f"Total Test Samples: {len(X_test)}")
        report.append("")
        
        # Sort results by accuracy
        sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        for model_name, metrics in sorted_results:
            report.append(f"Model: {model_name}")
            report.append(f"  Accuracy: {metrics['accuracy']:.4f}")
            report.append(f"  Precision: {metrics['precision']:.4f}")
            report.append(f"  Recall: {metrics['recall']:.4f}")
            report.append(f"  F1-Score: {metrics['f1']:.4f}")
            report.append(f"  CV Mean: {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
            report.append("")
        
        # Best model details
        best_name = sorted_results[0][0]
        best_metrics = sorted_results[0][1]
        
        report.append("BEST MODEL CLASSIFICATION REPORT:")
        report.append("-" * 50)
        report.append(classification_report(y_test, best_metrics['y_pred']))
        
        report.append("\nCONFUSION MATRIX:")
        report.append("-" * 20)
        cm = confusion_matrix(y_test, best_metrics['y_pred'])
        report.append(f"[[{cm[0][0]}, {cm[0][1]}],")
        report.append(f" [{cm[1][0]}, {cm[1][1]}]]")
        
        # Save report
        with open('training_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        # Print summary
        print('\n'.join(report))

def main():
    """Main training function"""
    logger.info("Starting spam detector training...")
    
    trainer = SpamDetectorTrainer()
    
    # Load and prepare data
    df = trainer.load_and_prepare_data()
    
    # Train and evaluate models
    results, X_test, y_test = trainer.train_and_evaluate(df)
    
    # Save best model
    trainer.save_models()
    
    # Generate report
    trainer.generate_report(results, X_test, y_test)
    
    logger.info("✅ Training completed successfully!")

if __name__ == "__main__":
    main()
