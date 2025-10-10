#!/usr/bin/env python3
"""
Enhanced Model Evaluation Script for Spam Detector
Provides comprehensive analysis of model performance
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import learning_curve, validation_curve
import logging
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation toolkit"""
    
    def __init__(self, model_path='model.pkl', vectorizer_path='vectorizer.pkl'):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.model = None
        self.vectorizer = None
        self.test_data = None
        self.predictions = None
        self.probabilities = None
        
    def load_model(self):
        """Load the trained model and vectorizer"""
        try:
            # Load model
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info("Model loaded successfully")
            
            # Load vectorizer (if separate)
            if os.path.exists(self.vectorizer_path):
                with open(self.vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                logger.info("Vectorizer loaded successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def load_test_data(self):
        """Load and prepare test data"""
        try:
            # Load datasets
            df1 = pd.read_csv('SMSSpamCollection.tsv', sep='\t', names=['label', 'message'])
            df2 = pd.read_csv('SinhalaSpamCollection.tsv', sep='\t', names=['label', 'message'])
            
            # Combine and process
            df = pd.concat([df1, df2], ignore_index=True)
            df = df.dropna()
            df['label'] = df['label'].map({'ham': 0, 'spam': 1})
            
            # Use last 20% as test data (deterministic split)
            test_size = int(len(df) * 0.2)
            self.test_data = df.tail(test_size).reset_index(drop=True)
            
            logger.info(f"Test data loaded: {len(self.test_data)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Error loading test data: {str(e)}")
            return False
    
    def make_predictions(self):
        """Make predictions on test data"""
        try:
            X_test = self.test_data['message']
            y_test = self.test_data['label']
            
            # Check if model is a pipeline or separate components
            if hasattr(self.model, 'predict'):
                # Pipeline model
                self.predictions = self.model.predict(X_test)
                if hasattr(self.model, 'predict_proba'):
                    self.probabilities = self.model.predict_proba(X_test)[:, 1]
            else:
                # Separate vectorizer and model
                X_test_vec = self.vectorizer.transform(X_test)
                self.predictions = self.model.predict(X_test_vec)
                if hasattr(self.model, 'predict_proba'):
                    self.probabilities = self.model.predict_proba(X_test_vec)[:, 1]
            
            logger.info("Predictions completed")
            return True
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return False
    
    def calculate_metrics(self):
        """Calculate comprehensive performance metrics"""
        y_true = self.test_data['label']
        y_pred = self.predictions
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = recall  # Same as recall
        
        # ROC AUC (if probabilities available)
        if self.probabilities is not None:
            roc_auc = roc_auc_score(y_true, self.probabilities)
            avg_precision = average_precision_score(y_true, self.probabilities)
        else:
            roc_auc = None
            avg_precision = None
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'roc_auc': roc_auc,
            'avg_precision': avg_precision,
            'confusion_matrix': cm,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        }
        
        return metrics
    
    def generate_plots(self, metrics):
        """Generate visualization plots"""
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Confusion Matrix
        cm = metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        axes[0, 0].set_xticklabels(['Ham', 'Spam'])
        axes[0, 0].set_yticklabels(['Ham', 'Spam'])
        
        # 2. Metrics Bar Chart
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']
        metric_values = [
            metrics['accuracy'], metrics['precision'], metrics['recall'],
            metrics['f1_score'], metrics['specificity']
        ]
        
        bars = axes[0, 1].bar(metric_names, metric_values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum'])
        axes[0, 1].set_title('Performance Metrics')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # 3. ROC Curve (if probabilities available)
        if self.probabilities is not None:
            fpr, tpr, _ = roc_curve(self.test_data['label'], self.probabilities)
            axes[1, 0].plot(fpr, tpr, label=f'ROC Curve (AUC = {metrics["roc_auc"]:.3f})')
            axes[1, 0].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
            axes[1, 0].set_xlabel('False Positive Rate')
            axes[1, 0].set_ylabel('True Positive Rate')
            axes[1, 0].set_title('ROC Curve')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        else:
            axes[1, 0].text(0.5, 0.5, 'ROC Curve\nNot Available\n(No Probabilities)', 
                          ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # 4. Precision-Recall Curve (if probabilities available)
        if self.probabilities is not None:
            precision_vals, recall_vals, _ = precision_recall_curve(self.test_data['label'], self.probabilities)
            axes[1, 1].plot(recall_vals, precision_vals, label=f'PR Curve (AP = {metrics["avg_precision"]:.3f})')
            axes[1, 1].set_xlabel('Recall')
            axes[1, 1].set_ylabel('Precision')
            axes[1, 1].set_title('Precision-Recall Curve')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        else:
            axes[1, 1].text(0.5, 0.5, 'Precision-Recall\nCurve Not Available\n(No Probabilities)', 
                          ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plt.savefig('model_evaluation_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Plots saved to model_evaluation_plots.png")
    
    def analyze_errors(self):
        """Analyze misclassified examples"""
        y_true = self.test_data['label']
        y_pred = self.predictions
        
        # Find misclassified examples
        misclassified = self.test_data[y_true != y_pred].copy()
        misclassified['predicted'] = y_pred[y_true != y_pred]
        misclassified['actual'] = y_true[y_true != y_pred]
        
        # Separate false positives and false negatives
        false_positives = misclassified[misclassified['actual'] == 0]  # Ham classified as Spam
        false_negatives = misclassified[misclassified['actual'] == 1]  # Spam classified as Ham
        
        error_analysis = {
            'total_errors': len(misclassified),
            'false_positives': len(false_positives),
            'false_negatives': len(false_negatives),
            'fp_examples': false_positives.head(5),
            'fn_examples': false_negatives.head(5)
        }
        
        return error_analysis
    
    def generate_report(self, metrics, error_analysis):
        """Generate comprehensive evaluation report"""
        report = []
        report.append("="*80)
        report.append("SPAM DETECTOR MODEL EVALUATION REPORT")
        report.append("="*80)
        report.append(f"Evaluation Date: {datetime.now()}")
        report.append(f"Test Dataset Size: {len(self.test_data)} samples")
        report.append("")
        
        # Performance Metrics
        report.append("PERFORMANCE METRICS:")
        report.append("-" * 30)
        report.append(f"Accuracy:     {metrics['accuracy']:.4f}")
        report.append(f"Precision:    {metrics['precision']:.4f}")
        report.append(f"Recall:       {metrics['recall']:.4f}")
        report.append(f"F1-Score:     {metrics['f1_score']:.4f}")
        report.append(f"Specificity:  {metrics['specificity']:.4f}")
        
        if metrics['roc_auc']:
            report.append(f"ROC AUC:      {metrics['roc_auc']:.4f}")
        if metrics['avg_precision']:
            report.append(f"Avg Precision: {metrics['avg_precision']:.4f}")
        
        report.append("")
        
        # Confusion Matrix
        report.append("CONFUSION MATRIX:")
        report.append("-" * 20)
        cm = metrics['confusion_matrix']
        report.append("           Predicted")
        report.append("          Ham   Spam")
        report.append(f"Actual Ham  {cm[0,0]:4d}  {cm[0,1]:4d}")
        report.append(f"     Spam  {cm[1,0]:4d}  {cm[1,1]:4d}")
        report.append("")
        
        # Error Analysis
        report.append("ERROR ANALYSIS:")
        report.append("-" * 20)
        report.append(f"Total Misclassifications: {error_analysis['total_errors']}")
        report.append(f"False Positives (Ham→Spam): {error_analysis['false_positives']}")
        report.append(f"False Negatives (Spam→Ham): {error_analysis['false_negatives']}")
        report.append("")
        
        # Sample False Positives
        if len(error_analysis['fp_examples']) > 0:
            report.append("SAMPLE FALSE POSITIVES (Ham classified as Spam):")
            report.append("-" * 50)
            for _, row in error_analysis['fp_examples'].iterrows():
                message = row['message'][:100] + "..." if len(row['message']) > 100 else row['message']
                report.append(f"• {message}")
            report.append("")
        
        # Sample False Negatives
        if len(error_analysis['fn_examples']) > 0:
            report.append("SAMPLE FALSE NEGATIVES (Spam classified as Ham):")
            report.append("-" * 50)
            for _, row in error_analysis['fn_examples'].iterrows():
                message = row['message'][:100] + "..." if len(row['message']) > 100 else row['message']
                report.append(f"• {message}")
            report.append("")
        
        # Classification Report
        report.append("DETAILED CLASSIFICATION REPORT:")
        report.append("-" * 40)
        y_true = self.test_data['label']
        y_pred = self.predictions
        class_report = classification_report(y_true, y_pred, target_names=['Ham', 'Spam'])
        report.append(class_report)
        
        # Save report
        report_text = '\n'.join(report)
        with open('model_evaluation_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        return report_text
    
    def run_evaluation(self):
        """Run complete evaluation pipeline"""
        logger.info("Starting model evaluation...")
        
        # Load model and data
        if not self.load_model():
            return False
        
        if not self.load_test_data():
            return False
        
        # Make predictions
        if not self.make_predictions():
            return False
        
        # Calculate metrics
        metrics = self.calculate_metrics()
        
        # Analyze errors
        error_analysis = self.analyze_errors()
        
        # Generate visualizations
        try:
            self.generate_plots(metrics)
        except Exception as e:
            logger.warning(f"Could not generate plots: {str(e)}")
        
        # Generate report
        report = self.generate_report(metrics, error_analysis)
        
        # Print summary
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"Total Errors: {error_analysis['total_errors']}")
        print("\nFull report saved to: model_evaluation_report.txt")
        if os.path.exists('model_evaluation_plots.png'):
            print("Plots saved to: model_evaluation_plots.png")
        
        logger.info("✅ Evaluation completed successfully!")
        return True

def main():
    """Main evaluation function"""
    evaluator = ModelEvaluator()
    evaluator.run_evaluation()

if __name__ == "__main__":
    main()