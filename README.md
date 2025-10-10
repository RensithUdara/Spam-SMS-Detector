# 🛡️ Advanced Spam Detector

An enhanced SMS/Email spam detection system with machine learning capabilities, supporting both English and Sinhala languages. Features a modern web interface with radar-style animations, REST API, confidence scoring, and comprehensive model evaluation tools.

## ✨ Features

### Core Functionality
- **Multi-language Support**: English and Sinhala spam detection
- **Advanced ML Models**: Multiple algorithms comparison (Naive Bayes, Random Forest, Logistic Regression, SVM)
- **Confidence Scoring**: Get prediction confidence levels
- **Real-time Processing**: Instant spam detection through web interface
- **Batch Processing**: Analyze multiple messages simultaneously via API

### Web Interface
- **Modern UI**: Radar-themed design with animated backgrounds
- **Message History**: Track previous analyses with confidence scores
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Real-time Feedback**: Animated confidence bars and status indicators

### API Features
- **REST API**: JSON-based endpoints for integration
- **Batch Processing**: Analyze up to 100 messages per request
- **Error Handling**: Comprehensive error responses
- **Health Checks**: Monitor application status
- **CORS Support**: Cross-origin requests enabled

### Advanced Features
- **Model Evaluation**: Comprehensive performance metrics and visualizations
- **Configuration Management**: Flexible settings via files or environment variables
- **Logging**: Rotating logs with configurable levels
- **Text Preprocessing**: Advanced cleaning and normalization
- **Cross-validation**: Robust model training and evaluation

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/spam-detector.git
cd spam-detector
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Train the model** (if model files don't exist)
```bash
python train_model.py
```

4. **Run the application**
```bash
python app.py
```

5. **Open in browser**
Navigate to `http://localhost:5000`

## 📖 Usage

### Web Interface

1. **Single Message Analysis**
   - Enter your message in the text area
   - Click "🔍 INITIATE SCAN"
   - View results with confidence score
   - Check analysis history

2. **Message History**
   - Previous analyses are automatically saved
   - View up to 10 recent messages
   - Clear history anytime

### REST API

#### Single Message Prediction
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Congratulations! You won $1000! Call now!"
  }'
```

**Response:**
```json
{
  "prediction": "spam",
  "confidence": 0.87,
  "is_spam": true,
  "processed_message": "congratulations you won NUMBER call now",
  "timestamp": "2024-01-15T10:30:00",
  "status": "success"
}
```

#### Batch Processing
```bash
curl -X POST http://localhost:5000/api/batch \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      "Hello, how are you?",
      "Win money fast! Click here!",
      "Meeting at 3pm today"
    ]
  }'
```

#### Model Information
```bash
curl http://localhost:5000/api/model/info
```

#### Health Check
```bash
curl http://localhost:5000/api/health
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file from the template:
```bash
cp .env.example .env
```

Key settings:
- `DEBUG`: Enable/disable debug mode
- `SECRET_KEY`: Session security key (change in production!)
- `PORT`: Application port (default: 5000)
- `LOG_LEVEL`: Logging verbosity (DEBUG, INFO, WARNING, ERROR)
- `MAX_BATCH_SIZE`: Maximum messages per batch request
- `CORS_ORIGINS`: Allowed origins for API requests

### Configuration File

Customize `config.json` for advanced settings:
```json
{
  "model": {
    "max_message_length": 1000,
    "confidence_threshold": 0.5
  },
  "api": {
    "max_batch_size": 100,
    "rate_limit_per_minute": 60
  },
  "logging": {
    "level": "INFO",
    "file_path": "logs/spam_detector.log",
    "backup_count": 5
  }
}
```

## 🧠 Model Training

### Basic Training
```bash
python train_model.py
```

### Advanced Options

The training script automatically:
- Combines English and Sinhala datasets
- Preprocesses text (URLs, phone numbers, normalization)
- Compares multiple algorithms
- Performs cross-validation
- Generates comprehensive reports

Training outputs:
- `model.pkl`: Best performing model
- `vectorizer.pkl`: Text vectorizer
- `model_metadata.pkl`: Model information
- `training_report.txt`: Detailed performance metrics
- `model_training.log`: Training logs

### Custom Training

Modify `train_model.py` to:
- Add new datasets
- Adjust preprocessing
- Include additional algorithms
- Change evaluation metrics

## 📊 Model Evaluation

Evaluate model performance:
```bash
python evaluate_model.py
```

Generates:
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score
- **Visualizations**: ROC curves, confusion matrix, precision-recall curves
- **Error Analysis**: Misclassified examples with explanations
- **Detailed Report**: Comprehensive evaluation summary

Output files:
- `model_evaluation_report.txt`: Detailed metrics and analysis
- `model_evaluation_plots.png`: Performance visualizations

## 🏗️ Architecture

### Project Structure
```
spam-detector/
├── app.py                    # Main Flask application
├── train_model.py           # Model training script
├── evaluate_model.py        # Model evaluation tools
├── config.py               # Configuration management
├── requirements.txt        # Python dependencies
├── config.json            # Application settings
├── .env.example           # Environment template
├── templates/
│   └── index.html         # Web interface
├── logs/                  # Log files
├── models/                # Trained models
└── data/                  # Training datasets
    ├── SMSSpamCollection.tsv
    └── SinhalaSpamCollection.tsv
```

### Technology Stack

**Backend:**
- Flask (Web framework)
- scikit-learn (Machine learning)
- pandas (Data processing)
- NumPy (Numerical computing)

**Frontend:**
- Bootstrap 5 (UI framework)
- Custom CSS (Radar animations)
- Vanilla JavaScript (Interactivity)

**Machine Learning:**
- TF-IDF Vectorization
- Multiple algorithms (Naive Bayes, Random Forest, etc.)
- Cross-validation
- Performance evaluation

## 🔌 API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web interface |
| POST | `/predict` | Web form prediction |
| POST | `/api/predict` | Single message API |
| POST | `/api/batch` | Batch prediction API |
| GET | `/api/model/info` | Model information |
| GET | `/api/health` | Health check |
| GET | `/history` | Message history page |
| POST | `/clear_history` | Clear history |

### Request/Response Examples

#### POST /api/predict
**Request:**
```json
{
  "message": "Your message here"
}
```

**Response (Success):**
```json
{
  "prediction": "spam|ham",
  "confidence": 0.85,
  "is_spam": true,
  "processed_message": "processed text",
  "timestamp": "2024-01-15T10:30:00",
  "status": "success"
}
```

**Response (Error):**
```json
{
  "error": "Error message",
  "status": "error"
}
```

### Error Codes

- `400`: Bad Request (invalid input)
- `404`: Endpoint not found
- `405`: Method not allowed
- `500`: Internal server error

## 🔒 Security

### Best Practices
1. **Change Secret Key**: Update `SECRET_KEY` in production
2. **Use HTTPS**: Enable SSL/TLS in production
3. **Rate Limiting**: Configure API rate limits
4. **Input Validation**: All inputs are validated and sanitized
5. **Error Handling**: No sensitive information in error responses

### Production Deployment
```bash
# Set production environment
export ENVIRONMENT=production
export SECRET_KEY=your-super-secret-production-key
export DEBUG=False

# Use production WSGI server
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

## 📈 Performance

### Optimization Tips
- **Model Size**: Current models are optimized for accuracy vs. size
- **Batch Processing**: Use batch API for multiple messages
- **Caching**: Consider adding Redis for prediction caching
- **Load Balancing**: Use multiple workers for high traffic

### Benchmarks
- **Single Prediction**: ~10ms average response time
- **Batch Processing**: ~50ms for 10 messages
- **Model Accuracy**: 95%+ on test datasets
- **Memory Usage**: ~100MB with loaded model

## 🧪 Testing

Run the evaluation script to test model performance:
```bash
python evaluate_model.py
```

For web interface testing:
1. Test with known spam messages
2. Test with normal messages
3. Verify confidence scores
4. Check error handling

## 📝 Logging

Logs are written to `logs/spam_detector.log` with rotation:
- **INFO**: Normal operations
- **WARNING**: Non-critical issues
- **ERROR**: Application errors
- **DEBUG**: Detailed debugging (development only)

Configure logging in `config.json`:
```json
{
  "logging": {
    "level": "INFO",
    "file_path": "logs/spam_detector.log",
    "max_file_size": 10485760,
    "backup_count": 5,
    "console_output": true
  }
}
```

## 🔄 Updates and Maintenance

### Model Retraining
- Retrain periodically with new data
- Monitor performance metrics
- Update preprocessing as needed

### Data Updates
- Add new spam/ham examples to datasets
- Include emerging spam patterns
- Balance dataset classes

### System Updates
```bash
# Update dependencies
pip install -r requirements.txt --upgrade

# Retrain model
python train_model.py

# Evaluate performance
python evaluate_model.py
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt

# Set development environment
export DEBUG=True
export LOG_LEVEL=DEBUG

# Run with auto-reload
python app.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♂️ Support

For support and questions:
- Create an issue on GitHub
- Check the logs for error details
- Review the evaluation report for model insights

## 🎯 Roadmap

### Upcoming Features
- [ ] Real-time model updates
- [ ] Advanced spam detection (images, attachments)
- [ ] Multi-language expansion
- [ ] Machine learning model comparison dashboard
- [ ] User feedback integration
- [ ] Advanced analytics and reporting
- [ ] Docker containerization
- [ ] Kubernetes deployment configs

### Performance Improvements
- [ ] Model quantization for faster inference
- [ ] Async prediction processing
- [ ] Result caching with Redis
- [ ] Load balancing configurations

---

**Made with ❤️ for spam-free communication**