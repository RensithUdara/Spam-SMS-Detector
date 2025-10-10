"""
Configuration Management for Spam Detector
Handles application settings, environment variables, and logging configuration
"""

import os
import logging
import logging.handlers
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import json

@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    url: str = "sqlite:///spam_detector.db"
    pool_size: int = 5
    pool_timeout: int = 30

@dataclass
class ModelConfig:
    """Model configuration settings"""
    model_path: str = "model.pkl"
    vectorizer_path: str = "vectorizer.pkl"
    metadata_path: str = "model_metadata.pkl"
    max_message_length: int = 1000
    confidence_threshold: float = 0.5

@dataclass
class APIConfig:
    """API configuration settings"""
    max_batch_size: int = 100
    rate_limit_per_minute: int = 60
    enable_cors: bool = True
    cors_origins: str = "*"

@dataclass
class LoggingConfig:
    """Logging configuration settings"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "spam_detector.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    console_output: bool = True

@dataclass
class SecurityConfig:
    """Security configuration settings"""
    secret_key: str = "change-this-in-production"
    session_timeout: int = 3600  # 1 hour
    max_login_attempts: int = 5
    login_timeout: int = 300  # 5 minutes

class Config:
    """Main configuration class"""
    
    def __init__(self):
        self.app_name = "Spam Detector"
        self.version = "2.0.0"
        self.debug = self._get_bool_env('DEBUG', False)
        self.host = os.getenv('HOST', '0.0.0.0')
        self.port = int(os.getenv('PORT', 5000))
        self.environment = os.getenv('ENVIRONMENT', 'development')
        
        # Load configurations
        self.database = DatabaseConfig()
        self.model = ModelConfig()
        self.api = APIConfig()
        self.logging = LoggingConfig()
        self.security = SecurityConfig()
        
        # Load from environment variables
        self._load_from_env()
        
        # Load from config file if exists
        self._load_from_file()
        
    def _get_bool_env(self, key: str, default: bool = False) -> bool:
        """Get boolean value from environment variable"""
        value = os.getenv(key, '').lower()
        return value in ('true', '1', 'yes', 'on') if value else default
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        # Database config
        if db_url := os.getenv('DATABASE_URL'):
            self.database.url = db_url
            
        # Model config
        if model_path := os.getenv('MODEL_PATH'):
            self.model.model_path = model_path
        if vectorizer_path := os.getenv('VECTORIZER_PATH'):
            self.model.vectorizer_path = vectorizer_path
        if max_length := os.getenv('MAX_MESSAGE_LENGTH'):
            self.model.max_message_length = int(max_length)
            
        # API config
        if max_batch := os.getenv('MAX_BATCH_SIZE'):
            self.api.max_batch_size = int(max_batch)
        if rate_limit := os.getenv('RATE_LIMIT_PER_MINUTE'):
            self.api.rate_limit_per_minute = int(rate_limit)
        if cors_origins := os.getenv('CORS_ORIGINS'):
            self.api.cors_origins = cors_origins
            
        # Logging config
        if log_level := os.getenv('LOG_LEVEL'):
            self.logging.level = log_level.upper()
        if log_file := os.getenv('LOG_FILE'):
            self.logging.file_path = log_file
            
        # Security config
        if secret_key := os.getenv('SECRET_KEY'):
            self.security.secret_key = secret_key
    
    def _load_from_file(self, config_file: str = 'config.json'):
        """Load configuration from JSON file"""
        if not os.path.exists(config_file):
            return
            
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update configurations from file
            if 'database' in config_data:
                for key, value in config_data['database'].items():
                    if hasattr(self.database, key):
                        setattr(self.database, key, value)
                        
            if 'model' in config_data:
                for key, value in config_data['model'].items():
                    if hasattr(self.model, key):
                        setattr(self.model, key, value)
                        
            if 'api' in config_data:
                for key, value in config_data['api'].items():
                    if hasattr(self.api, key):
                        setattr(self.api, key, value)
                        
            if 'logging' in config_data:
                for key, value in config_data['logging'].items():
                    if hasattr(self.logging, key):
                        setattr(self.logging, key, value)
                        
            if 'security' in config_data:
                for key, value in config_data['security'].items():
                    if hasattr(self.security, key):
                        setattr(self.security, key, value)
                        
        except Exception as e:
            print(f"Warning: Could not load configuration from {config_file}: {e}")
    
    def setup_logging(self):
        """Setup logging configuration"""
        # Create logs directory if it doesn't exist
        log_dir = Path(self.logging.file_path).parent
        log_dir.mkdir(exist_ok=True)
        
        # Configure root logger
        logging.basicConfig(level=logging.WARNING)
        
        # Create application logger
        logger = logging.getLogger('spam_detector')
        logger.setLevel(getattr(logging, self.logging.level))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # File handler with rotation
        if self.logging.file_path:
            file_handler = logging.handlers.RotatingFileHandler(
                self.logging.file_path,
                maxBytes=self.logging.max_file_size,
                backupCount=self.logging.backup_count
            )
            file_handler.setLevel(getattr(logging, self.logging.level))
            file_formatter = logging.Formatter(self.logging.format)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        # Console handler
        if self.logging.console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, self.logging.level))
            console_formatter = logging.Formatter(self.logging.format)
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def validate(self):
        """Validate configuration settings"""
        errors = []
        
        # Validate model paths
        if not os.path.exists(self.model.model_path):
            errors.append(f"Model file not found: {self.model.model_path}")
        
        # Validate security
        if self.environment == 'production' and self.security.secret_key == "change-this-in-production":
            errors.append("Secret key must be changed in production")
        
        # Validate logging
        if self.logging.level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            errors.append(f"Invalid log level: {self.logging.level}")
        
        # Validate API settings
        if self.api.max_batch_size <= 0:
            errors.append("Max batch size must be greater than 0")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary"""
        return {
            'app_name': self.app_name,
            'version': self.version,
            'debug': self.debug,
            'host': self.host,
            'port': self.port,
            'environment': self.environment,
            'database': {
                'url': self.database.url,
                'pool_size': self.database.pool_size,
                'pool_timeout': self.database.pool_timeout
            },
            'model': {
                'model_path': self.model.model_path,
                'vectorizer_path': self.model.vectorizer_path,
                'metadata_path': self.model.metadata_path,
                'max_message_length': self.model.max_message_length,
                'confidence_threshold': self.model.confidence_threshold
            },
            'api': {
                'max_batch_size': self.api.max_batch_size,
                'rate_limit_per_minute': self.api.rate_limit_per_minute,
                'enable_cors': self.api.enable_cors,
                'cors_origins': self.api.cors_origins
            },
            'logging': {
                'level': self.logging.level,
                'format': self.logging.format,
                'file_path': self.logging.file_path,
                'max_file_size': self.logging.max_file_size,
                'backup_count': self.logging.backup_count,
                'console_output': self.logging.console_output
            },
            'security': {
                'session_timeout': self.security.session_timeout,
                'max_login_attempts': self.security.max_login_attempts,
                'login_timeout': self.security.login_timeout
            }
        }
    
    def save_to_file(self, config_file: str = 'config.json'):
        """Save current configuration to file"""
        config_dict = self.to_dict()
        # Remove sensitive information
        if 'security' in config_dict:
            config_dict['security'].pop('secret_key', None)
        
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)

# Global configuration instance
config = Config()