"""
Simple text classifier for sentiment analysis
"""

import logging
from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

class TextClassifier:
    """Simple text classifier optimized for sentiment analysis"""
    
    def __init__(self, model_type: str = "logistic", use_pretrained: bool = False):
        self.model_type = model_type.lower()
        self.use_pretrained = use_pretrained
        # Use more realistic model parameters
        self.model = LogisticRegression(
            random_state=42,
            max_iter=2000,
            C=2.0,  # Slightly reduced regularization for higher accuracy
            class_weight='balanced',  # Handle class imbalance
            solver='lbfgs',  # More sophisticated optimizer
            multi_class='multinomial'  # Better for multi-class problems
        )
        
        # Enhanced text vectorization
        self.vectorizer = TfidfVectorizer(
            max_features=1000,  # Increased feature set for better discrimination
            stop_words='english',
            ngram_range=(1, 3),  # Use unigrams, bigrams, and trigrams
            min_df=2,  # Minimum document frequency
            max_df=0.95,  # Maximum document frequency
            norm='l2',  # Normalize features
            sublinear_tf=True,  # Apply sublinear scaling to term frequencies
            use_idf=True,  # Use inverse document frequency weighting
            smooth_idf=True  # Smooth IDF weights
        )
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
        # Setup logging
        self.logger = logging.getLogger("text_classifier")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def train(self, texts: List[str], labels: List[str]) -> None:
        """Train the classifier on the provided texts and labels"""
        try:
            # Prepare features
            X = self.vectorizer.fit_transform(texts)
            y = self.label_encoder.fit_transform(labels)
            
            # Train the model
            self.model.fit(X, y)
            self.is_trained = True
            self.logger.info(f"Model trained successfully on {len(texts)} samples")
            
        except Exception as e:
            self.logger.error(f"Training error: {str(e)}")
            self.is_trained = False
            raise

    def predict(self, texts: List[str]) -> List[str]:
        """Predict sentiment labels for the given texts"""
        if not self.is_trained:
            self.logger.error("Model not trained. Call train() first.")
            return ['neutral'] * len(texts)
            
        try:
            X = self.vectorizer.transform(texts)
            y_pred = self.model.predict(X)
            return self.label_encoder.inverse_transform(y_pred)
            
        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}")
            return ['neutral'] * len(texts)