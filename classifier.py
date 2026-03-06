"""
Text classification models for DeepLens.
Supports topic classification, sentiment analysis, and custom classification tasks.
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
from sklearn.preprocessing import LabelEncoder

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TextClassifier:
    """Main class for text classification."""
    
    def __init__(self, model_type: str = "logistic", use_pretrained: bool = True):
        self.model_type = model_type.lower()
        self.use_pretrained = use_pretrained
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.logger = self._setup_logger()
        self.pretrained_model = None
        self._load_model()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for the classifier."""
        logger = logging.getLogger("text_classifier")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_model(self):
        """Load the specified classification model."""
        if self.use_pretrained:
            self._load_pretrained_model()
        else:
            self._initialize_sklearn_model()
    
    def _load_pretrained_model(self):
        """Load pretrained transformer model."""
        try:
            from transformers import pipeline
            
            if self.model_type in ["sentiment", "emotion"]:
                # Default sentiment model if not using specialized subclass
                self.pretrained_model = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
                )
            elif self.model_type == "topic":
                # Default zero-shot model if not using specialized subclass
                self.pretrained_model = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli"
                )
            else:
                self.pretrained_model = pipeline(
                    "text-classification",
                    model="distilbert-base-uncased-finetuned-sst-2-english"
                )
            
            self.logger.info("✓ Pretrained model loaded successfully")
            
        except ImportError:
            self.logger.warning("transformers library not installed. Using sklearn models.")
            self.use_pretrained = False
            self._initialize_sklearn_model()
        except Exception as e:
            self.logger.error(f"Error loading pretrained model: {str(e)}")
            self.logger.info("Falling back to sklearn models.")
            self.use_pretrained = False
            self._initialize_sklearn_model()
    
    def _initialize_sklearn_model(self):
        """Initialize sklearn-based model."""
        if self.model_type == "naive_bayes":
            self.model = MultinomialNB()
        elif self.model_type == "logistic":
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier(random_state=42, n_estimators=100)
        else:
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        self.logger.info(f"✓ {self.model_type} sklearn model initialized")
    
    def train(self, texts: List[str], labels: List[str]) -> None:
        """
        Train the classification model on the entire dataset.
        
        Args:
            texts: List of training texts
            labels: List of corresponding labels
        """
        if self.use_pretrained:
            self.logger.warning("Pretrained models are used for inference and don't require training.")
            return
        
        self.logger.info(f"Training {self.model_type} model on {len(texts)} samples...")
        
        # Prepare data
        X = self.vectorizer.fit_transform(texts)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(labels)
        
        # Train model on all data
        self.model.fit(X, y)
        
        self.logger.info(f"Training completed. Accuracy: {accuracy:.3f}")
        
        return {
            "accuracy": accuracy,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "classes": len(self.label_encoder.classes_)
        }
    
    def predict(self, texts: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Predict labels for texts.
        
        Args:
            texts: Text or list of texts to classify
            
        Returns:
            Predicted label(s)
        """
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        else:
            single_text = False
        
        if self.use_pretrained:
            predictions = self._predict_pretrained(texts)
        else:
            predictions = self._predict_sklearn(texts)
        
        return predictions[0] if single_text else predictions
    
    def _predict_pretrained(self, texts: List[str]) -> List[str]:
        """Predict using pretrained model."""
        predictions = []
        
        for text in texts:
            try:
                # Specialized classifiers handle their own logic
                if isinstance(self, SentimentClassifier):
                    result = self.predict_sentiment(text)
                    predictions.append(result['label'])
                elif isinstance(self, TopicClassifier):
                    result = self.pretrained_model(text, self.topics)
                    predictions.append(result['labels'][0])
                else:
                    # Generic text classification pipeline
                    result = self.pretrained_model(text)
                    predictions.append(result[0]['label'])
            except Exception as e:
                self.logger.error(f"Error in pretrained prediction: {str(e)}")
                predictions.append("unknown")
        
        return predictions
    
    def _predict_sklearn(self, texts: List[str]) -> List[str]:
        """Predict using sklearn model."""
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained. Call train() first or load a trained model.")
        
        # Check if vectorizer has been fitted (only necessary if called externally without train)
        if not hasattr(self.vectorizer, 'vocabulary_'):
             raise ValueError("Vectorizer has not been fitted. Call train() first.")
             
        X = self.vectorizer.transform(texts)
        y_pred = self.model.predict(X)
        
        # Decode labels
        predictions = self.label_encoder.inverse_transform(y_pred)
        return predictions.tolist()
    
    def predict_proba(self, texts: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Predict class probabilities/confidence scores.
        
        Args:
            texts: Text or list of texts
            
        Returns:
            Probability arrays (sklearn) or confidence scores (pretrained)
        """
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        else:
            single_text = False
        
        if self.use_pretrained:
            # --- START REFINED LOGIC for Pretrained Models ---
            probas = []

            if isinstance(self, SentimentClassifier):
                # SentimentClassifier provides a score directly
                results = self.predict_sentiment(texts)
                for res in results:
                    score = res['score']
                    label = res['label']
                    
                    # Standardize: map to a simple NEG/NEUTRAL/POS probability array
                    # This relies on knowing the model's output labels (e.g., NEGATIVE, NEUTRAL, POSITIVE)
                    if label == 'NEGATIVE':
                        probas.append([score, 1 - score / 2, 1 - score / 2])
                    elif label == 'POSITIVE':
                        probas.append([1 - score / 2, 1 - score / 2, score])
                    else: # NEUTRAL or other
                        probas.append([1 - score / 2, score, 1 - score / 2])
                
                # Simple normalization (sum to 1) for the custom probas
                proba_array = np.array(probas)
                proba_array = proba_array / proba_array.sum(axis=1, keepdims=True)

            elif isinstance(self, TopicClassifier):
                # Zero-shot classification results already contain scores for all candidate labels
                for text in texts:
                    if not hasattr(self, 'topics'):
                        self.logger.error("Topic list missing for TopicClassifier.")
                        proba_array = np.array([[0.5, 0.5]] * len(texts))
                        break # Exit loop and use dummy array

                    result = self.pretrained_model(text, self.topics)
                    
                    # Create a dictionary mapping labels to scores
                    topic_probas_dict = {label: score for label, score in zip(result['labels'], result['scores'])}
                    
                    # Create a full probability array in the consistent order of self.topics
                    ordered_probas = [topic_probas_dict.get(topic, 0.0) for topic in self.topics]
                    probas.append(ordered_probas)
                
                proba_array = np.array(probas)
                # Zero-shot scores are already normalized, so no need to divide by sum

            else:
                # Generic text classification pipeline - try to get scores directly if possible
                probas = []
                for text in texts:
                    result = self.pretrained_model(text)
                    probas.append(result[0]['score'])
                
                # Return confidence score as a single value or as a simple two-class array
                proba_array = np.array([[1 - score, score] for score in probas])
            
            # --- END REFINED LOGIC ---
            
        else:
            # SKLearn models
            if self.model is None or self.vectorizer is None:
                raise ValueError("Model not trained. Call train() first or load a trained model.")
            
            X = self.vectorizer.transform(texts)
            proba_array = self.model.predict_proba(X)
        
        return proba_array[0] if single_text else proba_array
    
    def classify_dataframe(self, df: pd.DataFrame, text_column: str = 'content',
                             label_column: str = 'predicted_label',
                             proba_column: str = 'prediction_confidence') -> pd.DataFrame:
        """
        Classify texts in a DataFrame.
        
        Args:
            df: Input DataFrame
            text_column: Column containing texts
            label_column: Column name for predictions
            proba_column: Column name for confidence scores
            
        Returns:
            DataFrame with predictions
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        self.logger.info(f"Classifying {len(df)} texts...")
        
        df_classified = df.copy()
        texts = df[text_column].astype(str).tolist()
        
        # Get predictions
        predictions = self.predict(texts)
        df_classified[label_column] = predictions
        
        # Get confidence scores
        try:
            probabilities = self.predict_proba(texts)
            
            # For multi-class (2D array), confidence is the max probability
            if probabilities.ndim == 2:
                confidences = np.max(probabilities, axis=1)
            # For single-score output (1D array), use it directly
            elif probabilities.ndim == 1 and not isinstance(probabilities, np.ndarray):
                confidences = np.array([probabilities]) # Handle single-text case if needed
            else:
                confidences = probabilities # Should be 1D if single-score
                
            df_classified[proba_column] = confidences
            
        except Exception as e:
            self.logger.warning(f"Could not calculate robust probabilities: {str(e)}. Setting confidence to 1.0.")
            df_classified[proba_column] = 1.0
        
        self.logger.info("Classification completed")
        return df_classified
    
    def save_model(self, filepath: str):
        """Save the trained model (for sklearn only)."""
        if self.use_pretrained:
            self.logger.warning("Pretrained models are loaded remotely and cannot be saved via joblib.")
            return
        
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model (for sklearn only)."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        self.label_encoder = model_data['label_encoder']
        self.model_type = model_data['model_type']
        self.use_pretrained = False
        
        self.logger.info(f"Model loaded from {filepath}")


class TopicClassifier(TextClassifier):
    """Specialized classifier for zero-shot topic classification."""
    
    def __init__(self, topics: List[str] = None):
        self.topics = topics or [
            "technology", "business", "sports", "entertainment", 
            "politics", "science", "health", "education"
        ]
        super().__init__(model_type="topic", use_pretrained=True)
    
    def _load_pretrained_model(self):
        """Load topic classification model."""
        try:
            from transformers import pipeline
            self.pretrained_model = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )
            self.logger.info("✓ Topic classification model loaded")
        except Exception as e:
            self.logger.error(f"Error loading topic model: {str(e)}")
            # Fallback to base class's error handling/sklearn initialization
            super()._load_pretrained_model()


class SentimentClassifier(TextClassifier):
    """Specialized classifier for sentiment analysis."""
    
    def __init__(self):
        # Note: model_type is set to 'sentiment' for clarity and specific pipeline loading
        super().__init__(model_type="sentiment", use_pretrained=True)
    
    def _load_pretrained_model(self):
        """Load sentiment analysis model."""
        try:
            from transformers import pipeline
            self.pretrained_model = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
            self.logger.info("✓ Sentiment analysis model loaded")
        except Exception as e:
            self.logger.error(f"Error loading sentiment model: {str(e)}")
            # Fallback to base class's error handling/sklearn initialization
            super()._load_pretrained_model()
    
    def predict_sentiment(self, texts: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        """
        Predict sentiment with detailed results.
        
        Args:
            texts: Text or list of texts
            
        Returns:
            Sentiment results with label and score
        """
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        else:
            single_text = False
        
        results = []
        
        for text in texts:
            try:
                # The pipeline call returns a list of dictionaries for each text
                result = self.pretrained_model(text)[0]
                results.append({
                    'label': result['label'],
                    'score': result['score']
                })
            except Exception as e:
                self.logger.error(f"Error in sentiment prediction: {str(e)}")
                results.append({'label': 'NEUTRAL', 'score': 0.5})
        
        return results[0] if single_text else results


def main():
    """Main function for testing the classifier."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Text classification")
    parser.add_argument("--text", help="Text to classify")
    parser.add_argument("--input", help="Input CSV file")
    parser.add_argument("--output", help="Output CSV file")
    parser.add_argument("--type", default="sentiment", choices=["sentiment", "topic", "custom"],
                         help="Classification type")
    parser.add_argument("--text_column", default="content", help="Text column name")
    parser.add_argument("--train", help="Training CSV file")
    parser.add_argument("--label_column", default="label", help="Label column name")
    
    args = parser.parse_args()
    
    if args.type == "sentiment":
        classifier = SentimentClassifier()
    elif args.type == "topic":
        classifier = TopicClassifier()
    else:
        classifier = TextClassifier()
    
    if args.train:
        # Train model
        try:
            df_train = pd.read_csv(args.train)
            texts = df_train[args.text_column].astype(str).tolist()
            labels = df_train[args.label_column].astype(str).tolist()
            
            metrics = classifier.train(texts, labels)
            print(f"Training completed. Metrics: {metrics}")
            
            # Save the trained model for custom classifiers
            if not classifier.use_pretrained:
                save_path = "custom_classifier.joblib"
                classifier.save_model(save_path)
                print(f"Custom model saved to {save_path}")
            
        except Exception as e:
            print(f"Training error: {str(e)}")
            sys.exit(1)
    
    if args.text:
        # Classify single text
        if args.type == "sentiment":
            result = classifier.predict_sentiment(args.text)
            print(f"Text: {args.text}")
            print(f"Sentiment: {result}")
        else:
            prediction = classifier.predict(args.text)
            print(f"Text: {args.text}")
            print(f"Prediction: {prediction}")
    
    elif args.input:
        # Classify DataFrame
        try:
            df = pd.read_csv(args.input)
            df_classified = classifier.classify_dataframe(df, args.text_column)
            
            output_path = args.output or args.input.replace('.csv', '_classified.csv')
            df_classified.to_csv(output_path, index=False, encoding='utf-8')
            
            print(f"Classification completed. Results saved to {output_path}")
            
        except Exception as e:
            print(f"Error: {str(e)}")
            sys.exit(1)
    
    elif not args.train:
        print("Please provide --text, --input, or --train argument.")


if __name__ == "__main__":
    main()