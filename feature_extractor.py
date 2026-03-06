import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import re
from datetime import datetime

class FeatureExtractor:
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=10)  # Minimal features for demonstration
        
    def preprocess_text(self, text):
        """Basic text preprocessing"""
        if not isinstance(text, str):
            return ""
        # Convert to lowercase and remove special characters
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text

    def extract_features(self, df):
        """Extract basic features from text data"""
        features = {}
        
        # Get text content from dataframe
        text_content = []
        for idx in df.index:
            content = ""
            # Try different possible column names
            for col in ['content', 'Content', 'description', 'title', 'text']:
                if col in df.columns and pd.notnull(df.loc[idx, col]):
                    content = str(df.loc[idx, col])
                    break
            text_content.append(self.preprocess_text(content))
        
        if not any(text_content):
            # If no text found, create simple synthetic features
            features['text_length'] = [0] * len(df)
            features['sentiment'] = [0] * len(df)
            return pd.DataFrame(features)
        
        # Basic text features
        features['text_length'] = [len(text) for text in text_content]
        features['word_count'] = [len(text.split()) for text in text_content]
        
        # Sentiment
        try:
            features['sentiment'] = [TextBlob(text).sentiment.polarity for text in text_content]
        except:
            features['sentiment'] = [0] * len(text_content)
        
        # Simple keyword presence
        keywords = ['ai', 'ml', 'data', 'cloud', 'tech']
        for keyword in keywords:
            features[f'has_{keyword}'] = [
                1 if keyword in text else 0 
                for text in text_content
            ]
        
        return pd.DataFrame(features)


        # 4. Temporal features
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            features['hour'] = df['timestamp'].dt.hour
            features['day_of_week'] = df['timestamp'].dt.dayofweek

        # 5. Engagement features
        engagement_mapping = {
            'likes': ['likes', 'Like_Count', 'like_count'],
            'shares': ['shares', 'Share_Count', 'share_count', 'retweet_count'],
            'comments': ['comments', 'Comment_Count', 'comment_count'],
            'views': ['views', 'View_Count', 'view_count'],
            'favorites': ['favorites', 'favorite_count', 'Favorite_Count']
        }
        
        for metric, possible_cols in engagement_mapping.items():
            for col in possible_cols:
                if col in df.columns:
                    features[f'engagement_{metric}'] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    break

        # Convert features to DataFrame
        feature_df = pd.DataFrame(features)
        
        # Handle any missing values
        feature_df = feature_df.fillna(0)
        
        return feature_df

    def get_feature_names(self):
        """
        Get the names of all features that will be extracted
        """
        feature_names = []
        # TF-IDF feature names
        if hasattr(self, 'tfidf') and hasattr(self.tfidf, 'get_feature_names_out'):
            feature_names.extend([f'tfidf_{name}' for name in self.tfidf.get_feature_names_out()])
        
        # Other feature names
        basic_features = [
            'text_length', 'word_count', 'avg_word_length',
            'polarity', 'subjectivity',
            'hour', 'day_of_week'
        ]
        feature_names.extend(basic_features)
        
        # NER feature names
        ner_types = ['ORG', 'PERSON', 'GPE', 'DATE', 'PRODUCT']
        feature_names.extend([f'ner_{ent_type}' for ent_type in ner_types])
        
        # Engagement feature names
        engagement_features = [
            'engagement_likes', 'engagement_shares', 'engagement_comments',
            'engagement_retweets', 'engagement_favorites'
        ]
        feature_names.extend(engagement_features)
        
        return feature_names