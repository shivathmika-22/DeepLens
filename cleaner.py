"""
Text cleaning utilities for DeepLens data processing pipeline.
Handles emoji removal, HTML cleaning, special character removal, and more.
"""

import re
import html
import unicodedata
from typing import Optional, Dict, Any
try:
    import pandas as pd
except Exception:
    # Provide a light-weight stub so `pd.DataFrame` annotations evaluate at import time
    class _PdStub:
        DataFrame = object
    pd = _PdStub()


class TextCleaner:
    """Comprehensive text cleaning utility for social media and news data."""
    
    def __init__(self):
        # Emoji patterns
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"  # dingbats
            "\U000024C2-\U0001F251"  # enclosed characters
            "]+", flags=re.UNICODE
        )
        
        # URL patterns
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        
        # HTML tag patterns
        self.html_pattern = re.compile(r'<[^>]+>')
        
        # Special characters and extra whitespace
        self.special_chars_pattern = re.compile(r'[^\w\s\.\,\!\?\;\:\-\(\)]')
        self.whitespace_pattern = re.compile(r'\s+')
        
        # Common social media patterns
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')
        
    def clean_text(self, text: str, 
                   remove_emojis: bool = True,
                   remove_urls: bool = True,
                   remove_html: bool = True,
                   remove_special_chars: bool = True,
                   remove_mentions: bool = False,
                   remove_hashtags: bool = False,
                   normalize_whitespace: bool = True,
                   decode_html: bool = True) -> str:
        """
        Clean text with various options.
        
        Args:
            text: Input text to clean
            remove_emojis: Remove emoji characters
            remove_urls: Remove URLs
            remove_html: Remove HTML tags
            remove_special_chars: Remove special characters
            remove_mentions: Remove @mentions
            remove_hashtags: Remove #hashtags
            normalize_whitespace: Normalize whitespace
            decode_html: Decode HTML entities
            
        Returns:
            Cleaned text string
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Decode HTML entities first
        if decode_html:
            text = html.unescape(text)
        
        # Remove HTML tags
        if remove_html:
            text = self.html_pattern.sub(' ', text)
        
        # Remove URLs
        if remove_urls:
            text = self.url_pattern.sub(' ', text)
        
        # Remove mentions and hashtags
        if remove_mentions:
            text = self.mention_pattern.sub(' ', text)
        
        if remove_hashtags:
            text = self.hashtag_pattern.sub(' ', text)
        
        # Remove emojis
        if remove_emojis:
            text = self.emoji_pattern.sub(' ', text)
        
        # Remove special characters
        if remove_special_chars:
            text = self.special_chars_pattern.sub(' ', text)
        
        # Normalize whitespace
        if normalize_whitespace:
            text = self.whitespace_pattern.sub(' ', text)
        
        # Strip and return
        return text.strip()
    
    def clean_for_analysis(self, text: str) -> str:
        """Clean text specifically for NLP analysis (keeps basic punctuation)."""
        return self.clean_text(
            text,
            remove_emojis=True,
            remove_urls=True,
            remove_html=True,
            remove_special_chars=False,  # Keep basic punctuation
            remove_mentions=True,
            remove_hashtags=False,  # Keep hashtags for analysis
            normalize_whitespace=True,
            decode_html=True
        )
    
    def clean_for_display(self, text: str) -> str:
        """Clean text for display purposes (removes most noise)."""
        return self.clean_text(
            text,
            remove_emojis=True,
            remove_urls=True,
            remove_html=True,
            remove_special_chars=True,
            remove_mentions=True,
            remove_hashtags=True,
            normalize_whitespace=True,
            decode_html=True
        )
    
    def extract_hashtags(self, text: str) -> list:
        """Extract hashtags from text."""
        if not text:
            return []
        return self.hashtag_pattern.findall(text)
    
    def extract_mentions(self, text: str) -> list:
        """Extract mentions from text."""
        if not text:
            return []
        return self.mention_pattern.findall(text)
    
    def extract_urls(self, text: str) -> list:
        """Extract URLs from text."""
        if not text:
            return []
        return self.url_pattern.findall(text)


def clean_dataframe(df: pd.DataFrame, 
                   text_columns: list,
                   cleaner_config: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Clean text columns in a DataFrame.
    
    Args:
        df: Input DataFrame
        text_columns: List of column names containing text to clean
        cleaner_config: Configuration for text cleaning
        
    Returns:
        DataFrame with cleaned text columns
    """
    if cleaner_config is None:
        cleaner_config = {
            'remove_emojis': True,
            'remove_urls': True,
            'remove_html': True,
            'remove_special_chars': True,
            'normalize_whitespace': True
        }
    
    cleaner = TextCleaner()
    df_cleaned = df.copy()
    
    for col in text_columns:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].astype(str).apply(
                lambda x: cleaner.clean_text(x, **cleaner_config)
            )
    
    return df_cleaned


def standardize_platform_data(df: pd.DataFrame, platform: str) -> pd.DataFrame:
    """
    Standardize data format across different platforms.
    
    Args:
        df: Input DataFrame
        platform: Platform name (twitter, youtube, gnews, facebook, linkedin)
        
    Returns:
        Standardized DataFrame with common columns
    """
    df_std = df.copy()
    df_std['platform'] = platform
    
    # Standardize column names based on platform
    column_mapping = {
        'twitter': {
            'Content': 'content',
            'Posted_At': 'published_date',
            'Username': 'author',
            'URL': 'url',
            'Query': 'query'
        },
        'youtube': {
            'Title': 'title',
            'channel': 'author',
            'Published At': 'published_date',
            'Description': 'content',
            'URL': 'url'
        },
        'gnews': {
            'Title': 'title',
            'Description': 'content',
            'PublishedAt': 'published_date',
            'URL': 'url',
            'Source': 'author',
            'Query': 'query'
        },
        'facebook': {
            'Title': 'title',
            'Summary': 'content',
            'Published': 'published_date',
            'Link': 'url',
            'Page': 'author'
        },
        'linkedin': {
            'content': 'content',
            'scraped_at': 'published_date',
            'query': 'query'
        }
    }
    
    if platform in column_mapping:
        df_std = df_std.rename(columns=column_mapping[platform])
    
    # Ensure required columns exist
    required_columns = ['platform', 'query', 'title', 'content', 'published_date', 'author', 'url']
    
    for col in required_columns:
        if col not in df_std.columns:
            df_std[col] = ''
    
    # Clean text columns
    text_columns = ['title', 'content']
    df_std = clean_dataframe(df_std, text_columns)
    
    return df_std[required_columns]


# Convenience functions
def clean_text(text: str) -> str:
    """Quick text cleaning function."""
    cleaner = TextCleaner()
    return cleaner.clean_for_analysis(text)


def clean_for_display(text: str) -> str:
    """Quick text cleaning for display."""
    cleaner = TextCleaner()
    return cleaner.clean_for_display(text)
