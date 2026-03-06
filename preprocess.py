
"""
Data preprocessing pipeline for DeepLens.
Handles text cleaning, NER, region detection, validation, and feature extraction.
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple 
import logging
import re

# Add new imports
from pydantic import BaseModel # For data schema validation
import langdetect             # For language detection
from datetime import datetime, timedelta

# Add parent directory to path for imports (if needed)
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Pydantic Data Schema (New) ---
class DataValidator(BaseModel):
    """Defines the required schema and types for a valid data record."""
    title: str
    content: str
    published_date: datetime = None
    url: str = None

# --- Utility Class Definitions (Required by DataPreprocessor) ---

def clean_dataframe(df: pd.DataFrame, text_columns: List[str]) -> pd.DataFrame:
    """Placeholder for a utility function that was likely imported but unused."""
    return df

class TextCleaner:
    """Utility class for text cleaning operations."""
    
    def clean_for_analysis(self, text: str) -> str:
        """Lowercases text, removes newlines, and strips extra spaces, preserving punctuation."""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text) 
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def extract_hashtags(self, text: str) -> List[str]:
        """Extracts simple hashtags."""
        return re.findall(r'#(\w+)', text)

    def extract_mentions(self, text: str) -> List[str]:
        """Extracts simple mentions."""
        return re.findall(r'@(\w+)', text)

class FeatureExtractor:
    """Utility class for extracting engineered features."""
    
    def get_sentiment_score(self, text: str) -> float:
        """Simulated function to return a sentiment score (e.g., from VADER or similar)."""
        text_lower = str(text).lower()
        if 'great' in text_lower or 'success' in text_lower:
            return 0.8
        if 'fail' in text_lower or 'loss' in text_lower:
            return -0.6
        return 0.1

    def extract_named_entities(self, text: str, spacy_model: Any) -> List[Tuple[str, str]]:
        """Extracts named entities using the spaCy model."""
        if not spacy_model or pd.isna(text):
            return []
        doc = spacy_model(str(text))
        return [(ent.text, ent.label_) for ent in doc.ents]

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts basic text features and returns a new DataFrame (minimal implementation)."""
        # This method is replaced by the new logic inside DataPreprocessor.extract_features
        # but is kept minimal here to avoid breaking the DataPreprocessor __init__
        df_features = pd.DataFrame(index=df.index)
        return df_features


# --- DataPreprocessor Class (Updated) ---

class DataPreprocessor:
    """Main class for data preprocessing."""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.cleaner = TextCleaner()
        self.feature_extractor = FeatureExtractor()
        self.spacy_model = None
        self._load_spacy_model()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for the preprocessor."""
        logger = logging.getLogger("data_preprocessor")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_spacy_model(self):
        """Load spaCy model for NER."""
        try:
            import spacy
            # Ensure necessary data is available for spaCy, langdetect
            self.spacy_model = spacy.load("en_core_web_sm")
            self.logger.info("✓ spaCy model loaded successfully")
        except OSError:
            self.logger.warning("⚠️ spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.spacy_model = None
        except ImportError:
            self.logger.warning("⚠️ spaCy not installed. Install with: pip install spacy")
            self.spacy_model = None
    
    # --- NEW/UPDATED METHODS START HERE ---

    def validate_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate data, remove duplicates, filter by language, and enforce schema."""
        self.logger.info(f"Starting validation and structural cleaning on {len(df)} records...")
        initial_count = len(df)
        
        # 1. Remove duplicates (using the requested subset)
        df = df.drop_duplicates(subset=['title', 'content'], keep='first').reset_index(drop=True)
        self.logger.info(f"Removed duplicates: {initial_count - len(df)} records lost.")

        # 2. Remove rows with missing crucial data
        df = df.dropna(subset=['title', 'content', 'published_date'])
        self.logger.info(f"Removed NaT/NaN: {initial_count - len(df)} records lost (cumulative).")
        
        # 3. Detect and filter language
        try:
            # Note: langdetect can raise a LangDetectException on short/empty strings
            df['language'] = df['content'].apply(
                lambda x: langdetect.detect(str(x)) if pd.notna(x) and len(str(x).split()) > 5 else 'unknown'
            )
            df = df[df['language'] == 'en']
            self.logger.info(f"Filtered to English ('en'): {len(df)} records remaining.")
        except Exception as e:
            self.logger.error(f"Error during language detection: {e}. Skipping language filter.")

        # 4. Convert published_date to datetime for Pydantic validation
        if 'published_date' in df.columns:
            # Coerce to datetime, errors='coerce' turns invalid parsing into NaT
            df['published_date'] = pd.to_datetime(df['published_date'], errors='coerce')
        
        # 5. Validate data structure using Pydantic
        valid_records = []
        invalid_count = 0
        for _, row in df.iterrows():
            try:
                # Prepare data for Pydantic (handles datetime objects needed by DataValidator)
                record_data = row.to_dict()
                
                # Check for NaT (Not a Time) which Pydantic/DataValidator won't like
                if 'published_date' in record_data and pd.isna(record_data['published_date']):
                    invalid_count += 1
                    continue
                
                DataValidator(**record_data) # Validate schema
                valid_records.append(row)
            except Exception as e:
                # self.logger.warning(f"Invalid record dropped (Pydantic failure): {str(e)}")
                invalid_count += 1
        
        df_validated = pd.DataFrame(valid_records).reset_index(drop=True)
        self.logger.info(f"Dropped {invalid_count} records due to Pydantic validation failure.")
        self.logger.info(f"Validation completed. {len(df_validated)} records remaining.")
        
        return df_validated

    def filter_by_date(self, df: pd.DataFrame, days_back: int = 30) -> pd.DataFrame:
        """Filter data by date range."""
        if 'published_date' not in df.columns:
            self.logger.warning("No published_date column found. Skipping date filtering.")
            return df
        
        df['published_date'] = pd.to_datetime(df['published_date'], errors='coerce')
        
        # Drop rows where date conversion failed (NaT)
        df = df.dropna(subset=['published_date'])

        # Filter by date range
        cutoff_date = datetime.now() - timedelta(days=days_back)
        df_filtered = df[df['published_date'] >= cutoff_date]
        
        self.logger.info(f"Filtered to {len(df_filtered)} records from last {days_back} days")
        return df_filtered
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from text data, incorporating new requested features."""
        self.logger.info("Extracting features...")

        if 'content' not in df.columns:
            self.logger.error("Cannot extract features: 'content' column is missing.")
            return df

        try:
            # 1. Basic Text Features
            df['text_length'] = df['content'].astype(str).str.len()
            df['word_count'] = df['content'].astype(str).str.split().str.len()
            df['average_word_length'] = df['content'].astype(str).apply(
                 lambda x: np.mean([len(word) for word in str(x).split()]) if str(x) else 0.0
            )
            
            # 2. Sentiment Scores (using FeatureExtractor utility)
            df['sentiment_scores'] = df['content'].astype(str).apply(
                self.feature_extractor.get_sentiment_score
            )
            
            # 3. Named Entities (using spaCy utility)
            if self.spacy_model:
                df['named_entities'] = df['content'].astype(str).apply(
                    lambda x: self.feature_extractor.extract_named_entities(x, self.spacy_model)
                )
                df['num_named_entities'] = df['named_entities'].apply(len)
                self.logger.info("✓ Extracted named entities.")
            else:
                df['named_entities'] = ''
                df['num_named_entities'] = 0
                self.logger.warning("Named entity extraction skipped (spaCy model missing).")
            
            self.logger.info(f"✓ Feature extraction complete. {df.shape[1] - len(df.columns) + 5} new columns added.")
            return df
            
        except Exception as e:
            self.logger.error(f"Feature extraction error: {str(e)}")
            return df

    # --- OLD METHODS (Adjusted/Retained) ---

    def clean_text_data(self, df: pd.DataFrame, text_columns: List[str] = None) -> pd.DataFrame:
        """Clean text data in the DataFrame (retained, but structural cleaning moved to validate_and_clean_data)."""
        if text_columns is None:
            text_columns = ['title', 'content', 'description']
        
        self.logger.info(f"Performing deep cleaning on text in columns: {text_columns}")
        
        df_cleaned = df.copy()
        
        for col in text_columns:
            if col in df_cleaned.columns:
                # Clean text for analysis (keeps basic punctuation)
                df_cleaned[col] = df_cleaned[col].astype(str).apply(
                    lambda x: self.cleaner.clean_for_analysis(x) if pd.notna(x) else ""
                )
                
                # Note: Removal of empty/short texts is now handled in validate_and_clean_data
        
        self.logger.info(f"Deep text cleaning completed.")
        return df_cleaned
    
    def detect_regions(self, df: pd.DataFrame, text_columns: List[str] = None) -> pd.DataFrame:
        """Detect regions/countries mentioned in text using NER (retained)."""
        if text_columns is None:
            text_columns = ['title', 'content']
        
        if self.spacy_model is None:
            self.logger.warning("spaCy model not available. Skipping region detection.")
            df['detected_regions'] = ''
            df['primary_region'] = ''
            return df
        
        self.logger.info("Detecting regions using NER...")
        
        df_regions = df.copy()
        df_regions['detected_regions'] = ''
        df_regions['primary_region'] = ''
        
        # Common country/region mappings
        region_mappings = {
            'india': 'India', 'united states': 'USA', 'usa': 'USA', 'us': 'USA', 'america': 'USA',
            'united kingdom': 'UK', 'uk': 'UK', 'britain': 'UK', 'china': 'China', 'japan': 'Japan', 
            'germany': 'Germany', 'france': 'France', 'canada': 'Canada', 'australia': 'Australia', 
            'brazil': 'Brazil', 'russia': 'Russia', 'south korea': 'South Korea', 
            'singapore': 'Singapore', 'uae': 'UAE', 'united arab emirates': 'UAE'
        }
        region_keys = sorted(list(region_mappings.keys()), key=len, reverse=True) 

        for idx, row in df_regions.iterrows():
            regions = set()
            text_to_analyze = ""
            
            for col in text_columns:
                if col in row and pd.notna(row[col]):
                    text_to_analyze += " " + str(row[col])
            
            if text_to_analyze.strip():
                doc = self.spacy_model(text_to_analyze)
                for ent in doc.ents:
                    if ent.label_ == "GPE": 
                        region_name = ent.text.lower()
                        if region_name in region_mappings:
                            regions.add(region_mappings[region_name])
                
                text_lower = self.cleaner.clean_for_analysis(text_to_analyze)
                for key in region_keys:
                    if re.search(r'\b' + re.escape(key) + r'\b', text_lower):
                        regions.add(region_mappings[key])
            
            df_regions.at[idx, 'detected_regions'] = ', '.join(sorted(regions))
            df_regions.at[idx, 'primary_region'] = list(regions)[0] if regions else ''
        
        self.logger.info("Region detection completed")
        return df_regions
    
    # NOTE: remove_duplicates method is replaced by the duplicate removal logic inside validate_and_clean_data
    # The original filter_by_date_range is replaced by filter_by_date
    
    def preprocess_pipeline(self, df: pd.DataFrame, 
                          text_columns: List[str] = None,
                          remove_duplicates: bool = True, # Retained for control, but validation does it first
                          detect_regions: bool = True,
                          extract_features: bool = True,
                          filter_by_date: bool = False,
                          days_back: int = 30) -> pd.DataFrame:
        """
        Run the complete preprocessing pipeline.
        
        (The order is adjusted to perform validation/language filtering early.)
        """
        self.logger.info("Starting updated preprocessing pipeline...")
        
        if df.empty:
            self.logger.warning("Empty DataFrame provided")
            return df
        
        # Step 1: Validation, Dedup, Language Filter, and Schema Enforcement (NEW CORE STEP)
        df_processed = self.validate_and_clean_data(df)
        
        # Step 2: Deep Text Cleaning (tokenization prep, etc.)
        df_processed = self.clean_text_data(df_processed, text_columns)
        
        # Step 3: Filter by date (Uses new filter_by_date method)
        if filter_by_date:
            df_processed = self.filter_by_date(df_processed, days_back)
        
        # Step 4: Detect regions
        if detect_regions:
            df_processed = self.detect_regions(df_processed, text_columns)
        
        # Step 5: Extract features (Uses new, updated method)
        if extract_features:
            df_processed = self.extract_features(df_processed)
        
        self.logger.info(f"Preprocessing completed. {len(df_processed)} final records processed")
        
        self._print_preprocessing_summary(df_processed)
        
        return df_processed
    
    def _print_preprocessing_summary(self, df: pd.DataFrame) -> None:
        """Print preprocessing summary."""
        print("\n" + "="*50)
        print("🔧 PREPROCESSING SUMMARY")
        print("="*50)
        print(f"Total records: {len(df)}")
        
        if 'platform' in df.columns:
            print(f"Platforms: {df['platform'].value_counts(dropna=False).head().to_dict()}")
        
        if 'primary_region' in df.columns and not df['primary_region'].empty:
            regions = df['primary_region'].value_counts().head(5)
            print(f"Top regions: {regions.to_dict()}")
        
        if 'word_count' in df.columns:
            print(f"Avg word count: {df['word_count'].mean():.1f}")
        
        if 'sentiment_scores' in df.columns:
            print(f"Avg sentiment: {df['sentiment_scores'].mean():.2f}")
        
        print("="*50)


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess scraped data")
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--output", help="Output CSV file path")
    parser.add_argument("--text_columns", nargs="+", default=['title', 'content'],
                       help="Text columns to clean")
    parser.add_argument("--no_regions", action="store_true", help="Skip region detection")
    parser.add_argument("--no_features", action="store_true", help="Skip feature extraction")
    parser.add_argument("--filter_days", type=int, help="Filter to last N days")
    
    args = parser.parse_args()
    
    # Load data
    try:
        df = pd.read_csv(args.input)
        print(f"Loaded {len(df)} records from {args.input}")
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Preprocess
    preprocessor = DataPreprocessor()
    
    try:
        df_processed = preprocessor.preprocess_pipeline(
            df=df,
            text_columns=args.text_columns,
            detect_regions=not args.no_regions,
            extract_features=not args.no_features,
            filter_by_date=args.filter_days is not None,
            days_back=args.filter_days or 30
        )
        
        # Save processed data
        output_path = args.output or args.input.replace('.csv', '_processed.csv')
        df_processed.to_csv(output_path, index=False, encoding='utf-8')
        
        print(f"\n✅ Preprocessing completed")
        print(f"📁 Processed data saved to: {output_path}")
        
    except Exception as e:
        print(f"\n❌ A critical error occurred during the pipeline: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    # This part requires a test CSV file and installation of: 
    # pip install pandas numpy spacy langdetect pydantic
    # python -m spacy download en_core_web_sm
    
    # To run: python your_script_name.py --input data.csv
    # main()
    pass