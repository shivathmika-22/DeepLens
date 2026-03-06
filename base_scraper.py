"""
Base scraper class for consistent data extraction across platforms.
"""

import os
import sys
import argparse
try:
    import pandas as pd
except Exception:
    # Provide a light-weight stub so `pd.DataFrame` annotations evaluate at import time
    class _PdStub:
        DataFrame = object
    pd = _PdStub()
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

# Removed manual sys.path manipulation to rely on the fix in app/app.py

from utils.cleaner import TextCleaner, standardize_platform_data


class BaseScraper(ABC):
    """Base class for all platform scrapers."""
    
    def __init__(self, platform_name: str):
        self.platform_name = platform_name
        self.cleaner = TextCleaner()
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for the scraper."""
        logger = logging.getLogger(f"{self.platform_name}_scraper")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Create a UTF-8 compatible StreamHandler
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(logging.INFO)
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    @abstractmethod
    def fetch_data(self, query: str, max_results: int = 50, **kwargs) -> List[Dict[str, Any]]:
        """
        Fetch data from the platform.
        
        Args:
            query: Search query
            max_results: Maximum number of results to fetch
            **kwargs: Additional platform-specific parameters
            
        Returns:
            List of dictionaries containing raw data
        """
        pass
    
    def clean_and_standardize(self, raw_data: List[Dict[str, Any]], query: str) -> pd.DataFrame:
        """
        Clean and standardize the raw data.
        
        Args:
            raw_data: List of raw data dictionaries
            query: Original search query
            
        Returns:
            Standardized and cleaned DataFrame
        """
        if not raw_data:
            self.logger.warning("No data to clean and standardize")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(raw_data)
        
        # Add query to all records
        df['query'] = query
        
        # Standardize platform data
        df_std = standardize_platform_data(df, self.platform_name)
        
        self.logger.info(f"Cleaned and standardized {len(df_std)} records")
        return df_std
    
    def save_raw_data(self, raw_data: List[Dict[str, Any]], output_path: str) -> None:
        """Save raw data to CSV file."""
        if not raw_data:
            self.logger.warning("No raw data to save")
            return
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df_raw = pd.DataFrame(raw_data)
        df_raw.to_csv(output_path, index=False, encoding='utf-8')
        
        self.logger.info(f"Saved {len(df_raw)} raw records to {output_path}")
    
    def save_cleaned_data(self, df_cleaned: pd.DataFrame, output_path: str) -> None:
        """Save cleaned data to CSV file."""
        if df_cleaned.empty:
            self.logger.warning("No cleaned data to save")
            return
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df_cleaned.to_csv(output_path, index=False, encoding='utf-8')
        
        self.logger.info(f"Saved {len(df_cleaned)} cleaned records to {output_path}")
    
    def run_scraper(self, query: str, max_results: int = 50, 
                    raw_output: Optional[str] = None,
                    cleaned_output: Optional[str] = None,
                    **kwargs) -> pd.DataFrame:
        """
        Run the complete scraping pipeline.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            raw_output: Path to save raw data
            cleaned_output: Path to save cleaned data
            **kwargs: Additional platform-specific parameters
            
        Returns:
            Cleaned and standardized DataFrame
        """
        self.logger.info(f"Starting {self.platform_name} scraper for query: '{query}'")
        
        try:
            # Fetch raw data
            raw_data = self.fetch_data(query, max_results, **kwargs)
            
            if not raw_data:
                self.logger.warning(f"No data found for query: '{query}'")
                return pd.DataFrame()
            
            # Save raw data if path provided
            if raw_output:
                self.save_raw_data(raw_data, raw_output)
            
            # Clean and standardize
            df_cleaned = self.clean_and_standardize(raw_data, query)
            
            # Save cleaned data if path provided
            if cleaned_output:
                self.save_cleaned_data(df_cleaned, cleaned_output)
            
            self.logger.info(f"Successfully processed {len(df_cleaned)} records")
            return df_cleaned
            
        except Exception as e:
            self.logger.error(f"Error in scraper: {str(e)}")
            raise
    
    @staticmethod
    def create_arg_parser(platform_name: str, description: str) -> argparse.ArgumentParser:
        """Create standardized argument parser."""
        parser = argparse.ArgumentParser(
            description=f"{description} - {platform_name.title()} Scraper"
        )
        parser.add_argument(
            "--query", 
            required=True, 
            help=f"Search query for {platform_name}"
        )
        parser.add_argument(
            "--max_results", 
            type=int, 
            default=50, 
            help="Maximum number of results to fetch"
        )
        parser.add_argument(
            "--raw_output", 
            default=f"data/raw/{platform_name}_raw.csv",
            help="Path to save raw data"
        )
        parser.add_argument(
            "--cleaned_output", 
            default=f"data/cleaned/{platform_name}_cleaned.csv",
            help="Path to save cleaned data"
        )
        
        return parser