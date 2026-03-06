"""
News fetching pipeline that coordinates multiple scrapers.
"""

import os
import sys
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.cleaner import clean_dataframe

def get_scrapers():
    """Lazy import scrapers to avoid circular dependencies"""
    from scripts.twitter_scraper import TwitterScraper
    from scripts.youtube_scraper import YoutubeScraper
    from scripts.gnews_scraper import GNewsScraper
    from scripts.github_scraper import GithubScraper
    from scripts.kaggle_scraper import KaggleScraper
    return {
        'twitter': TwitterScraper,
        'youtube': YoutubeScraper,
        'gnews': GNewsScraper,
        'github': GithubScraper,
        'kaggle': KaggleScraper
    }


class NewsFetcher:
    """Main class for fetching news from multiple sources."""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self._scrapers = None
    
    @property
    def scrapers(self):
        """Lazy load scrapers only when needed"""
        if self._scrapers is None:
            scraper_classes = get_scrapers()
            self._scrapers = {
                name: scraper_class() 
                for name, scraper_class in scraper_classes.items()
            }
        return self._scrapers
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for the news fetcher."""
        logger = logging.getLogger("news_fetcher")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def fetch_from_platform(self, platform: str, query: str, max_results: int = 50, **kwargs) -> pd.DataFrame:
        """
        Fetch data from a specific platform.
        
        Args:
            platform: Platform name (twitter, youtube, gnews, facebook, linkedin)
            query: Search query
            max_results: Maximum number of results
            **kwargs: Platform-specific parameters
            
        Returns:
            DataFrame with cleaned data
        """
        if platform not in self.scrapers:
            raise ValueError(f"Unknown platform: {platform}. Available: {list(self.scrapers.keys())}")
        
        self.logger.info(f"Fetching data from {platform} for query: '{query}'")
        
        try:
            scraper = self.scrapers[platform]
            df = scraper.run_scraper(
                query=query,
                max_results=max_results,
                **kwargs
            )
            
            self.logger.info(f"Successfully fetched {len(df)} records from {platform}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching from {platform}: {str(e)}")
            raise
    
    def fetch_from_all_platforms(self, query: str, max_results_per_platform: int = 50, 
                                platforms: Optional[List[str]] = None, **kwargs) -> pd.DataFrame:
        """
        Fetch data from all or specified platforms.
        
        Args:
            query: Search query
            max_results_per_platform: Maximum results per platform
            platforms: List of platforms to fetch from (None for all)
            **kwargs: Additional parameters
            
        Returns:
            Combined DataFrame with data from all platforms
        """
        if platforms is None:
            platforms = list(self.scrapers.keys())
        
        all_data = []
        
        for platform in platforms:
            try:
                self.logger.info(f"Fetching from {platform}...")
                df = self.fetch_from_platform(platform, query, max_results_per_platform, **kwargs)
                
                if not df.empty:
                    all_data.append(df)
                    self.logger.info(f"✓ {platform}: {len(df)} records")
                else:
                    self.logger.warning(f"⚠ {platform}: No data found")
                    
            except Exception as e:
                self.logger.error(f"❌ {platform}: {str(e)}")
                continue
        
        if not all_data:
            self.logger.warning("No data fetched from any platform")
            return pd.DataFrame()
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Add metadata
        combined_df['fetched_at'] = datetime.now().isoformat()
        combined_df['total_sources'] = len(all_data)
        
        self.logger.info(f"Successfully combined data from {len(all_data)} platforms: {len(combined_df)} total records")
        
        return combined_df
    
    def save_combined_dataset(self, df: pd.DataFrame, output_path: str = "data/cleaned/cleaned_dataset.csv") -> None:
        """
        Save combined dataset to CSV.
        
        Args:
            df: Combined DataFrame
            output_path: Output file path
        """
        if df.empty:
            self.logger.warning("No data to save")
            return
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Ensure UTF-8 encoding
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        self.logger.info(f"Saved combined dataset with {len(df)} records to {output_path}")
        
        # Print summary
        self._print_dataset_summary(df)
    
    def _print_dataset_summary(self, df: pd.DataFrame) -> None:
        """Print summary of the dataset."""
        print("\n" + "="*50)
        print("📊 DATASET SUMMARY")
        print("="*50)
        print(f"Total records: {len(df)}")
        print(f"Platforms: {df['platform'].value_counts().to_dict()}")
        print(f"Date range: {df['published_date'].min()} to {df['published_date'].max()}")
        print(f"Unique queries: {df['query'].nunique()}")
        print("="*50)
    
    def run_full_pipeline(self, query: str, max_results_per_platform: int = 50,
                         platforms: Optional[List[str]] = None,
                         output_path: str = "data/cleaned/cleaned_dataset.csv",
                         **kwargs) -> pd.DataFrame:
        """
        Run the complete news fetching pipeline.
        
        Args:
            query: Search query
            max_results_per_platform: Maximum results per platform
            platforms: List of platforms to fetch from
            output_path: Output file path
            **kwargs: Additional parameters
            
        Returns:
            Combined DataFrame
        """
        self.logger.info(f"Starting full pipeline for query: '{query}'")
        
        # Fetch data from all platforms
        df = self.fetch_from_all_platforms(
            query=query,
            max_results_per_platform=max_results_per_platform,
            platforms=platforms,
            **kwargs
        )
        
        if not df.empty:
            # Save combined dataset
            self.save_combined_dataset(df, output_path)
            
            self.logger.info("✅ Pipeline completed successfully")
        else:
            self.logger.warning("⚠️ Pipeline completed but no data was fetched")
        
        return df


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch news from multiple platforms")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--max_results", type=int, default=50, help="Max results per platform")
    parser.add_argument("--platforms", nargs="+", choices=['twitter', 'youtube', 'gnews', 'facebook', 'linkedin'],
                       help="Platforms to fetch from (default: all)")
    parser.add_argument("--output", default="data/cleaned/cleaned_dataset.csv", help="Output file path")
    
    args = parser.parse_args()
    
    fetcher = NewsFetcher()
    
    try:
        df = fetcher.run_full_pipeline(
            query=args.query,
            max_results_per_platform=args.max_results,
            platforms=args.platforms,
            output_path=args.output
        )
        
        if not df.empty:
            print(f"\n✅ Successfully processed {len(df)} records")
            print(f"📁 Data saved to: {args.output}")
        else:
            print("\n❌ No data was fetched")
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
