"""
GNews scraper using GNews API for data extraction.
Refactored to use BaseScraper for consistency.
"""

import sys
import os
import requests
import urllib.parse
from typing import List, Dict, Any
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.api_keys_template import GNEWS_API_KEY
from utils.base_scraper import BaseScraper


class GNewsScraper(BaseScraper):
    """GNews scraper using GNews API."""
    
    def __init__(self):
        super().__init__("gnews")
        self.api_key = GNEWS_API_KEY
        self.base_url = "https://gnews.io/api/v4/search"
    
    def fetch_data(self, query: str, max_results: int = 50, **kwargs) -> List[Dict[str, Any]]:
        """
        Fetch news articles using GNews API.
        
        Args:
            query: Search query
            max_results: Maximum number of articles to fetch
            **kwargs: Additional parameters (lang, country, etc.)
            
        Returns:
            List of article dictionaries
        """
        articles = []
        
        try:
            self.logger.info(f"Fetching GNews articles for query: '{query}'")
            
            # Prepare request parameters
            params = {
                'q': query,
                'lang': kwargs.get('lang', 'en'),
                'max': min(max_results, 100),  # API limit is 100
                'apikey': self.api_key
            }
            
            # Add optional parameters
            if 'country' in kwargs:
                params['country'] = kwargs['country']
            if 'from' in kwargs:
                params['from'] = kwargs['from']
            if 'to' in kwargs:
                params['to'] = kwargs['to']
            if 'sortby' in kwargs:
                params['sortby'] = kwargs['sortby']
            
            # Make API request
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "articles" not in data or not data["articles"]:
                self.logger.warning(f"No articles found for query: {query}")
                return articles
            
            # Process articles
            for article in data["articles"]:
                article_data = {
                    'Title': article.get("title", ""),
                    'Description': article.get("description", ""),
                    'PublishedAt': article.get("publishedAt", ""),
                    'URL': article.get("url", ""),
                    'Source': article.get("source", {}).get("name", ""),
                    'Source_URL': article.get("source", {}).get("url", ""),
                    'Image_URL': article.get("image", ""),
                    'Content': article.get("content", ""),
                    'Language': article.get("language", "en"),
                    'Country': article.get("country", []),
                    'Category': article.get("category", ""),
                    'Authors': ', '.join(article.get("authors", [])),
                    'Tags': ', '.join(article.get("tags", [])),
                    'Related_Articles': len(article.get("related", []))
                }
                
                articles.append(article_data)
            
            self.logger.info(f"Successfully fetched {len(articles)} articles")
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request error: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Error fetching GNews articles: {str(e)}")
            raise
        
        return articles


def main():
    """Main function for command line usage."""
    parser = BaseScraper.create_arg_parser("gnews", "Scrape GNews articles by keyword")
    parser.add_argument("--lang", default="en", help="Language code (e.g., 'en', 'es', 'fr')")
    parser.add_argument("--country", help="Country code (e.g., 'us', 'in', 'gb')")
    parser.add_argument("--sortby", default="publishedAt", choices=["publishedAt", "relevance"],
                       help="Sort order")
    parser.add_argument("--from_date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--to_date", help="End date (YYYY-MM-DD)")
    args = parser.parse_args()
    
    scraper = GNewsScraper()
    
    try:
        # Prepare kwargs
        kwargs = {
            'lang': args.lang,
            'sortby': args.sortby
        }
        
        if args.country:
            kwargs['country'] = args.country
        if args.from_date:
            kwargs['from'] = args.from_date
        if args.to_date:
            kwargs['to'] = args.to_date
        
        df_cleaned = scraper.run_scraper(
            query=args.query,
            max_results=args.max_results,
            raw_output=args.raw_output,
            cleaned_output=args.cleaned_output,
            **kwargs
        )
        
        if not df_cleaned.empty:
            print(f"✅ Successfully processed {len(df_cleaned)} articles")
            print(f"📁 Raw data saved to: {args.raw_output}")
            print(f"📁 Cleaned data saved to: {args.cleaned_output}")
        else:
            print("❌ No articles found for the given query")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
