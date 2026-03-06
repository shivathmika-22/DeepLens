"""
Twitter scraper using snscrape for data extraction.
Refactored to use BaseScraper for consistency.
"""

import sys
import os
import snscrape.modules.twitter as snstwitter
from typing import List, Dict, Any
from datetime import datetime
import certifi
import requests

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.base_scraper import BaseScraper


class TwitterScraper(BaseScraper):
    """Twitter scraper using snscrape."""
    
    def __init__(self):
        super().__init__("twitter")
    
    def fetch_data(self, query: str, max_results: int = 50, **kwargs) -> List[Dict[str, Any]]:
        """
        Fetch tweets using snscrape.
        
        Args:
            query: Search query
            max_results: Maximum number of tweets to fetch
            **kwargs: Additional parameters (not used for Twitter)
            
        Returns:
            List of tweet dictionaries
        """
        tweets = []
        
        try:
            self.logger.info(f"Fetching tweets for query: '{query}'")
            
            # Create scraper with proper SSL settings
            scraper = snstwitter.TwitterSearchScraper(query)
            scraper._session.verify = certifi.where()

            for i, tweet in enumerate(scraper.get_items()):
                if i >= max_results:
                    break
                
                tweet_data = {
                    'Content': tweet.content,
                    'Posted_At': tweet.date.strftime('%Y-%m-%d %H:%M:%S'),
                    'Username': tweet.user.username,
                    'URL': tweet.url,
                    'User_Followers': getattr(tweet.user, 'followersCount', 0),
                    'Retweet_Count': getattr(tweet, 'retweetCount', 0),
                    'Like_Count': getattr(tweet, 'likeCount', 0),
                    'Reply_Count': getattr(tweet, 'replyCount', 0),
                    'Language': getattr(tweet, 'lang', 'unknown'),
                    'Is_Retweet': getattr(tweet, 'retweetedTweet', None) is not None
                }
                
                tweets.append(tweet_data)
                
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Fetched {i + 1} tweets...")
            
            self.logger.info(f"Successfully fetched {len(tweets)} tweets")
            
        except Exception as e:
            self.logger.error(f"Error fetching tweets: {str(e)}")
            raise
        
        return tweets


def main():
    """Main function for command line usage."""
    parser = BaseScraper.create_arg_parser("twitter", "Scrape Twitter posts by query")
    args = parser.parse_args()
    
    scraper = TwitterScraper()
    
    try:
        df_cleaned = scraper.run_scraper(
            query=args.query,
            max_results=args.max_results,
            raw_output=args.raw_output,
            cleaned_output=args.cleaned_output
        )
        
        if not df_cleaned.empty:
            print(f"✅ Successfully processed {len(df_cleaned)} tweets")
            print(f"📁 Raw data saved to: {args.raw_output}")
            print(f"📁 Cleaned data saved to: {args.cleaned_output}")
        else:
            print("❌ No tweets found for the given query")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()