"""
Facebook scraper using RSS feeds for data extraction.
Refactored to use BaseScraper for consistency.
"""

import sys
import os
import feedparser
from typing import List, Dict, Any
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.base_scraper import BaseScraper


class FacebookScraper(BaseScraper):
    """Facebook scraper using RSS feeds."""
    
    def __init__(self):
        super().__init__("facebook")
    
    def fetch_data(self, query: str, max_results: int = 50, **kwargs) -> List[Dict[str, Any]]:
        """
        Fetch Facebook posts using RSS feeds.
        
        Args:
            query: Facebook page name to scrape
            max_results: Maximum number of posts to fetch
            **kwargs: Additional parameters (not used for Facebook)
            
        Returns:
            List of post dictionaries
        """
        posts = []
        
        try:
            page_name = query  # For Facebook, query is the page name
            self.logger.info(f"Fetching Facebook posts from page: '{page_name}'")
            
            # Construct Facebook RSS URL
            rss_url = f"https://www.facebook.com/pg/{page_name}/posts/?format=rss"
            
            self.logger.info(f"Fetching RSS feed from: {rss_url}")
            
            # Parse RSS feed
            feed = feedparser.parse(rss_url)
            
            if not feed.entries:
                self.logger.warning(f"No posts found for page: {page_name}")
                return posts
            
            # Process feed entries
            for entry in feed.entries[:max_results]:
                # Parse published date
                published_date = ""
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    published_date = datetime(*entry.published_parsed[:6]).isoformat()
                elif hasattr(entry, 'published'):
                    published_date = entry.published
                
                post_data = {
                    'Title': entry.get("title", ""),
                    'Summary': entry.get("summary", "").replace("\n", " ").strip(),
                    'Link': entry.get("link", ""),
                    'Published': published_date,
                    'Page': page_name,
                    'Author': entry.get("author", ""),
                    'Tags': ', '.join([tag.term for tag in entry.get("tags", [])]),
                    'Feed_Title': feed.feed.get("title", ""),
                    'Feed_Description': feed.feed.get("description", ""),
                    'Feed_Link': feed.feed.get("link", ""),
                    'Entry_ID': entry.get("id", ""),
                    'Updated': entry.get("updated", "")
                }
                
                posts.append(post_data)
            
            self.logger.info(f"Successfully fetched {len(posts)} posts")
            
        except Exception as e:
            self.logger.error(f"Error fetching Facebook posts: {str(e)}")
            raise
        
        return posts


def main():
    """Main function for command line usage."""
    parser = BaseScraper.create_arg_parser("facebook", "Fetch posts from Facebook public page via RSS")
    parser.add_argument("--page", required=True, help="Facebook page name (e.g., techcrunch)")
    args = parser.parse_args()
    
    scraper = FacebookScraper()
    
    try:
        df_cleaned = scraper.run_scraper(
            query=args.page,  # Use page name as query
            max_results=args.max_results,
            raw_output=args.raw_output,
            cleaned_output=args.cleaned_output
        )
        
        if not df_cleaned.empty:
            print(f"✅ Successfully processed {len(df_cleaned)} posts")
            print(f"📁 Raw data saved to: {args.raw_output}")
            print(f"📁 Cleaned data saved to: {args.cleaned_output}")
        else:
            print("❌ No posts found for the given page")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
