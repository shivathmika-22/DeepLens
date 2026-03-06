"""
YouTube scraper using Google API for data extraction.
Refactored to use BaseScraper for consistency.
"""

import sys
import os
try:
    from googleapiclient.discovery import build
except Exception:
    build = None  # Will check at runtime and provide a clear error message
from typing import List, Dict, Any
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.api_keys_template import YOUTUBE_API_KEY
from utils.base_scraper import BaseScraper


class YouTubeScraper(BaseScraper):
    """YouTube scraper using Google API."""
    
    def __init__(self):
        super().__init__("youtube")
        self.youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    
    def fetch_data(self, query: str, max_results: int = 50, **kwargs) -> List[Dict[str, Any]]:
        """
        Fetch YouTube videos using Google API.
        
        Args:
            query: Search query
            max_results: Maximum number of videos to fetch
            **kwargs: Additional parameters (order, type, etc.)
            
        Returns:
            List of video dictionaries
        """
        videos = []
        
        try:
            self.logger.info(f"Fetching YouTube videos for query: '{query}'")
            
            # Get search results
            request = self.youtube.search().list(
                q=query,
                part='snippet',
                maxResults=min(max_results, 50),  # API limit is 50 per request
                type='video',
                order=kwargs.get('order', 'date')
            )
            
            response = request.execute()
            
            # Get video details for each result
            video_ids = [item['id']['videoId'] for item in response['items']]
            
            if video_ids:
                # Get detailed video information
                video_request = self.youtube.videos().list(
                    part='snippet,statistics',
                    id=','.join(video_ids)
                )
                video_response = video_request.execute()
                
                # Create mapping of video_id to details
                video_details = {video['id']: video for video in video_response['items']}
                
                for item in response['items']:
                    video_id = item['id']['videoId']
                    snippet = item['snippet']
                    details = video_details.get(video_id, {})
                    stats = details.get('statistics', {})
                    
                    video_data = {
                        'Title': snippet['title'],
                        'channel': snippet['channelTitle'],
                        'Published At': snippet['publishedAt'],
                        'Description': snippet['description'],
                        'URL': f"https://www.youtube.com/watch?v={video_id}",
                        'Video_ID': video_id,
                        'Channel_ID': snippet['channelId'],
                        'View_Count': int(stats.get('viewCount', 0)),
                        'Like_Count': int(stats.get('likeCount', 0)),
                        'Comment_Count': int(stats.get('commentCount', 0)),
                        'Duration': details.get('contentDetails', {}).get('duration', ''),
                        'Thumbnail_URL': snippet['thumbnails']['high']['url'],
                        'Category_ID': details.get('snippet', {}).get('categoryId', ''),
                        'Tags': ', '.join(details.get('snippet', {}).get('tags', []))
                    }
                    
                    videos.append(video_data)
            
            self.logger.info(f"Successfully fetched {len(videos)} videos")
            
        except Exception as e:
            self.logger.error(f"Error fetching YouTube videos: {str(e)}")
            raise
        
        return videos


# Backwards-compatible alias: some modules import `YoutubeScraper` (different casing)
YoutubeScraper = YouTubeScraper


def main():
    """Main function for command line usage."""
    parser = BaseScraper.create_arg_parser("youtube", "Fetch YouTube videos by query")
    parser.add_argument("--order", default="date", choices=["date", "rating", "relevance", "title", "videoCount", "viewCount"],
                       help="Order of results")
    args = parser.parse_args()
    
    scraper = YouTubeScraper()
    
    try:
        df_cleaned = scraper.run_scraper(
            query=args.query,
            max_results=args.max_results,
            raw_output=args.raw_output,
            cleaned_output=args.cleaned_output,
            order=args.order
        )
        
        if not df_cleaned.empty:
            print(f"✅ Successfully processed {len(df_cleaned)} videos")
            print(f"📁 Raw data saved to: {args.raw_output}")
            print(f"📁 Cleaned data saved to: {args.cleaned_output}")
        else:
            print("❌ No videos found for the given query")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()