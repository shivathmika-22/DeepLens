import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import streamlit as st
import os
import sys

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.feature_extractor import FeatureExtractor

# Lazy imports to avoid circular dependencies
def get_scrapers():
    from scripts.youtube_scraper import YoutubeScraper
    from scripts.twitter_scraper import TwitterScraper
    from scripts.github_scraper import GithubScraper
    from scripts.kaggle_scraper import KaggleScraper
    return {
        'youtube': YoutubeScraper,
        'twitter': TwitterScraper,
        'github': GithubScraper,
        'kaggle': KaggleScraper
    }


# ... (Assume this is a function that needs the fetcher) ...
def some_utility_function(self, some_parameter):
    # ADD the import here, inside the function
    from pipeline.fetch_news import NewsFetcher 

    fetcher = NewsFetcher()
    # ... rest of the function logic ...
def load_data(query, platforms, max_results):
    """Load and process data from selected platforms."""
    data = []
    # NOTE: Assuming FeatureExtractor is imported correctly at the top of utils.py
    feature_extractor = FeatureExtractor() 
    
    for platform in platforms:
        # Use st.spinner for user feedback if needed (optional)
        # with st.spinner(f"Fetching data from {platform}..."):
        try:
            if platform == "gnews":
                # Import NewsFetcher here, inside the function, to break the circular import.
                from pipeline.fetch_news import NewsFetcher 
                fetcher = NewsFetcher()
                platform_data = fetcher.fetch_news(query, max_results)
            else:
                # Get the appropriate scraper class
                scrapers = get_scrapers()
                if platform in scrapers:
                    scraper = scrapers[platform]()
                    if platform == "youtube":
                        platform_data = scraper.search_videos(query, max_results)
                    elif platform == "twitter":
                        platform_data = scraper.fetch_tweets(query, max_results)
                    elif platform == "github":
                        platform_data = scraper.search_repos(query, max_results)
                    elif platform == "kaggle":
                        platform_data = scraper.search_datasets(query, max_results)
                    else:
                        platform_data = None
                else:
                    platform_data = None
                
            if platform_data:
                # Extract features
                for item in platform_data:
                    # Check if 'content' exists before extracting features
                    content = item.get("content", "")
                    if content:
                        features = feature_extractor.extract_features(content)
                        item.update(features)
                    
                    item["platform"] = platform
                data.extend(platform_data)
                
        except Exception as e:
            # You are using st.error, so this function is likely called from the app context
            st.error(f"Error fetching data from {platform}: {str(e)}")
    
    return pd.DataFrame(data) if data else pd.DataFrame()

def plot_sentiment_analysis(df):
    """Create sentiment analysis visualizations."""
    # Sentiment distribution
    fig_sentiment = px.histogram(
        df,
        x="polarity",
        nbins=20,
        title="Sentiment Distribution",
        color="platform"
    )
    st.plotly_chart(fig_sentiment)
    
    # Average sentiment by platform
    avg_sentiment = df.groupby("platform")["polarity"].mean().reset_index()
    fig_platform = px.bar(
        avg_sentiment,
        x="platform",
        y="polarity",
        title="Average Sentiment by Platform",
        color="platform"
    )
    st.plotly_chart(fig_platform)

def plot_entity_distribution(df):
    #Create entity distribution visualizations.
    entity_counts = Counter()
    for entities in df["named_entities"]:
        entity_counts.update(entities)
    
    # Convert to DataFrame
    entity_df = pd.DataFrame.from_dict(entity_counts, orient="index").reset_index()
    entity_df.columns = ["Entity", "Count"]
    entity_df = entity_df.sort_values("Count", ascending=True).tail(20)
    
    fig = px.bar(
        entity_df,
        x="Count",
        y="Entity",
        orientation="h",
        title="Top 20 Named Entities"
    )
    st.plotly_chart(fig)

def plot_engagement_metrics(df):
    """Create engagement metrics visualizations."""
    # Get engagement columns
    engagement_cols = [col for col in df.columns if col.startswith("engagement_")]
    
    if engagement_cols:
        # Average engagement by platform
        engagement_data = []
        for col in engagement_cols:
            metric = col.replace("engagement_", "").title()
            platform_avg = df.groupby("platform")[col].mean().reset_index()
            platform_avg["Metric"] = metric
            engagement_data.append(platform_avg)
        
        engagement_df = pd.concat(engagement_data)
        
        fig = px.bar(
            engagement_df,
            x="platform",
            y=col,
            color="Metric",
            title="Average Engagement by Platform",
            barmode="group"
        )
        st.plotly_chart(fig)
        
        # Engagement over time if timestamp available
        if "timestamp" in df.columns:
            df["date"] = pd.to_datetime(df["timestamp"]).dt.date
            time_data = []
            for col in engagement_cols:
                metric = col.replace("engagement_", "").title()
                daily_avg = df.groupby("date")[col].mean().reset_index()
                daily_avg["Metric"] = metric
                time_data.append(daily_avg)
            
            time_df = pd.concat(time_data)
            
            fig = px.line(
                time_df,
                x="date",
                y=col,
                color="Metric",
                title="Engagement Trends Over Time"
            )
            st.plotly_chart(fig)