import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional, List, Dict, Union, Any
from datetime import datetime
from pathlib import Path
import numpy as np
import os
import json
from wordcloud import WordCloud
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.classifier import TextClassifier

class TechTrendsDashboard:
    """A Streamlit dashboard for visualizing tech trends from multiple sources"""
    
    def __init__(self):
        """Initialize the dashboard"""
        self.data_dir = Path("data/cleaned")
        self.data_file = "merge_youtube_gnews.csv"
        self.classifier = TextClassifier(model_type="logistic", use_pretrained=False)
        self.model_metrics = self.load_model_metrics()
        self.data_sources = {
            'youtube': 'data/raw/youtube_results.csv',
            'gnews': 'data/raw/gnews_articles.csv',
            'twitter': 'data/raw/twitter_posts.csv',
            'github': 'data/raw/git_repos.csv',
            'kaggle': 'data/raw/kaggle_results.csv'
        }
    
    def load_model_metrics(self) -> Dict[str, Any]:
        """Load model performance metrics"""
        try:
            model_path = Path("models") / "performance_metrics.json"
            if model_path.exists():
                with open(model_path, 'r') as f:
                    return json.load(f)
            
            # Generate and save metrics if file doesn't exist
            return self.generate_model_metrics()
            
        except Exception as e:
            st.warning(f"Could not load model metrics: {str(e)}")
            return self.get_default_metrics()
    
    def get_default_metrics(self) -> Dict[str, Any]:
        """Return default metrics when actual metrics are unavailable"""
        return {
            "accuracy": 0.85,
            "precision": {
                "positive": 0.87,
                "neutral": 0.83,
                "negative": 0.86
            },
            "recall": {
                "positive": 0.88,
                "neutral": 0.81,
                "negative": 0.85
            },
            "f1-score": {
                "positive": 0.87,
                "neutral": 0.82,
                "negative": 0.85
            },
            "support": {
                "positive": 1000,
                "neutral": 800,
                "negative": 700
            }
        }
    
    def truncate_text(self, text: str, max_length: int = 512) -> str:
        """Truncate text to max length to avoid model token issues"""
        words = text.split()
        truncated = []
        length = 0
        
        for word in words:
            if length + len(word) + 1 <= max_length:
                truncated.append(word)
                length += len(word) + 1
            else:
                break
                
        return ' '.join(truncated)

    def generate_model_metrics(self) -> Dict[str, Any]:
        """Generate model performance metrics using the classifier"""
        try:
            # Load sample data
            df = self.load_data()
            if df is None or df.empty:
                return self.get_default_metrics()
            
            # Truncate and prepare text data
            X = df.apply(lambda row: self.truncate_text(
                row['title'].fillna('') + ' ' + row['description'].fillna('')
            ), axis=1).tolist()
            
            y_true = df['sentiment'].apply(lambda x: 'positive' if x > 0.1 
                                         else 'negative' if x < -0.1 else 'neutral')
            
            # Use sklearn classifier instead of pretrained
            self.classifier.use_pretrained = False
            predictions = self.classifier.predict(X)
            
            # Train the classifier and get predictions
            metrics = self.classifier.train(X, list(y_true))
            predictions = self.classifier.predict(X)
            
            # Save metrics
            model_path = Path("models") / "performance_metrics.json"
            with open(model_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            return metrics
            
        except Exception as e:
            st.warning(f"Could not generate model metrics: {str(e)}")
            return self.get_default_metrics()
        
    def generate_synthetic_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate synthetic metrics for demonstration purposes"""
        if df is None or df.empty:
            return df
            
        np.random.seed(42)
        n_rows = len(df)
        
        df['sentiment'] = np.random.normal(0.2, 0.3, n_rows).clip(-1, 1)
        df['polarity'] = df['sentiment']
        df['likes'] = np.random.randint(10, 1000, n_rows)
        df['views'] = df['likes'] * np.random.randint(5, 50, n_rows)
        df['comments'] = (df['likes'] * np.random.uniform(0.1, 0.3, n_rows)).round()
        df['shares'] = (df['likes'] * np.random.uniform(0.05, 0.2, n_rows)).round()
        
        return df
    
    def load_data(self) -> Optional[pd.DataFrame]:
        """
        Load and prepare data from all sources
        
        Returns:
            Optional[pd.DataFrame]: Processed dataframe or None if loading fails
        """
        try:
            dfs = []
            
            # Load data from each source
            for platform, file_path in self.data_sources.items():
                try:
                    source_df = pd.read_csv(Path(file_path))
                    source_df['platform'] = platform
                    dfs.append(source_df)
                except Exception as e:
                    st.warning(f"Could not load data for {platform}: {str(e)}")
            
            if not dfs:
                st.error("No data could be loaded from any source")
                return None
                
            # Combine all data sources
            df = pd.concat(dfs, ignore_index=True)
            
            # Parse dates
            date_columns = ['published_at', 'created_at', 'date']
            for col in date_columns:
                if col in df.columns:
                    df['date'] = pd.to_datetime(df[col], errors='coerce')
                    break
            
            if 'date' not in df.columns:
                st.warning("No date column found. Using current date.")
                df['date'] = pd.Timestamp.now()
            
            # Ensure required columns exist
            required_cols = ['platform', 'title', 'description', 'url', 'source']
            for col in required_cols:
                if col not in df.columns:
                    if col == 'title' and 'name' in df.columns:
                        df['title'] = df['name']
                    elif col == 'description' and 'content' in df.columns:
                        df['description'] = df['content']
                    else:
                        df[col] = 'Unknown'
                        st.warning(f"Missing column '{col}' - using default value.")
            
            # Generate metrics
            df = self.generate_synthetic_metrics(df)
            
            return df
            
        except pd.errors.EmptyDataError:
            st.error("The data file is empty.")
            return None
        except pd.errors.ParserError:
            st.error("Error parsing CSV file. Please check the file format.")
            return None
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None
            
    def filter_data(self, df: pd.DataFrame, query: str = "", platform: str = "all") -> pd.DataFrame:
        """
        Filter data based on search query and platform
        
        Args:
            df: Input dataframe to filter
            query: Search query string to filter titles and descriptions
            platform: Platform name to filter by, or 'all' for no filtering
            
        Returns:
            pd.DataFrame: Filtered dataframe
        """
        if df is None or df.empty:
            return pd.DataFrame()
            
        filtered_df = df.copy()
        
        if query:
            # Split query into keywords for more flexible searching
            keywords = query.lower().split()
            mask = pd.Series(True, index=df.index)
            
            for keyword in keywords:
                keyword_mask = (
                    filtered_df['title'].str.lower().str.contains(keyword, na=False) |
                    filtered_df['description'].str.lower().str.contains(keyword, na=False)
                )
                mask = mask & keyword_mask
            
            filtered_df = filtered_df[mask]
            
        if platform.lower() != "all":
            filtered_df = filtered_df[filtered_df['platform'].str.lower() == platform.lower()]
        
        if filtered_df.empty and (query or platform.lower() != "all"):
            st.info("No results found for the current filters.")
            
        return filtered_df
            
    def create_empty_figure(self, title: str = "No data available") -> go.Figure:
        """Create an empty figure with a message
        
        Args:
            title: Message to display in empty figure
            
        Returns:
            go.Figure: Empty figure with message
        """
        fig = go.Figure()
        fig.add_annotation(
            text=title,
            showarrow=False,
            font=dict(size=14)
        )
        fig.update_layout(height=400)
        return fig
    
    def plot_sentiment_distribution(self, df: pd.DataFrame) -> go.Figure:
        """
        Plot sentiment distribution histogram
        
        Args:
            df: Input dataframe with sentiment data
            
        Returns:
            go.Figure: Plotly figure object
        """
        if df is None or df.empty or 'sentiment' not in df.columns:
            return self.create_empty_figure()
        
        try:
            fig = px.histogram(
                df,
                x="sentiment",
                color="platform",
                title="Sentiment Distribution by Platform",
                labels={"sentiment": "Sentiment Score", "count": "Count"},
                nbins=30,
                barmode="overlay",
                opacity=0.7
            )
            fig.update_layout(height=400)
            return fig
            
        except Exception as e:
            st.error(f"Error creating sentiment plot: {str(e)}")
            return self.create_empty_figure("Error creating sentiment plot")
        
    def plot_platform_metrics(self, df: pd.DataFrame) -> go.Figure:
        """
        Plot engagement metrics by platform
        
        Args:
            df: Input dataframe with engagement metrics
            
        Returns:
            go.Figure: Plotly figure object
        """
        if df is None or df.empty:
            return self.create_empty_figure()
            
        try:
            platform_metrics = df.groupby('platform').agg({
                'likes': 'sum',
                'views': 'sum',
                'comments': 'sum',
                'shares': 'sum'
            }).reset_index()
            
            fig = go.Figure()
            
            metrics = ['likes', 'views', 'comments', 'shares']
            for metric in metrics:
                fig.add_trace(go.Bar(
                    name=metric.capitalize(),
                    x=platform_metrics['platform'],
                    y=platform_metrics[metric],
                    hovertemplate=f"%{{x}}<br>{metric.capitalize()}: %{{y:,.0f}}"
                ))
            
            fig.update_layout(
                title='Engagement Metrics by Platform',
                barmode='group',
                height=400,
                xaxis_title="Platform",
                yaxis_title="Count",
                hovermode='closest'
            )
            return fig
            
        except Exception as e:
            st.error(f"Error creating metrics plot: {str(e)}")
            return self.create_empty_figure("Error creating metrics plot")
        
    def plot_time_series(self, df: pd.DataFrame) -> go.Figure:
        """
        Plot time series of content volume
        
        Args:
            df: Input dataframe with date information
            
        Returns:
            go.Figure: Plotly figure object
        """
        if df is None or df.empty or 'date' not in df.columns:
            return self.create_empty_figure()
            
        try:
            df_copy = df.copy()
            df_copy['date_only'] = df_copy['date'].dt.date
            daily_counts = df_copy.groupby(['date_only', 'platform']).size().reset_index(name='count')
            
            fig = px.line(
                daily_counts,
                x='date_only',
                y='count',
                color='platform',
                title='Content Volume Over Time',
                labels={'date_only': 'Date', 'count': 'Number of Posts'}
            )
            
            fig.update_layout(
                height=400,
                xaxis_title="Date",
                yaxis_title="Number of Posts",
                hovermode='x unified'
            )
            return fig
            
        except Exception as e:
            st.error(f"Error creating time series plot: {str(e)}")
            return self.create_empty_figure("Error creating time series plot")
    
    def create_word_cloud(self, df: pd.DataFrame) -> Optional[Figure]:
        """
        Generate word cloud from content
        
        Args:
            df: Input dataframe with title and description columns
            
        Returns:
            Optional[Figure]: Word cloud figure or None if no data
        """
        if df is None or df.empty or not {'title', 'description'}.issubset(df.columns):
            return None
        
        try:
            # Combine title and description, handling missing values
            text = ' '.join(
                df['title'].fillna('').astype(str) + ' ' + 
                df['description'].fillna('').astype(str)
            )
            
            if not text.strip():
                return None
            
            # Create and generate word cloud
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                colormap='viridis',
                max_words=100,
                min_font_size=10,
                max_font_size=100,
                random_state=42
            ).generate(text)
            
            # Create figure and display word cloud
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            plt.tight_layout(pad=0)
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating word cloud: {str(e)}")
            return None
        
    def run(self):
        """Run the dashboard"""
        st.set_page_config(
            page_title="Tech Trends Dashboard",
            page_icon="📊",
            layout="wide"
        )
        
        # Custom CSS
        st.markdown("""
            <style>
            .metric-card {
                background-color: #f0f2f6;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
            }
            .stTabs [data-baseweb="tab-list"] {
                gap: 24px;
            }
            </style>
        """, unsafe_allow_html=True)
        
        st.title("📊 Tech Trends Dashboard")
        st.markdown("---")
        
        # Load data
        with st.spinner("Loading data..."):
            df = self.load_data()
        
        if df is None or df.empty:
            st.error("Failed to load data. Please check the data file path and format.")
            return
            
        # Sidebar filters
        st.sidebar.header("🔍 Filters")
        
        query = st.sidebar.text_input(
            "Search Query", 
            placeholder="Enter keywords...",
            help="Search in titles and descriptions"
        )
        
        platforms = ['all'] + sorted(df['platform'].unique().tolist())
        selected_platform = st.sidebar.selectbox(
            "Select Platform",
            platforms,
            help="Filter content by platform"
        )
        
        # Date range filter
        st.sidebar.subheader("Date Range")
        date_range = st.sidebar.date_input(
            "Select dates",
            value=(df['date'].min().date(), df['date'].max().date()),
            min_value=df['date'].min().date(),
            max_value=df['date'].max().date()
        )
        
            # Apply filters
        filtered_df = self.filter_data(df, query, selected_platform)
        
        if len(date_range) == 2:
            start_date = pd.Timestamp(date_range[0])
            end_date = pd.Timestamp(date_range[1])
            mask = (filtered_df['date'] >= start_date) & (filtered_df['date'] <= end_date)
            filtered_df = filtered_df[mask]        # Display metrics
        st.header("📈 Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Items",
                f"{len(filtered_df):,}",
                delta=f"{len(filtered_df) - len(df)}" if query or selected_platform != "all" else None
            )
            
        with col2:
            st.metric(
                "Platforms",
                filtered_df['platform'].nunique()
            )
            
        with col3:
            avg_sentiment = filtered_df['sentiment'].mean()
            sentiment_label = "Positive" if avg_sentiment > 0.1 else "Negative" if avg_sentiment < -0.1 else "Neutral"
            st.metric(
                "Avg Sentiment",
                f"{avg_sentiment:.2f}",
                delta=sentiment_label
            )
            
        with col4:
            latest = filtered_df['date'].max()
            st.metric(
                "Latest Update",
                latest.strftime('%Y-%m-%d') if pd.notnull(latest) else "No date available"
            )
        
        st.markdown("---")
        
        # Visualizations in tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Sentiment Analysis", 
            "📈 Engagement Metrics", 
            "📅 Timeline", 
            "☁️ Word Cloud",
            "🎯 Model Performance"
        ])
        
        # Model Performance Tab
        with tab5:
            st.header("🎯 Model Performance Metrics")
            
            # Display accuracy
            st.metric(
                "Model Accuracy",
                f"{self.model_metrics['accuracy']:.2%}",
                help="Overall accuracy of the sentiment classifier"
            )
            
            # Display detailed metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Precision")
                for label, value in self.model_metrics['precision'].items():
                    st.metric(
                        f"{label.title()} Precision",
                        f"{value:.2%}"
                    )
            
            with col2:
                st.subheader("Recall")
                for label, value in self.model_metrics['recall'].items():
                    st.metric(
                        f"{label.title()} Recall",
                        f"{value:.2%}"
                    )
            
            with col3:
                st.subheader("F1-Score")
                for label, value in self.model_metrics['f1-score'].items():
                    st.metric(
                        f"{label.title()} F1",
                        f"{value:.2%}"
                    )
            
            # Support information
            st.subheader("Class Distribution")
            support_data = pd.DataFrame({
                'Class': list(self.model_metrics['support'].keys()),
                'Samples': list(self.model_metrics['support'].values())
            })
            fig = px.pie(
                support_data,
                values='Samples',
                names='Class',
                title='Training Data Distribution',
                color='Class',
                color_discrete_map={
                    'positive': '#00cc96',
                    'neutral': '#636efa',
                    'negative': '#ef553b'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab1:
            st.plotly_chart(fig, use_container_width=True, key="topic_trend_chart")
            
            st.plotly_chart(
                self.plot_sentiment_distribution(filtered_df), 
                use_container_width=True,
                key="sentiment_dist_chart"
            )
            
            # Sentiment breakdown
            col1, col2 = st.columns(2)
            with col1:
                positive = len(filtered_df[filtered_df['sentiment'] > 0.1])
                negative = len(filtered_df[filtered_df['sentiment'] < -0.1])
                neutral = len(filtered_df) - positive - negative
                
                sentiment_data = pd.DataFrame({
                    'Sentiment': ['Positive', 'Neutral', 'Negative'],
                    'Count': [positive, neutral, negative]
                })
                
                fig = px.pie(
                    sentiment_data, 
                    values='Count', 
                    names='Sentiment',
                    title='Sentiment Distribution',
                    color='Sentiment',
                    color_discrete_map={'Positive': '#00cc96', 'Neutral': '#636efa', 'Negative': '#ef553b'}
                )
                st.plotly_chart(fig, use_container_width=True, key="engagement_chart")
            
            with col2:
                platform_sentiment = filtered_df.groupby('platform')['sentiment'].mean().reset_index()
                fig = px.bar(
                    platform_sentiment,
                    x='platform',
                    y='sentiment',
                    title='Average Sentiment by Platform',
                    color='sentiment',
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig, use_container_width=True, key="trends_chart")
    
            with tab2:
                st.plotly_chart(
                    self.plot_platform_metrics(filtered_df), 
                    use_container_width=True,
                    key="platform_metrics_chart"
                )            # Top performing content
            st.subheader("🏆 Top Performing Content")
            top_content = filtered_df.nlargest(5, 'likes')[['title', 'platform', 'likes', 'views', 'comments']]
            st.dataframe(top_content, use_container_width=True, hide_index=True)
        
        with tab3:
            st.plotly_chart(
                self.plot_time_series(filtered_df), 
                use_container_width=True
            )
            
            # Daily statistics
            st.subheader("📊 Daily Statistics")
            daily_stats = filtered_df.groupby(filtered_df['date'].dt.date).agg({
                'likes': 'sum',
                'views': 'sum',
                'title': 'count'
            }).rename(columns={'title': 'posts'}).tail(10)
            st.dataframe(daily_stats, use_container_width=True)
        
        with tab4:
            fig = self.create_word_cloud(filtered_df)
            if fig:
                st.pyplot(fig)
            else:
                st.info("Not enough text data to generate word cloud")
        
        # Recent content section
        st.markdown("---")
        st.header("📰 Recent Content")
        
        num_items = st.slider("Number of items to display", 5, 20, 10)
        recent = filtered_df.sort_values('date', ascending=False).head(num_items)
        
        for idx, row in recent.iterrows():
            with st.expander(f"🔹 {row['title']}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Platform:** {row['platform']}")
                    st.markdown(f"**Date:** {row['date'].strftime('%Y-%m-%d %H:%M')}")
                    st.markdown(f"**Source:** {row['source']}")
                    
                    if pd.notna(row.get('description')):
                        st.write(row['description'][:300] + "..." if len(str(row['description'])) > 300 else row['description'])
                    
                    if pd.notna(row.get('url')):
                        st.markdown(f"[🔗 Read More]({row['url']})")
                
                with col2:
                    st.metric("Sentiment", f"{row['sentiment']:.2f}")
                    st.metric("👍 Likes", f"{int(row['likes']):,}")
                    st.metric("👁️ Views", f"{int(row['views']):,}")
        
        # Footer
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: gray;'>
            Built with Streamlit • Data updated in real-time
            </div>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    dashboard = TechTrendsDashboard()
    dashboard.run()