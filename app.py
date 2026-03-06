"""
DeepLens Streamlit Application
A comprehensive news aggregation and analysis platform.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os
import time

# Go up two levels from app/app.py to the project root (Deeplens)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# ... (All other imports follow) ...
from pipeline.fetch_news import NewsFetcher
from pipeline.preprocess import DataPreprocessor
from models.summarizer import TextSummarizer, MultiModelSummarizer
from models.classifier import SentimentClassifier, TopicClassifier
from utils.cleaner import TextCleaner


# Page configuration
st.set_page_config(
    page_title="DeepLens - Innovative platform for learners",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Basic CSS (restored to simpler appearance)
st.markdown("""
<style>
    .main-header {
        font-size: 2.6rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1.25rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #ffffff; /* ensure contrast inside the box */
        padding: 1rem;
        border-radius: 0.6rem;
        border-left: 5px solid #2563eb;
        box-shadow: 0 6px 14px rgba(15, 23, 42, 0.06);
        color: #0f172a; /* dark text for readability */
    }
    .metric-card h4 { margin: 0 0 6px 0; color: #0f172a; font-weight:600; }
    .metric-card h2 { margin: 0; font-size: 1.5rem; color: #0b1220; font-weight:700; }
    .metric-small { color: #475569; margin-top: 6px; }
    .app-container { background: linear-gradient(180deg, #ffffff 0%, #f7f9fc 100%); padding: 1rem 2rem; }
    /* Right-side insights panel: use dark background with light text for visibility */
    .insights-card { background: transparent; padding: 0.9rem; border-radius: 0.6rem; margin-bottom: 0.75rem; color: #e6eef8; }
    .insights-title { font-weight:700; color:#e6eef8; margin-bottom:0.5rem; }
    /* Ensure list items and small text inside insights panel are visible on dark themes */
    .insights-card, .insights-card p, .insights-card li, .insights-card span, .insights-card a {
        color: #e6eef8 !important;
    }
    .insights-card ul { padding-left: 1rem; margin: 0; }
    .insights-card li { margin-bottom: 6px; }
    /* Style Streamlit buttons inside insights card to be prominent */
    .insights-card .stButton>button, .insights-card .stDownloadButton>button {
        background: #2563eb !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 6px 12px rgba(37,99,235,0.16) !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_sample_data():
    """Load sample data for demonstration."""
    sample_data = {
        'platform': ['twitter', 'youtube', 'gnews', 'facebook', 'linkedin'],
        'title': [
            'AI Revolution in Healthcare',
            'Latest Tech Trends 2024',
            'Breaking: New AI Model Released',
            'Tech Industry Updates',
            'Data Science Career Tips'
        ],
        'content': [
            'Artificial Intelligence is transforming healthcare with new diagnostic tools...',
            'The latest technology trends for 2024 include AI, blockchain, and quantum computing...',
            'A new AI model has been released that can process natural language...',
            'The tech industry continues to evolve with new innovations...',
            'Tips for building a successful career in data science...'
        ],
        'published_date': [
            '2024-01-15T10:30:00Z',
            '2024-01-15T09:15:00Z',
            '2024-01-15T08:45:00Z',
            '2024-01-15T07:20:00Z',
            '2024-01-15T06:10:00Z'
        ],
        'author': ['@technews', 'TechChannel', 'AI Research', 'TechPage', 'DataExpert'],
        'url': ['https://twitter.com/...', 'https://youtube.com/...', 'https://gnews.io/...', 
                'https://facebook.com/...', 'https://linkedin.com/...'],
        'query': ['AI healthcare', 'AI healthcare', 'AI healthcare', 'AI healthcare', 'AI healthcare']
    }
    return pd.DataFrame(sample_data)


def initialize_session_state():
    """Initialize session state variables."""
    if 'data_fetched' not in st.session_state:
        st.session_state.data_fetched = False
    if 'current_data' not in st.session_state:
        st.session_state.current_data = pd.DataFrame()
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = pd.DataFrame()
    if 'summaries' not in st.session_state:
        st.session_state.summaries = {}
    if 'classifications' not in st.session_state:
        st.session_state.classifications = {}
    if 'last_clicked_metric' not in st.session_state:
        st.session_state.last_clicked_metric = None
    if 'metric_value' not in st.session_state:
        st.session_state.metric_value = None


def main():
    """Main application function."""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🔍 DeepLens</h1>', unsafe_allow_html=True)
    st.markdown('<p class="brand-subtitle">News Intelligence Platform — Aggregate, Analyze, Act</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Control Panel")
        # Theme controls removed (restored to simpler UI)
        
        # Data Source Selection
        st.subheader("📊 Data Source")
        data_source = st.selectbox(
            "Choose data source:",
            ["Fetch New Data", "Load Sample Data", "Upload CSV"]
        )
        
        # Query Input
        if data_source == "Fetch New Data":
            st.subheader("🔍 Search Query")
            query = st.text_input(
                "Enter search query:",
                placeholder="e.g., AI in healthcare, climate change, tech news"
            )
            
            # Platform Selection
            st.subheader("🌐 Platforms")
            platforms = st.multiselect(
                "Select platforms:",
                ["twitter", "youtube", "gnews", "facebook", "linkedin"],
                default=["gnews", "twitter"]
            )
            
            # Parameters
            st.subheader("⚙️ Parameters")
            max_results = st.slider("Max results per platform:", 10, 100, 50)
            
            # Fetch Button
            if st.button("🚀 Fetch Data", type="primary"):
                if query and platforms:
                    fetch_data(query, platforms, max_results)
                else:
                    st.error("Please enter a query and select at least one platform.")
        
        elif data_source == "Load Sample Data":
            if st.button("📋 Load Sample Data"):
                st.session_state.current_data = load_sample_data()
                st.session_state.data_fetched = True
                st.success("Sample data loaded successfully!")
        
        elif data_source == "Upload CSV":
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.current_data = df
                    st.session_state.data_fetched = True
                    st.success(f"Data loaded successfully! {len(df)} records.")
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
    
    # Main Content
    if st.session_state.data_fetched and not st.session_state.current_data.empty:
        display_main_content()
    else:
        display_welcome_screen()


def render_metric_card(col, title, value, subtitle=None, color="#667eea"):
    """Render a small metric card inside a column."""
    with col:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: {color};">
            <h4>{title}</h4>
            <h2>{value}</h2>
            <div class="metric-small">{subtitle or ''}</div>
        </div>
        """, unsafe_allow_html=True)


def fetch_data(query, platforms, max_results):
    """Fetch data from selected platforms."""
    with st.spinner("🔄 Fetching data from platforms..."):
        try:
            fetcher = NewsFetcher()
            df = fetcher.fetch_from_all_platforms(
                query=query,
                max_results_per_platform=max_results,
                platforms=platforms
            )
            
            if not df.empty:
                st.session_state.current_data = df
                st.session_state.data_fetched = True
                st.success(f"✅ Successfully fetched {len(df)} records!")
            else:
                st.error("❌ No data was fetched. Please try a different query.")
                
        except Exception as e:
            st.error(f"❌ Error fetching data: {str(e)}")


def display_welcome_screen():
    """Display welcome screen when no data is loaded."""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        ## Welcome to DeepLens! 🎉
        
        DeepLens is your comprehensive news intelligence platform that helps you:
        
        - 🔍 **Aggregate** news from multiple sources (Twitter, YouTube, GNews, Facebook, LinkedIn)
        - 🧹 **Clean** and preprocess text data automatically
        - 📝 **Summarize** articles using AI models
        - 🏷️ **Classify** content by topics and sentiment
        - 📊 **Visualize** trends and insights
        
        ### Getting Started:
        1. **Enter a search query** in the sidebar
        2. **Select platforms** to fetch data from
        3. **Click "Fetch Data"** to start the process
        4. **Explore** the results and insights
        
        ### Features:
        - **Multi-source aggregation**: Get news from 5+ platforms
        - **AI-powered summarization**: T5, BART, and extractive methods
        - **Smart classification**: Topic and sentiment analysis
        - **Interactive visualizations**: Charts and graphs
        - **Export capabilities**: Download processed data
        
        Ready to start? Use the sidebar to begin! 🚀
        """)


def display_main_content():
    """Display main content when data is available."""
    df = st.session_state.current_data
    # Data Overview
    st.header("📊 Data Overview")

    cols = st.columns(4)
    total_records = len(df)
    platforms = int(df['platform'].nunique()) if 'platform' in df.columns else 0
    unique_queries = int(df['query'].nunique()) if 'query' in df.columns else 0
    if 'published_date' in df.columns:
        latest_date = pd.to_datetime(df['published_date']).max()
        latest_str = latest_date.strftime('%Y-%m-%d')
    else:
        latest_str = "N/A"

    # Higher-contrast metric cards
    render_metric_card(cols[0], "Total Records", total_records, subtitle="Total items in view", color="#2563eb")
    render_metric_card(cols[1], "Platforms", platforms, subtitle="Distinct sources", color="#059669")
    render_metric_card(cols[2], "Unique Queries", unique_queries, subtitle="Search terms tracked", color="#d97706")
    render_metric_card(cols[3], "Latest Update", latest_str, subtitle="Most recent publish date", color="#dc2626")

    # Main two-column layout: left for tabs, right for quick insights
    left_col, right_col = st.columns([3, 1], gap="large")

    with left_col:
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Raw Data", "🔧 Processing", "📝 Summarization", "🏷️ Classification", "📈 Analytics"])

        with tab1:
            display_raw_data_tab(df)

        with tab2:
            display_processing_tab(df)

        with tab3:
            display_summarization_tab(df)

        with tab4:
            display_classification_tab(df)

        with tab5:
            display_analytics_tab(df)

    with right_col:
        # Insights / quick actions panel
        st.markdown('<div class="insights-card">', unsafe_allow_html=True)
        st.markdown('<div class="insights-title">Top Platforms</div>', unsafe_allow_html=True)
        if 'platform' in df.columns:
            top_platforms = df['platform'].value_counts().head(5)
            for p, c in top_platforms.items():
                st.write(f"- **{p}** — {int(c)}")
        else:
            st.write("No platform data")

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="insights-card">', unsafe_allow_html=True)
        st.markdown('<div class="insights-title">Recent Queries</div>', unsafe_allow_html=True)
        if 'query' in df.columns:
            recent_queries = pd.Series(df['query']).dropna().unique()[:6]
            for q in recent_queries:
                st.write(f"- {q}")
        else:
            st.write("No queries captured")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="insights-card actions">', unsafe_allow_html=True)
        st.markdown('<div class="insights-title">Quick Actions</div>', unsafe_allow_html=True)
    # Quick action buttons (interactive)
        if st.button('Export Filtered CSV'):
            csv = df.to_csv(index=False)
            st.download_button('Download CSV', data=csv, file_name='exported_data.csv', mime='text/csv')
        if st.button('Refresh Data'):
            if hasattr(st, 'experimental_rerun'):
                try:
                    st.experimental_rerun()
                except Exception:
                    st.info('Please refresh the page manually (F5).')
            else:
                st.info('Please refresh the page manually (F5).')
        st.markdown('</div>', unsafe_allow_html=True)


def display_raw_data_tab(df):
    """Display raw data tab."""
    st.subheader("📋 Raw Data")
    
    # Data preview
    st.write("**Data Preview:**")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Data info
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Data Info:**")
        st.write(f"- Shape: {df.shape}")
        st.write(f"- Columns: {list(df.columns)}")
        st.write(f"- Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    with col2:
        st.write("**Platform Distribution:**")
        if 'platform' in df.columns:
            platform_counts = df['platform'].value_counts()
            fig = px.pie(values=platform_counts.values, names=platform_counts.index, title="Platform Distribution")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No platform information available")
    
    # Download button
    csv = df.to_csv(index=False, encoding='utf-8')
    st.download_button(
        label="📥 Download Raw Data",
        data=csv,
        file_name=f"raw_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )


def display_processing_tab(df):
    """Display data processing tab."""
    st.subheader("🔧 Data Processing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Processing Options:**")
        clean_text = st.checkbox("Clean Text", value=True)
        remove_duplicates = st.checkbox("Remove Duplicates", value=True)
        detect_regions = st.checkbox("Detect Regions", value=True)
        extract_features = st.checkbox("Extract Features", value=True)
    
    with col2:
        st.write("**Text Columns to Process:**")
        text_columns = st.multiselect(
            "Select columns:",
            [col for col in df.columns if df[col].dtype == 'object'],
            default=['title', 'content'] if 'title' in df.columns and 'content' in df.columns else []
        )
    
    if st.button("🔄 Process Data", type="primary"):
        with st.spinner("Processing data..."):
            try:
                preprocessor = DataPreprocessor()
                processed_df = preprocessor.preprocess_pipeline(
                    df=df,
                    text_columns=text_columns,
                    remove_duplicates=remove_duplicates,
                    detect_regions=detect_regions,
                    extract_features=extract_features
                )
                
                st.session_state.processed_data = processed_df
                st.success(f"✅ Processing completed! {len(processed_df)} records processed.")
                
                # Display processed data
                st.write("**Processed Data Preview:**")
                st.dataframe(processed_df.head(), use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error processing data: {str(e)}")


def display_summarization_tab(df):
    """Display summarization tab."""
    st.subheader("📝 Text Summarization")
    
    # Model selection
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox(
            "Select Summarization Model:",
            ["t5", "bart", "pegasus", "extractive", "ensemble"]
        )
    
    with col2:
        max_length = st.slider("Max Summary Length:", 50, 300, 150)
    
    # Text selection
    text_column = st.selectbox(
        "Select Text Column:",
        [col for col in df.columns if df[col].dtype == 'object' and col in ['content', 'title', 'description']]
    )
    
    # Summarization options
    col1, col2 = st.columns(2)
    
    with col1:
        summarize_all = st.checkbox("Summarize All Texts", value=False)
        num_texts = st.number_input("Number of texts to summarize:", min_value=1, max_value=len(df), value=min(5, len(df)))
    
    with col2:
        show_original = st.checkbox("Show Original Text", value=True)
        show_summary = st.checkbox("Show Summary", value=True)
    
    if st.button("📝 Generate Summaries", type="primary"):
        # --- START TIMING ---
        start_time = time.time()
        
        with st.spinner("Generating summaries..."):
            try:
                if model_type == "ensemble":
                    summarizer = MultiModelSummarizer()
                else:
                    summarizer = TextSummarizer(model_type)
                
                # Select texts to summarize
                if summarize_all:
                    texts_to_summarize = df[text_column].head(num_texts).tolist()
                else:
                    texts_to_summarize = df[text_column].head(num_texts).tolist()
                
                # Generate summaries
                summaries = []
                for i, text in enumerate(texts_to_summarize):
                    if pd.notna(text) and str(text).strip():
                        if model_type == "ensemble":
                            summary = summarizer.get_best_summary(str(text), max_length)
                        else:
                            # Your original model call
                            summary = summarizer.summarize_text(str(text), max_length)
                        summaries.append(summary)
                    else:
                        summaries.append("")
                        
                    # Progress bar
                    progress = (i + 1) / len(texts_to_summarize)
                    st.progress(progress)
                
                # --- STOP TIMING ---
                end_time = time.time()
                total_time = end_time - start_time
                num_summaries = len([s for s in summaries if s])
                
                # Store summaries
                st.session_state.summaries = {
                    'texts': texts_to_summarize,
                    'summaries': summaries,
                    'model': model_type
                }
                
                st.success(f"✅ Generated {num_summaries} summaries!")
                if num_summaries > 0:
                    st.info(f"Total processing time: **{total_time:.2f} seconds** (Avg: **{total_time / num_summaries:.3f} s/text**).")
                
                # Display results
                for i, (text, summary) in enumerate(zip(texts_to_summarize, summaries)):
                    if summary:
                        st.write(f"**Text {i+1}:**")
                        if show_original:
                            st.write(f"**Original:** {text[:200]}...")
                        if show_summary:
                            st.write(f"**Summary:** {summary}")
                        st.divider()
                
            except Exception as e:
                st.error(f"❌ Error generating summaries: {str(e)}")

def display_summarization_tab(df):
    """Display summarization tab."""
    st.subheader("📝 Text Summarization")
    
    # Model selection
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox(
            "Select Summarization Model:",
            ["t5", "bart", "pegasus", "extractive", "ensemble"]
        )
    
    with col2:
        max_length = st.slider("Max Summary Length:", 50, 300, 150)
    
    # Text selection
    text_column = st.selectbox(
        "Select Text Column:",
        [col for col in df.columns if df[col].dtype == 'object' and col in ['content', 'title', 'description']]
    )
    
    # Summarization options
    col1, col2 = st.columns(2)
    
    with col1:
        summarize_all = st.checkbox("Summarize All Texts", value=False)
        num_texts = st.number_input("Number of texts to summarize:", min_value=1, max_value=len(df), value=min(5, len(df)))
    
    with col2:
        show_original = st.checkbox("Show Original Text", value=True)
        show_summary = st.checkbox("Show Summary", value=True)
    
    if st.button("📝 Generate Summaries", type="primary"):
        # --- START TIMING ---
        start_time = time.time()
        
        with st.spinner("Generating summaries..."):
            try:
                if model_type == "ensemble":
                    summarizer = MultiModelSummarizer()
                else:
                    summarizer = TextSummarizer(model_type)
                
                # Select texts to summarize
                if summarize_all:
                    texts_to_summarize = df[text_column].head(num_texts).tolist()
                else:
                    texts_to_summarize = df[text_column].head(num_texts).tolist()
                
                # Generate summaries
                summaries = []
                for i, text in enumerate(texts_to_summarize):
                    if pd.notna(text) and str(text).strip():
                        if model_type == "ensemble":
                            summary = summarizer.get_best_summary(str(text), max_length)
                        else:
                            # Your original model call
                            summary = summarizer.summarize_text(str(text), max_length)
                        summaries.append(summary)
                    else:
                        summaries.append("")
                        
                    # Progress bar
                    progress = (i + 1) / len(texts_to_summarize)
                    st.progress(progress)
                
                # --- STOP TIMING ---
                end_time = time.time()
                total_time = end_time - start_time
                num_summaries = len([s for s in summaries if s])
                
                # Store summaries
                st.session_state.summaries = {
                    'texts': texts_to_summarize,
                    'summaries': summaries,
                    'model': model_type
                }
                
                st.success(f"✅ Generated {num_summaries} summaries!")
                if num_summaries > 0:
                    st.info(f"Total processing time: **{total_time:.2f} seconds** (Avg: **{total_time / num_summaries:.3f} s/text**).")
                
                # Display results
                for i, (text, summary) in enumerate(zip(texts_to_summarize, summaries)):
                    if summary:
                        st.write(f"**Text {i+1}:**")
                        if show_original:
                            st.write(f"**Original:** {text[:200]}...")
                        if show_summary:
                            st.write(f"**Summary:** {summary}")
                        st.divider()
                
            except Exception as e:
                st.error(f"❌ Error generating summaries: {str(e)}")
                
def display_analytics_tab(df):
    """Display analytics tab."""
    st.subheader("📈 Analytics & Insights")
    
    # Time series analysis
    if 'published_date' in df.columns:
        st.write("**📅 Timeline Analysis**")
        
        # Convert to datetime
        df['published_date'] = pd.to_datetime(df['published_date'], errors='coerce')
        df_time = df.dropna(subset=['published_date'])
        
        if not df_time.empty:
            # Daily counts
            daily_counts = df_time.groupby(df_time['published_date'].dt.date).size().reset_index()
            daily_counts.columns = ['date', 'count']
            
            fig = px.line(daily_counts, x='date', y='count', title="Daily News Count")
            st.plotly_chart(fig, use_container_width=True)
    
    # Platform analysis
    if 'platform' in df.columns:
        st.write("**🌐 Platform Analysis**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            platform_counts = df['platform'].value_counts()
            fig = px.bar(x=platform_counts.index, y=platform_counts.values, title="Platform Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Platform over time
            if 'published_date' in df.columns and not df_time.empty:
                platform_time = df_time.groupby(['platform', df_time['published_date'].dt.date]).size().reset_index()
                platform_time.columns = ['platform', 'date', 'count']
                
                fig = px.line(platform_time, x='date', y='count', color='platform', title="Platform Activity Over Time")
                st.plotly_chart(fig, use_container_width=True)
    
    # Content analysis
    if 'content' in df.columns:
        st.write("**📝 Content Analysis**")
        
        # Word count analysis
        df['word_count'] = df['content'].str.split().str.len()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df, x='word_count', title="Content Length Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Average word count by platform
            if 'platform' in df.columns:
                avg_word_count = df.groupby('platform')['word_count'].mean().reset_index()
                fig = px.bar(avg_word_count, x='platform', y='word_count', title="Average Content Length by Platform")
                st.plotly_chart(fig, use_container_width=True)
    
    # Export processed data
    if not st.session_state.processed_data.empty:
        st.write("**📥 Export Processed Data**")
        
        csv = st.session_state.processed_data.to_csv(index=False, encoding='utf-8')
        st.download_button(
            label="📥 Download Processed Data",
            data=csv,
            file_name=f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )


def display_classification_tab(df):
    """Display classification tab for sentiment/topic classification."""
    st.subheader("🏷️ Classification")

    # Prefer processed data if available
    data_to_use = st.session_state.processed_data if not st.session_state.processed_data.empty else df

    if data_to_use.empty:
        st.warning("No data available for classification. Please fetch or process data first.")
        return

    # Choose classifier
    col1, col2 = st.columns(2)
    with col1:
        clf_type = st.selectbox("Select classifier:", ["sentiment", "topic"])

    with col2:
        text_column = st.selectbox("Text column:", [c for c in data_to_use.columns if data_to_use[c].dtype == 'object'], index=0)

    # Topic options
    topics = None
    if clf_type == 'topic':
        topics_input = st.text_input("Candidate topics (comma-separated)", value="technology,business,science,health")
        topics = [t.strip() for t in topics_input.split(',') if t.strip()]

    if st.button("Run Classification"):
        with st.spinner("Classifying texts..."):
            try:
                if clf_type == 'sentiment':
                    classifier = SentimentClassifier()
                else:
                    classifier = TopicClassifier(topics=topics)

                # Perform classification
                df_classified = classifier.classify_dataframe(data_to_use.copy(), text_column)

                # Store in session state
                st.session_state.classifications = df_classified

                st.success(f"✅ Classification completed: {len(df_classified)} records")

                # Show distribution
                if 'predicted_label' in df_classified.columns:
                    st.write("**Prediction Distribution:**")
                    dist = df_classified['predicted_label'].value_counts()
                    fig = px.bar(x=dist.index, y=dist.values, title="Prediction Distribution")
                    st.plotly_chart(fig, use_container_width=True)

                # Show sample predictions
                st.markdown("**Sample Predictions:**")
                st.dataframe(df_classified[[text_column, 'predicted_label', 'prediction_confidence']].head(10), use_container_width=True)

                # Download
                csv_out = df_classified.to_csv(index=False, encoding='utf-8')
                st.download_button("📥 Download Classified Data", data=csv_out, file_name=f"classified_{clf_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

            except Exception as e:
                st.error(f"❌ Error during classification: {str(e)}")


if __name__ == "__main__":
    main()
