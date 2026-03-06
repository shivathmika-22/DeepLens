# 🔍 DeepLens - News Intelligence Platform

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.25+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

DeepLens is a comprehensive news intelligence platform that aggregates, processes, and analyzes news content from multiple sources using advanced NLP techniques. It provides a production-ready pipeline for news collection, text cleaning, summarization, classification, and visualization.

## 🌟 Features

### 📊 Multi-Source Data Aggregation
- **Twitter**: Real-time tweets using snscrape
- **YouTube**: Video metadata via Google API
- **GNews**: News articles from GNews API
- **Facebook**: Public page posts via RSS feeds
- **LinkedIn**: Professional content via Playwright

### 🧹 Advanced Text Processing
- **Text Cleaning**: Remove emojis, HTML tags, special characters
- **Duplicate Removal**: Smart deduplication across platforms
- **Region Detection**: NER-based location extraction
- **Feature Extraction**: Word counts, sentiment indicators, engagement metrics

### 🤖 AI-Powered Analysis
- **Summarization**: T5, BART, Pegasus, and extractive methods
- **Topic Classification**: Zero-shot classification with BART
- **Sentiment Analysis**: RoBERTa-based sentiment detection
- **Ensemble Methods**: Multi-model approach for better results

### 📈 Interactive Visualization
- **Streamlit UI**: Modern, responsive web interface
- **Real-time Analytics**: Platform distribution, timeline analysis
- **Export Capabilities**: CSV, JSON, and processed data downloads
- **Custom Dashboards**: Configurable charts and metrics

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Git
- API Keys (see Configuration section)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/deeplens.git
   cd deeplens
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirement.txt
   ```

4. **Install spaCy model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Install Playwright browsers**
   ```bash
   playwright install
   ```

### Configuration

1. **Set up API keys** in `config/api_keys.py`:
   ```python
   YOUTUBE_API_KEY = "your_youtube_api_key"
   GNEWS_API_KEY = "your_gnews_api_key"
   # Add other API keys as needed
   ```

2. **Create data directories**
   ```bash
   mkdir -p data/raw data/cleaned
   ```

### Running the Application

1. **Start the Streamlit app**
   ```bash
   streamlit run app/app.py
   ```

2. **Access the web interface**
   - Open your browser to `http://localhost:8501`
   - Use the sidebar to configure data sources and parameters
   - Click "Fetch Data" to start the pipeline

## 📁 Project Structure

```
deeplens/
├── app/                    # Streamlit web application
│   ├── __init__.py
│   └── app.py             # Main Streamlit app
├── config/                # Configuration files
│   ├── __init__.py
│   └── api_keys.py        # API keys and credentials
├── data/                  # Data storage
│   ├── raw/              # Raw scraped data
│   └── cleaned/          # Processed data
├── models/               # NLP models
│   ├── __init__.py
│   ├── classifier.py     # Text classification models
│   └── summarizer.py     # Text summarization models
├── pipeline/             # Data processing pipeline
│   ├── __init__.py
│   ├── fetch_news.py     # Multi-source data fetching
│   └── preprocess.py     # Data preprocessing
├── scripts/              # Individual scraper scripts
│   ├── facebook_scraper.py
│   ├── gnews_scrapper.py
│   ├── linkdin_scrapper.py
│   ├── twitter_scraper.py
│   └── youtube_scrapper.py
├── utils/                # Utility modules
│   ├── __init__.py
│   ├── base_scraper.py   # Base scraper class
│   └── cleaner.py        # Text cleaning utilities
├── notebooks/            # Jupyter notebooks for analysis
├── requirement.txt       # Python dependencies
└── README.md            # This file
```

## 🔧 Usage

### Command Line Interface

#### Individual Scrapers

```bash
# Twitter scraper
python scripts/twitter_scraper.py --query "AI healthcare" --max_results 100

# YouTube scraper
python scripts/youtube_scrapper.py --query "machine learning" --max_results 50

# GNews scraper
python scripts/gnews_scrapper.py --query "climate change" --max_results 100

# Facebook scraper
python scripts/facebook_scraper.py --page "techcrunch" --limit 20

# LinkedIn scraper
python scripts/linkdin_scrapper.py --query "data science" --max_results 30
```

#### Pipeline Operations

```bash
# Fetch from all platforms
python pipeline/fetch_news.py --query "AI trends" --max_results 50

# Preprocess data
python pipeline/preprocess.py --input data/raw/combined_data.csv --output data/cleaned/processed_data.csv

# Summarize texts
python models/summarizer.py --input data/cleaned/processed_data.csv --model t5 --max_length 150

# Classify content
python models/classifier.py --input data/cleaned/processed_data.csv --type sentiment
```

### Python API

```python
from pipeline.fetch_news import NewsFetcher
from pipeline.preprocess import DataPreprocessor
from models.summarizer import TextSummarizer
from models.classifier import SentimentClassifier

# Fetch data
fetcher = NewsFetcher()
df = fetcher.fetch_from_all_platforms(
    query="artificial intelligence",
    max_results_per_platform=50,
    platforms=["twitter", "gnews", "youtube"]
)

# Preprocess
preprocessor = DataPreprocessor()
df_processed = preprocessor.preprocess_pipeline(df)

# Summarize
summarizer = TextSummarizer("t5")
df_summarized = summarizer.summarize_dataframe(df_processed, "content")

# Classify
classifier = SentimentClassifier()
df_classified = classifier.classify_dataframe(df_summarized, "content")
```

## 🎯 Phase Roadmap

### ✅ Phase 1: Core MVP (Weeks 1–2)
- [x] Multi-source data collection
- [x] Text preprocessing and cleaning
- [x] Basic summarization (T5, BART)
- [x] Streamlit UI with input/output
- [x] CSV export functionality

### 🎯 Phase 2: NLP + Personalization (Weeks 3–4)
- [ ] Advanced summarization (Pegasus, ensemble methods)
- [ ] Fine-tuned classification models
- [ ] Personalization engine (TF-IDF, LightFM)
- [ ] Database integration (MongoDB/SQLite)
- [ ] Evaluation metrics (ROUGE scores)

### 🚀 Phase 3: Advanced Extensions (Weeks 5–6)
- [ ] Weekly digest generator
- [ ] Trend detection (BERTopic, LDA)
- [ ] Semantic search with embeddings
- [ ] Mobile-responsive UI
- [ ] Audio summaries (TTS)

## 🔧 Configuration

### API Keys Setup

1. **YouTube API**
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Enable YouTube Data API v3
   - Create credentials and get API key

2. **GNews API**
   - Sign up at [GNews.io](https://gnews.io/)
   - Get your API key from dashboard

3. **Twitter (Optional)**
   - Apply for Twitter API access
   - Get Bearer Token for enhanced features

### Environment Variables

```bash
export YOUTUBE_API_KEY="your_youtube_api_key"
export GNEWS_API_KEY="your_gnews_api_key"
export TWITTER_BEARER_TOKEN="your_twitter_token"
```

## 📊 Data Schema

### Standardized Output Format

```csv
platform,query,title,content,published_date,author,url
twitter,AI healthcare,New AI breakthrough,Content text...,2024-01-15T10:30:00Z,@username,https://twitter.com/...
youtube,AI healthcare,AI in Medicine,Video description...,2024-01-15T09:15:00Z,Channel Name,https://youtube.com/...
```

### Processed Data Features

- **Text Features**: Cleaned content, word count, character count
- **Temporal Features**: Day of week, hour, weekend indicator
- **Platform Features**: Platform encoding, source type
- **Engagement Features**: Like count, retweet count, view count
- **NLP Features**: Sentiment score, topic classification, region detection

## 🧪 Testing

```bash
# Run individual scraper tests
python -m pytest tests/test_scrapers.py

# Run pipeline tests
python -m pytest tests/test_pipeline.py

# Run model tests
python -m pytest tests/test_models.py

# Run all tests
python -m pytest
```

## 📈 Performance

### Benchmarks

- **Data Fetching**: ~100 records/minute per platform
- **Text Cleaning**: ~1000 texts/second
- **Summarization**: ~10 texts/second (T5-small)
- **Classification**: ~50 texts/second (RoBERTa)

### Optimization Tips

1. **Batch Processing**: Process texts in batches for better performance
2. **Caching**: Use Streamlit caching for repeated operations
3. **Model Selection**: Choose appropriate models based on speed/quality trade-offs
4. **Parallel Processing**: Use multiprocessing for large datasets

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run linting
black .
flake8 .

# Run type checking
mypy .
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for transformer models
- [Streamlit](https://streamlit.io/) for the web framework
- [spaCy](https://spacy.io/) for NLP processing
- [Plotly](https://plotly.com/) for visualizations

## 📞 Support

- **Documentation**: [Wiki](https://github.com/yourusername/deeplens/wiki)
- **Issues**: [GitHub Issues](https://github.com/yourusername/deeplens/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/deeplens/discussions)
- **Email**: support@deeplens.ai

## 🔮 Future Enhancements

- [ ] Real-time streaming data processing
- [ ] Multi-language support
- [ ] Advanced visualization dashboards
- [ ] API endpoints for external integration
- [ ] Mobile app development
- [ ] Cloud deployment (AWS, GCP, Azure)
- [ ] Advanced analytics and reporting
- [ ] Integration with news APIs (NewsAPI, Guardian API)

---

**Made with ❤️ by the DeepLens Team**
