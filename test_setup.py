#!/usr/bin/env python3
"""
Test script to verify DeepLens setup and basic functionality.
"""

import sys
import os
import importlib
import traceback

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all required modules can be imported."""
    print("🔍 Testing imports...")
    
    required_modules = [
        'pandas',
        'numpy',
        'requests',
        'streamlit',
        'plotly',
        'sklearn',
        'transformers',
        'spacy',
        'snscrape',
        'feedparser',
        'playwright'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n⚠️  Failed to import: {', '.join(failed_imports)}")
        print("Please install missing dependencies with: pip install -r requirement.txt")
        return False
    
    print("✅ All imports successful!")
    return True

def test_deeplens_modules():
    """Test if DeepLens modules can be imported."""
    print("\n🔍 Testing DeepLens modules...")
    
    deeplens_modules = [
        'utils.cleaner',
        'utils.base_scraper',
        'pipeline.fetch_news',
        'pipeline.preprocess',
        'models.summarizer',
        'models.classifier'
    ]
    
    failed_imports = []
    
    for module in deeplens_modules:
        try:
            importlib.import_module(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n⚠️  Failed to import: {', '.join(failed_imports)}")
        return False
    
    print("✅ All DeepLens modules imported successfully!")
    return True

def test_basic_functionality():
    """Test basic functionality of key components."""
    print("\n🔍 Testing basic functionality...")
    
    try:
        # Test text cleaner
        from utils.cleaner import TextCleaner
        cleaner = TextCleaner()
        test_text = "Hello world! 🌍 This is a test with <b>HTML</b> and https://example.com"
        cleaned = cleaner.clean_text(test_text)
        print(f"  ✅ Text cleaner: '{cleaned[:50]}...'")
        
        # Test data preprocessor
        from pipeline.preprocess import DataPreprocessor
        preprocessor = DataPreprocessor()
        print("  ✅ Data preprocessor initialized")
        
        # Test news fetcher
        from pipeline.fetch_news import NewsFetcher
        fetcher = NewsFetcher()
        print("  ✅ News fetcher initialized")
        
        # Test summarizer (without loading heavy models)
        from models.summarizer import TextSummarizer
        summarizer = TextSummarizer("extractive")
        print("  ✅ Text summarizer initialized")
        
        # Test classifier (without loading heavy models)
        from models.classifier import SentimentClassifier
        classifier = SentimentClassifier()
        print("  ✅ Sentiment classifier initialized")
        
        print("✅ Basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_config():
    """Test configuration files."""
    print("\n🔍 Testing configuration...")
    
    # Check if api_keys.py exists
    api_keys_path = "config/api_keys.py"
    if os.path.exists(api_keys_path):
        print("  ✅ API keys file found")
        try:
            from config.api_keys import YOUTUBE_API_KEY, GNEWS_API_KEY
            if YOUTUBE_API_KEY and YOUTUBE_API_KEY != "your_youtube_api_key_here":
                print("  ✅ YouTube API key configured")
            else:
                print("  ⚠️  YouTube API key not configured")
            
            if GNEWS_API_KEY and GNEWS_API_KEY != "your_gnews_api_key_here":
                print("  ✅ GNews API key configured")
            else:
                print("  ⚠️  GNews API key not configured")
        except ImportError as e:
            print(f"  ❌ Error importing API keys: {e}")
    else:
        print("  ⚠️  API keys file not found. Copy config/api_keys_template.py to config/api_keys.py")
    
    # Check data directories
    data_dirs = ["data", "data/raw", "data/cleaned"]
    for dir_path in data_dirs:
        if os.path.exists(dir_path):
            print(f"  ✅ Directory exists: {dir_path}")
        else:
            print(f"  ⚠️  Directory missing: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
            print(f"  ✅ Created directory: {dir_path}")

def main():
    """Main test function."""
    print("🚀 DeepLens Setup Test")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Run tests
    all_tests_passed &= test_imports()
    all_tests_passed &= test_deeplens_modules()
    all_tests_passed &= test_basic_functionality()
    test_config()  # This doesn't affect the overall result
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 All tests passed! DeepLens is ready to use.")
        print("\nNext steps:")
        print("1. Configure your API keys in config/api_keys.py")
        print("2. Run: streamlit run app/app.py")
        print("3. Or run: python run_pipeline.py --query 'your query' --platforms gnews twitter")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Install dependencies: pip install -r requirement.txt")
        print("2. Install spaCy model: python -m spacy download en_core_web_sm")
        print("3. Install Playwright: playwright install")
        sys.exit(1)

if __name__ == "__main__":
    main()
