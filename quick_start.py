#!/usr/bin/env python3
"""
DeepLens Quick Start Script
Demonstrates basic functionality with sample data.
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_sample_data():
    """Create sample data for demonstration."""
    print("📊 Creating sample data...")
    
    sample_data = {
        'platform': ['twitter', 'youtube', 'gnews', 'facebook', 'linkedin'] * 3,
        'query': ['AI healthcare'] * 15,
        'title': [
            'AI Revolution in Healthcare: New Breakthrough',
            'Machine Learning in Medical Diagnosis',
            'Artificial Intelligence Transforming Medicine',
            'Healthcare AI: Latest Developments',
            'AI-Powered Medical Devices',
            'Deep Learning for Drug Discovery',
            'AI in Radiology: Improving Accuracy',
            'Healthcare Automation with AI',
            'AI Ethics in Medical Practice',
            'Future of AI in Healthcare',
            'AI-Assisted Surgery Techniques',
            'Healthcare Data Analytics with AI',
            'AI for Personalized Medicine',
            'Medical AI: Challenges and Opportunities',
            'AI in Telemedicine and Remote Care'
        ],
        'content': [
            'Artificial Intelligence is revolutionizing healthcare with new diagnostic tools that can detect diseases earlier and more accurately than ever before. These AI systems analyze medical images, patient data, and symptoms to provide doctors with valuable insights.',
            'Machine learning algorithms are being used to develop personalized treatment plans for patients. By analyzing vast amounts of medical data, AI can identify patterns and predict patient outcomes with remarkable accuracy.',
            'The integration of AI in healthcare is transforming how medical professionals approach diagnosis and treatment. From robotic surgery to virtual health assistants, technology is enhancing patient care.',
            'Healthcare organizations are adopting AI solutions to improve efficiency and reduce costs. These technologies help streamline administrative tasks and enhance clinical decision-making processes.',
            'AI-powered medical devices are becoming increasingly sophisticated, offering real-time monitoring and analysis of patient vital signs. These innovations are improving patient outcomes and reducing hospital readmissions.',
            'Deep learning models are accelerating drug discovery by identifying potential compounds and predicting their effectiveness. This approach is reducing the time and cost of bringing new medications to market.',
            'AI in radiology is improving the accuracy of medical imaging interpretation. Machine learning algorithms can detect abnormalities in X-rays, MRIs, and CT scans with high precision.',
            'Healthcare automation with AI is streamlining various processes, from appointment scheduling to insurance claims processing. This technology is reducing administrative burden on medical staff.',
            'The ethical implications of AI in healthcare are being carefully considered. Issues such as patient privacy, algorithmic bias, and the role of human judgment in medical decisions are being addressed.',
            'The future of AI in healthcare looks promising, with continued advancements in machine learning, natural language processing, and computer vision. These technologies will further enhance patient care.',
            'AI-assisted surgery techniques are improving precision and reducing recovery times. Robotic systems guided by AI algorithms can perform complex procedures with minimal invasiveness.',
            'Healthcare data analytics with AI is providing valuable insights into population health trends and treatment effectiveness. This information helps healthcare providers make informed decisions.',
            'AI for personalized medicine is tailoring treatments to individual patients based on their genetic makeup, lifestyle, and medical history. This approach is improving treatment outcomes.',
            'Medical AI presents both challenges and opportunities. While it offers significant benefits, concerns about data security, regulatory compliance, and human oversight remain important considerations.',
            'AI in telemedicine and remote care is expanding access to healthcare services. Virtual consultations and remote monitoring are becoming more sophisticated and effective.'
        ],
        'published_date': [
            '2024-01-15T10:30:00Z',
            '2024-01-15T09:15:00Z',
            '2024-01-15T08:45:00Z',
            '2024-01-15T07:20:00Z',
            '2024-01-15T06:10:00Z',
            '2024-01-14T15:30:00Z',
            '2024-01-14T14:15:00Z',
            '2024-01-14T13:45:00Z',
            '2024-01-14T12:20:00Z',
            '2024-01-14T11:10:00Z',
            '2024-01-13T16:30:00Z',
            '2024-01-13T15:15:00Z',
            '2024-01-13T14:45:00Z',
            '2024-01-13T13:20:00Z',
            '2024-01-13T12:10:00Z'
        ],
        'author': [
            '@healthtech', 'MedTech Channel', 'AI Research News', 'Healthcare Today', 'DataMed Expert',
            '@aihealth', 'Medical AI Channel', 'Health Innovation', 'Tech Medicine', 'AI Healthcare',
            '@medtech', 'Surgical AI Channel', 'Health Analytics', 'Medical Data', 'AI Medicine'
        ],
        'url': [
            'https://twitter.com/healthtech/status/1234567890',
            'https://youtube.com/watch?v=abc123',
            'https://gnews.io/article/123',
            'https://facebook.com/healthcare/posts/123',
            'https://linkedin.com/posts/123',
            'https://twitter.com/aihealth/status/1234567891',
            'https://youtube.com/watch?v=def456',
            'https://gnews.io/article/124',
            'https://facebook.com/healthcare/posts/124',
            'https://linkedin.com/posts/124',
            'https://twitter.com/medtech/status/1234567892',
            'https://youtube.com/watch?v=ghi789',
            'https://gnews.io/article/125',
            'https://facebook.com/healthcare/posts/125',
            'https://linkedin.com/posts/125'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Save sample data
    os.makedirs('data/raw', exist_ok=True)
    sample_file = 'data/raw/sample_data.csv'
    df.to_csv(sample_file, index=False, encoding='utf-8')
    
    print(f"✅ Sample data created: {sample_file}")
    print(f"   Records: {len(df)}")
    print(f"   Platforms: {df['platform'].nunique()}")
    
    return df

def demonstrate_preprocessing(df):
    """Demonstrate data preprocessing."""
    print("\n🔧 Demonstrating data preprocessing...")
    
    try:
        from pipeline.preprocess import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        df_processed = preprocessor.preprocess_pipeline(
            df=df,
            text_columns=['title', 'content'],
            remove_duplicates=True,
            detect_regions=True,
            extract_features=True
        )
        
        # Save processed data
        os.makedirs('data/cleaned', exist_ok=True)
        processed_file = 'data/cleaned/sample_processed.csv'
        df_processed.to_csv(processed_file, index=False, encoding='utf-8')
        
        print(f"✅ Data preprocessing completed: {processed_file}")
        print(f"   Original records: {len(df)}")
        print(f"   Processed records: {len(df_processed)}")
        
        # Show some features
        if 'word_count' in df_processed.columns:
            avg_words = df_processed['word_count'].mean()
            print(f"   Average word count: {avg_words:.1f}")
        
        if 'primary_region' in df_processed.columns:
            regions = df_processed['primary_region'].value_counts()
            print(f"   Top regions: {regions.head(3).to_dict()}")
        
        return df_processed
        
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        return df

def demonstrate_summarization(df):
    """Demonstrate text summarization."""
    print("\n📝 Demonstrating text summarization...")
    
    try:
        from models.summarizer import TextSummarizer
        
        # Use extractive summarization for demo (faster)
        summarizer = TextSummarizer("extractive")
        
        # Summarize first few texts
        sample_texts = df['content'].head(3).tolist()
        summaries = []
        
        for i, text in enumerate(sample_texts):
            summary = summarizer.summarize_text(str(text), max_length=100)
            summaries.append(summary)
            print(f"  Text {i+1} summary: {summary[:100]}...")
        
        print("✅ Summarization demonstration completed")
        return summaries
        
    except Exception as e:
        print(f"❌ Summarization failed: {e}")
        return []

def demonstrate_classification(df):
    """Demonstrate text classification."""
    print("\n🏷️ Demonstrating text classification...")
    
    try:
        from models.classifier import SentimentClassifier, TopicClassifier
        
        # Test sentiment classification
        sentiment_classifier = SentimentClassifier()
        sample_text = df['content'].iloc[0]
        
        sentiment_result = sentiment_classifier.predict_sentiment(sample_text)
        print(f"  Sentiment analysis: {sentiment_result}")
        
        # Test topic classification
        topic_classifier = TopicClassifier()
        topic_result = topic_classifier.predict(sample_text)
        print(f"  Topic classification: {topic_result}")
        
        print("✅ Classification demonstration completed")
        return True
        
    except Exception as e:
        print(f"❌ Classification failed: {e}")
        return False

def main():
    """Main demonstration function."""
    print("🚀 DeepLens Quick Start Demonstration")
    print("=" * 50)
    
    # Create sample data
    df = create_sample_data()
    
    # Demonstrate preprocessing
    df_processed = demonstrate_preprocessing(df)
    
    # Demonstrate summarization
    summaries = demonstrate_summarization(df_processed)
    
    # Demonstrate classification
    classification_success = demonstrate_classification(df_processed)
    
    print("\n" + "=" * 50)
    print("🎉 Quick Start Demonstration Completed!")
    print("\nGenerated files:")
    print("  📄 data/raw/sample_data.csv - Sample raw data")
    print("  📄 data/cleaned/sample_processed.csv - Processed data")
    
    print("\nNext steps:")
    print("1. Run the Streamlit app: streamlit run app/app.py")
    print("2. Or run the full pipeline: python run_pipeline.py --query 'AI healthcare' --platforms gnews twitter")
    print("3. Check the generated CSV files in data/ directory")
    
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    main()
