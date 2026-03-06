"""
Text summarization models for DeepLens.
Supports multiple summarization approaches including T5, BART, and extractive methods.
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TextSummarizer:
    """Main class for text summarization."""
    
    def __init__(self, model_type: str = "t5", model_name: Optional[str] = None):
        self.model_type = model_type.lower()
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.extractive_models = None # Initialize to None
        self.logger = self._setup_logger()
        self._load_model()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for the summarizer."""
        logger = logging.getLogger("text_summarizer")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_model(self):
        """Load the specified summarization model."""
        try:
            if self.model_type == "t5":
                self._load_t5_model()
            elif self.model_type == "bart":
                self._load_bart_model()
            elif self.model_type == "pegasus":
                self._load_pegasus_model()
            elif self.model_type == "extractive":
                self._load_extractive_model()
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            self.logger.info(f"✓ {self.model_type.upper()} model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            self.logger.info("Falling back to extractive summarization")
            self.model_type = "extractive"
            self._load_extractive_model()
    
    def _load_t5_model(self):
        """Load T5 model for summarization."""
        try:
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            
            model_name = self.model_name or "t5-small"
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            
        except ImportError:
            raise ImportError("transformers library not installed. Install with: pip install transformers")
    
    def _load_bart_model(self):
        """Load BART model for summarization."""
        try:
            from transformers import BartForConditionalGeneration, BartTokenizer
            
            model_name = self.model_name or "facebook/bart-large-cnn"
            self.tokenizer = BartTokenizer.from_pretrained(model_name)
            self.model = BartForConditionalGeneration.from_pretrained(model_name)
            
        except ImportError:
            raise ImportError("transformers library not installed. Install with: pip install transformers")
    
    def _load_pegasus_model(self):
        """Load Pegasus model for summarization."""
        try:
            from transformers import PegasusForConditionalGeneration, PegasusTokenizer
            
            model_name = self.model_name or "google/pegasus-xsum"
            self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
            self.model = PegasusForConditionalGeneration.from_pretrained(model_name)
            
        except ImportError:
            raise ImportError("transformers library not installed. Install with: pip install transformers")
    
    def _load_extractive_model(self):
        """
        Load extractive summarization models using sumy.
        Initializes LSA Summarizer with a stemmer for better performance.
        """
        try:
            # Import necessary sumy components
            from sumy.summarizers.lex_rank import LexRankSummarizer
            from sumy.summarizers.lsa import LsaSummarizer
            from sumy.summarizers.text_rank import TextRankSummarizer
            from sumy.nlp.stemmers import Stemmer
            from sumy.utils import get_stop_words 

            LANGUAGE = "english"
            stemmer = Stemmer(LANGUAGE)
            
            # Initialize LSA with stemmer and stop words
            lsa_summarizer = LsaSummarizer(stemmer)
            lsa_summarizer.stop_words = get_stop_words(LANGUAGE)
            
            self.extractive_models = {
                'lexrank': LexRankSummarizer(stemmer), # LexRank can also benefit from stemmer
                'lsa': lsa_summarizer, 
                'textrank': TextRankSummarizer()
            }
            
        except ImportError:
            self.logger.warning("sumy library not installed. Using simple extractive method.")
            self.extractive_models = None
    
    def summarize_text(self, text: str, max_length: int = 150, min_length: int = 30) -> str:
        """
        Summarize a single text.
        
        Args:
            text: Input text to summarize
            max_length: Maximum length of summary
            min_length: Minimum length of summary
            
        Returns:
            Summarized text
        """
        if not text or len(text.strip()) < 50:
            return text
        
        try:
            if self.model_type == "extractive":
                return self._extractive_summarize(text, max_length)
            else:
                return self._abstractive_summarize(text, max_length, min_length)
                
        except Exception as e:
            self.logger.error(f"Error summarizing text: {str(e)}")
            return self._fallback_summarize(text, max_length)
    
    def _abstractive_summarize(self, text: str, max_length: int, min_length: int) -> str:
        """Abstractive summarization using transformer models."""
        try:
            import torch
            
            # Prepare input
            if self.model_type == "t5":
                input_text = f"summarize: {text}"
            else:
                input_text = text
            
            # Tokenize
            inputs = self.tokenizer.encode(
                input_text,
                return_tensors="pt",
                max_length=512,
                truncation=True
            )
            
            # Generate summary
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs,
                    max_length=max_length,
                    min_length=min_length,
                    length_penalty=2.0,
                    num_beams=4,
                    early_stopping=True
                )
            
            # Decode
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary.strip()
            
        except Exception as e:
            self.logger.error(f"Error in abstractive summarization: {str(e)}")
            return self._fallback_summarize(text, max_length)
    
    def _extractive_summarize(self, text: str, max_length: int) -> str:
        """Extractive summarization using sumy or simple method."""
        if self.extractive_models:
            try:
                from sumy.parsers.plaintext import PlaintextParser
                from sumy.nlp.tokenizers import Tokenizer
                
                # Parse text
                parser = PlaintextParser.from_string(text, Tokenizer("english"))
                
                # Use LexRank summarizer as a default extractive choice
                summarizer = self.extractive_models['lexrank']
                
                # Calculate number of sentences to extract (approximate 1/10 of max length tokens)
                # This is a heuristic, typically based on number of sentences not tokens
                sentence_count = max(3, max_length // 30) 
                
                sentences = summarizer(parser.document, sentence_count)
                
                summary = " ".join([str(sentence) for sentence in sentences])
                return summary
                
            except Exception as e:
                self.logger.error(f"Error in extractive summarization: {str(e)}")
        
        # Fallback to simple extractive method
        return self._fallback_summarize(text, max_length)
    
    def _fallback_summarize(self, text: str, max_length: int) -> str:
        """Simple fallback summarization method: takes the first few sentences."""
        # Simple splitting on '. ' to get sentences
        sentences = text.split('. ')
        
        if len(sentences) <= 3:
            return text
        
        # Take first few sentences that fit within max_length
        summary_sentences = []
        current_length = 0
        
        for sentence in sentences[:5]:  # Take max 5 sentences
            sentence_with_period = sentence + '. '
            if current_length + len(sentence_with_period) <= max_length:
                summary_sentences.append(sentence)
                current_length += len(sentence_with_period)
            else:
                break
        
        summary = '. '.join(summary_sentences)
        
        # Ensure the summary ends with a period if it contains sentences
        if summary and not summary.endswith('.'):
            summary += '.'
        
        # Final fallback if the text was too short or splitting failed
        if not summary:
            return text[:max_length] + "..." if len(text) > max_length else text
            
        return summary
    
    def summarize_dataframe(self, df: pd.DataFrame, text_column: str = 'content',
                             summary_column: str = 'summary', max_length: int = 150) -> pd.DataFrame:
        """
        Summarize text in a DataFrame column.
        
        Args:
            df: Input DataFrame
            text_column: Column containing text to summarize
            summary_column: Column name for summaries
            max_length: Maximum length of summaries
            
        Returns:
            DataFrame with summaries
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        self.logger.info(f"Summarizing {len(df)} texts...")
        
        df_summarized = df.copy()
        summaries = []
        
        for idx, text in enumerate(df[text_column]):
            if pd.notna(text) and str(text).strip():
                # Pass max_length to individual summarizer call
                summary = self.summarize_text(str(text), max_length=max_length) 
                summaries.append(summary)
            else:
                summaries.append("")
            
            if (idx + 1) % 10 == 0:
                self.logger.info(f"Processed {idx + 1}/{len(df)} texts")
        
        df_summarized[summary_column] = summaries
        
        self.logger.info("Summarization completed")
        return df_summarized
    
    def batch_summarize(self, texts: List[str], max_length: int = 150) -> List[str]:
        """
        Summarize a batch of texts.
        
        Args:
            texts: List of texts to summarize
            max_length: Maximum length of summaries
            
        Returns:
            List of summaries
        """
        summaries = []
        
        for i, text in enumerate(texts):
            if text and str(text).strip():
                # Pass max_length to individual summarizer call
                summary = self.summarize_text(str(text), max_length=max_length)
                summaries.append(summary)
            else:
                summaries.append("")
            
            if (i + 1) % 10 == 0:
                self.logger.info(f"Processed {i + 1}/{len(texts)} texts")
        
        return summaries


class MultiModelSummarizer:
    """Summarizer that can use multiple models and combine results."""
    
    def __init__(self, models: List[str] = None):
        self.models = models or ["t5", "extractive"]
        self.summarizers = {}
        self.logger = self._setup_logger()
        self._initialize_summarizers()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger."""
        logger = logging.getLogger("multi_model_summarizer")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_summarizers(self):
        """Initialize all summarizers."""
        for model_type in self.models:
            try:
                self.summarizers[model_type] = TextSummarizer(model_type)
                self.logger.info(f"✓ Initialized {model_type} summarizer")
            except Exception as e:
                self.logger.warning(f"Failed to initialize {model_type}: {str(e)}")
    
    def summarize_with_ensemble(self, text: str, max_length: int = 150) -> Dict[str, str]:
        """
        Summarize text using multiple models.
        
        Args:
            text: Input text
            max_length: Maximum length of summaries
            
        Returns:
            Dictionary with summaries from different models
        """
        summaries = {}
        
        for model_type, summarizer in self.summarizers.items():
            try:
                summary = summarizer.summarize_text(text, max_length)
                summaries[model_type] = summary
            except Exception as e:
                self.logger.error(f"Error with {model_type}: {str(e)}")
                summaries[model_type] = ""
        
        return summaries
    
    def get_best_summary(self, text: str, max_length: int = 150) -> str:
        """
        Get the best summary by trying multiple models.
        
        Args:
            text: Input text
            max_length: Maximum length of summary
            
        Returns:
            Best summary
        """
        summaries = self.summarize_with_ensemble(text, max_length)
        
        # Simple heuristic: prefer abstractive over extractive
        for model_type in ["t5", "bart", "pegasus", "extractive"]:
            if model_type in summaries and summaries[model_type]:
                return summaries[model_type]
        
        # Fallback: simple truncation
        return text[:max_length] + "..." if len(text) > max_length else text


def main():
    """Main function for testing the summarizer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Text summarization")
    parser.add_argument("--text", help="Text to summarize")
    parser.add_argument("--input", help="Input CSV file")
    parser.add_argument("--output", help="Output CSV file")
    parser.add_argument("--model", default="t5", choices=["t5", "bart", "pegasus", "extractive"],
                         help="Summarization model")
    parser.add_argument("--max_length", type=int, default=150, help="Maximum summary length")
    parser.add_argument("--text_column", default="content", help="Text column name")
    
    args = parser.parse_args()
    
    if args.text:
        # Summarize single text
        summarizer = TextSummarizer(args.model)
        summary = summarizer.summarize_text(args.text, args.max_length)
        print(f"Original: {args.text}")
        print(f"Summary: {summary}")
    
    elif args.input:
        # Summarize DataFrame
        try:
            df = pd.read_csv(args.input)
            summarizer = TextSummarizer(args.model)
            df_summarized = summarizer.summarize_dataframe(
                df, args.text_column, "summary", args.max_length
            )
            
            output_path = args.output or args.input.replace('.csv', '_summarized.csv')
            df_summarized.to_csv(output_path, index=False, encoding='utf-8')
            
            print(f"Summarization completed. Results saved to {output_path}")
            
        except Exception as e:
            print(f"Error: {str(e)}")
            sys.exit(1)
    
    else:
        print("Please provide either --text or --input argument")


if __name__ == "__main__":
    main()