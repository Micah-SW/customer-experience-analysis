import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
import warnings
import os
import logging
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import joblib
from dataclasses import dataclass
from datetime import datetime
import json
import numpy as np
import re  # Added for text cleaning

# Setup
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure NLTK resources - UPDATED to include punkt_tab
NLTK_RESOURCES = ['stopwords', 'punkt', 'punkt_tab', 'wordnet', 'averaged_perceptron_tagger', 'omw-eng']

def ensure_nltk_resources(resources: List[str] = NLTK_RESOURCES):
    """Ensure all required NLTK resources are available."""
    for resource in resources:
        try:
            # Check for resource existence
            nltk.data.find(resource)
            logger.debug(f"NLTK resource '{resource}' is already available")
        except LookupError:
            logger.info(f"Downloading NLTK resource: {resource}")
            # Download the resource quietly
            try:
                nltk.download(resource, quiet=True)
                logger.info(f"Successfully downloaded NLTK resource: {resource}")
            except Exception as e:
                logger.warning(f"Failed to download {resource}: {e}")
                # For punkt_tab, try alternative approach if direct download fails
                if resource == 'punkt_tab':
                    logger.info("Attempting to download full punkt package...")
                    nltk.download('punkt', quiet=True)

# Download ALL required NLTK resources at module import time
ensure_nltk_resources()

# Configuration
@dataclass
class AnalysisConfig:
    """Configuration for analysis pipeline."""
    # Updated default to reflect the file is within a 'data/' folder relative to the project root
    data_file: str = 'data/bank_reviews_vader_analysis.csv'
    n_topics: int = 4
    cache_dir: str = 'cache'
    output_dir: str = 'outputs'
    chunk_size: int = 10000  # For large files
    
    def __post_init__(self):
        """Create necessary directories after initialization."""
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)


class EfficientDataLoader:
    """Efficient data loading and preprocessing."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.df = None
    
    def load(self) -> pd.DataFrame:
        """Load data with memory optimization and robust path resolution."""
        
        # 1. Get the absolute path of the current script (which is in src/)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 2. Construct the data path by going up one level ('..') from src/ 
        #    and then appending the relative path from the config ('data/bank_reviews_vader_analysis.csv')
        data_path = os.path.join(script_dir, '..', self.config.data_file)

        logger.info(f"Loading data from {data_path}")
        
        if not os.path.exists(data_path):
            # Error message now includes the full path checked
            raise FileNotFoundError(f"Data file not found: {self.config.data_file}. Checked path: {data_path}")
        
        # Optimized dtype mapping for memory reduction
        dtype_map = {
            'review_text': 'string',
            'bank': 'category',
            'sentiment_label': 'category',
            'source': 'category',
            'rating': 'int8',
            'sentiment_score': 'float32'
        }
        
        try:
            # Use the resolved data_path
            self.df = pd.read_csv(
                data_path,
                dtype=dtype_map,
                parse_dates=['date'],
                usecols=list(dtype_map.keys()) + ['date'],
                engine='c',  # C engine for speed
                low_memory=False
            )
            
            # Clean and validate
            self._clean_data()
            memory_usage = self.df.memory_usage(deep=True).sum() / 1024**2
            logger.info(f"Loaded {len(self.df)} rows, {memory_usage:.2f} MB")
            return self.df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}", exc_info=True)
            raise
    
    def _clean_data(self):
        """Clean and validate data."""
        # Remove duplicates
        initial_len = len(self.df)
        self.df.drop_duplicates(subset=['review_text'], inplace=True)
        logger.info(f"Removed {initial_len - len(self.df)} duplicates")
        
        # Handle missing values in review text
        self.df['review_text'] = self.df['review_text'].fillna('')
        self.df = self.df[self.df['review_text'].str.strip() != '']
        
        # Ensure sentiment labels are valid
        valid_labels = {'Positive', 'Negative', 'Neutral'}
        invalid_mask = ~self.df['sentiment_label'].isin(valid_labels)
        if invalid_mask.any():
            logger.warning(f"Removing {invalid_mask.sum()} rows with invalid sentiment labels")
            self.df = self.df[~invalid_mask]


class TextPreprocessor:
    """Text preprocessing for NLP tasks."""
    
    def __init__(self, language='english'):
        self.lemmatizer = WordNetLemmatizer()
        # Ensure stopwords are available
        try:
            self.stop_words = set(stopwords.words(language))
        except LookupError:
            logger.warning("Stopwords not found, downloading...")
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words(language))
        
        self._add_custom_stopwords()
    
    def _add_custom_stopwords(self):
        """Add domain-specific stopwords to improve topic model quality."""
        custom = {
            'app', 'bank', 'mobile', 'cbe', 'dashen', 'boa',
            'banking', 'account', 'customer', 'service',
            'please', 'thank', 'thanks', 'hello', 'hi',
            # Add common noise words
            'good', 'great', 'review', 'user', 'make', 'use', 'like',
            'just', 'get', 'dont', 'go', 'us', 'say', 'would', 'could',
            'also', 'one', 'two', 'three', 'first', 'second', 'third'
        }
        self.stop_words.update(custom)
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text with error handling for tokenization."""
        if not isinstance(text, str) or not text.strip():
            return ""
        
        try:
            # Basic text cleaning before tokenization
            text = text.lower()
            # Remove URLs, special characters, numbers
            text = re.sub(r'http\S+|www\S+', '', text)
            text = re.sub(r'[^a-zA-Z\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Tokenize with error handling
            try:
                tokens = word_tokenize(text)
            except LookupError as e:
                logger.warning(f"Tokenization failed: {e}. Attempting to download required resources...")
                # Try to download punkt if missing
                nltk.download('punkt', quiet=True)
                nltk.download('punkt_tab', quiet=True)
                tokens = word_tokenize(text)
            
            # Remove stopwords and non-alphabetic tokens, lemmatize
            processed_tokens = []
            for token in tokens:
                # Check if token is purely alphabetic, not a stopword, and is long enough
                if token.isalpha() and token not in self.stop_words and len(token) > 2:
                    # Try to lemmatize as verb first, then noun
                    lemma = self.lemmatizer.lemmatize(token, pos='v')
                    lemma = self.lemmatizer.lemmatize(lemma, pos='n')
                    processed_tokens.append(lemma)
            
            return ' '.join(processed_tokens)
            
        except Exception as e:
            logger.warning(f"Error preprocessing text: {e}")
            return ""  # Return empty string for failed preprocessing


class SentimentAnalyzer:
    """Analyze sentiment trends efficiently."""
    
    def __init__(self, df: pd.DataFrame):
        # Ensure 'date' is datetime type for time series analysis
        self.df = df.copy()
        if 'date' in self.df.columns and not pd.api.types.is_datetime64_any_dtype(self.df['date']):
            self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
        self.df.dropna(subset=['date'], inplace=True)
        self.results = {}
    
    def analyze_time_trends(self, freq: str = 'M') -> pd.DataFrame:
        """Calculate sentiment trends over time (Monthly by default)."""
        logger.info(f"Analyzing sentiment trends with frequency: {freq}")
        
        # Create time period column
        self.df['period'] = self.df['date'].dt.to_period(freq).dt.to_timestamp()
        
        # Group and aggregate
        grouped = self.df.groupby(['period', 'bank'])['sentiment_score'].agg(
            mean='mean',
            median='median',
            count='count',
            std='std'
        ).reset_index()
        
        # Calculate standard error and confidence intervals (95%)
        grouped['std_err'] = grouped['std'] / np.sqrt(grouped['count'])
        grouped['ci_lower'] = grouped['mean'] - 1.96 * grouped['std_err']
        grouped['ci_upper'] = grouped['mean'] + 1.96 * grouped['std_err']
        
        self.results['time_trends'] = grouped
        return grouped
    
    def plot_trends(self, output_path: str = None, **kwargs):
        """Create optimized sentiment trend plot."""
        if 'time_trends' not in self.results:
            self.analyze_time_trends()
        
        data = self.results['time_trends']
        
        # Use seaborn style for better aesthetics
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(14, 7), dpi=150)
        
        # Plot each bank
        banks = data['bank'].unique()
        # Use the 'viridis' colormap for distinct, colorblind-friendly colors
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(banks)))
        
        for bank, color in zip(banks, colors):
            bank_data = data[data['bank'] == bank]
            # Plot mean sentiment score
            ax.plot(bank_data['period'], bank_data['mean'], 
                    color=color, linewidth=2.5, marker='o', markersize=6,
                    label=bank, alpha=0.9)
            
            # Add confidence interval shading
            ax.fill_between(bank_data['period'], 
                            bank_data['ci_lower'], 
                            bank_data['ci_upper'],
                            color=color, alpha=0.2, linewidth=0)
        
        # Add reference lines and annotations
        ax.axhline(y=0, color='#e74c3c', linestyle='--', alpha=0.7, 
                   label='Neutral Threshold')
        
        # Optimize layout
        ax.set_xlabel('Date (Monthly)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Sentiment Score (VADER) ± 95% CI', fontsize=12, fontweight='bold')
        ax.set_title('Customer Sentiment Trends Over Time by Bank', 
                     fontsize=16, fontweight='bold', pad=20)
        
        # Improve legend and set score range
        ax.legend(title='Bank', title_fontsize=12, fontsize=10, 
                  loc='upper left', frameon=True, fancybox=True)
        ax.set_ylim(-1, 1)  # VADER scores range from -1 to 1
        
        # Format x-axis
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
        fig.autofmt_xdate(rotation=45, ha='right')
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Save plot
        if output_path:
            fig.savefig(output_path, bbox_inches='tight', dpi=300, 
                        facecolor='white', edgecolor='none',
                        metadata={'CreationDate': datetime.now().isoformat()})
            plt.close(fig)
            logger.info(f"Saved plot to {output_path}")
        
        return fig, ax


class LDATopicModeler:
    """Optimized LDA topic modeling with caching."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.preprocessor = TextPreprocessor()
        self.pipeline = None
    
    def create_pipeline(self, n_topics: int = None):
        """Create optimized LDA pipeline."""
        n_topics = n_topics or self.config.n_topics
        
        # The pipeline includes CountVectorizer and LDA
        return Pipeline([
            ('vectorizer', CountVectorizer(
                max_df=0.85,  # Ignore terms that appear in more than 85% of documents
                min_df=5,  # Ignore terms that appear in less than 5 documents
                max_features=3000,  # Limit vocabulary size
                ngram_range=(1, 2),  # Use unigrams and bigrams
                stop_words='english',  # Use built-in English stopwords
                dtype='int32',  # Optimize memory for count matrix
                binary=False  # Use term frequencies
            )),
            ('lda', LatentDirichletAllocation(
                n_components=n_topics,
                max_iter=15,
                learning_method='online',  # Use online learning for large datasets
                learning_offset=50.,
                batch_size=128,
                evaluate_every=2,
                n_jobs=-1,  # Use all available cores
                random_state=42,
                verbose=0  # Reduced verbosity
            ))
        ])
    
    def get_cache_key(self, texts: List[str], n_topics: int) -> str:
        """Generate cache key based on content and configuration."""
        # Hash a snippet of the content (first 500 reviews) for speed and all params
        snippet = ''.join(texts[:500]) if len(texts) > 500 else ''.join(texts)
        config_params = f"{n_topics}_{self.config.data_file}_{self.config.chunk_size}"
        content_hash = hashlib.md5((snippet + config_params).encode()).hexdigest()[:16]
        return os.path.join(self.config.cache_dir, 
                            f"lda_{content_hash}_{n_topics}.pkl")
    
    def extract_topics(self, df: pd.DataFrame, use_cache: bool = True) -> Dict:
        """Extract topics from negative reviews."""
        # Filter only negative reviews for issue identification
        neg_reviews = df[df['sentiment_label'] == 'Negative']['review_text'].astype(str)
        
        if len(neg_reviews) < 20:
            logger.warning("Insufficient negative reviews for topic modeling (<20)")
            return {}
        
        # --- Preprocessing Step ---
        logger.info(f"Preprocessing {len(neg_reviews)} negative reviews")
        
        # Process texts sequentially with error handling
        processed_texts = []
        failed_count = 0
        
        for text in neg_reviews:
            try:
                processed = self.preprocessor.preprocess_text(text)
                if processed.strip():  # Only add non-empty texts
                    processed_texts.append(processed)
            except Exception as e:
                failed_count += 1
                if failed_count <= 5:  # Log first 5 failures
                    logger.debug(f"Failed to preprocess text: {e}")
        
        if failed_count > 0:
            logger.warning(f"Failed to preprocess {failed_count} texts")
        
        # Filter out empty texts after processing
        processed_texts = [text for text in processed_texts if text.strip()]
        
        if len(processed_texts) < 10:
            logger.warning(f"No valid text after preprocessing ({len(processed_texts)} < 10)")
            return {}
        
        # --- Caching and Model Training ---
        cache_key = self.get_cache_key(processed_texts, self.config.n_topics)
        
        if use_cache and os.path.exists(cache_key):
            logger.info("Loading cached LDA results")
            # Handle potential loading errors gracefully
            try:
                return joblib.load(cache_key)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Retraining model.")
        
        # Train LDA
        logger.info(f"Training LDA with {len(processed_texts)} documents")
        try:
            pipeline = self.create_pipeline()
            pipeline.fit(processed_texts)
        except ValueError as e:
            logger.error(f"LDA training failed: {e}")
            if "empty vocabulary" in str(e):
                logger.error("No valid features after preprocessing. Check stopwords and preprocessing.")
            return {}
        
        # Extract results
        vectorizer = pipeline.named_steps['vectorizer']
        lda = pipeline.named_steps['lda']
        
        feature_names = vectorizer.get_feature_names_out()
        topics = {'model_config': {'n_topics': self.config.n_topics, 'max_features': 3000}}
        
        for topic_idx, topic_weights in enumerate(lda.components_):
            top_indices = topic_weights.argsort()[-15:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            top_weights = topic_weights[top_indices].tolist()
            
            topics[f"topic_{topic_idx + 1}"] = {
                'words': top_words,
                'weights': top_weights,
                # Simplified coherence metric for quick check
                'simplified_coherence': self._calculate_coherence(topic_weights, top_indices)
            }
        
        # Calculate document-topic distribution (use a slice to reduce size)
        doc_topic_dist = pipeline.transform(processed_texts)
        topics['doc_topic_distribution_sample'] = doc_topic_dist[:min(100, len(processed_texts))].tolist()  # Sample up to 100
        topics['log_likelihood'] = lda.score(vectorizer.transform(processed_texts))
        topics['n_documents'] = len(processed_texts)
        
        # Cache results
        if use_cache:
            try:
                joblib.dump(topics, cache_key)
                logger.info(f"Cached results to {cache_key}")
            except Exception as e:
                logger.warning(f"Failed to cache results: {e}")
        
        return topics
    
    def _calculate_coherence(self, topic_weights: np.ndarray, top_indices: np.ndarray) -> float:
        """Calculate simple topic coherence score (mean weight / std dev)."""
        # This is not true C_v or UMass coherence but a quick metric.
        top_weights = topic_weights[top_indices]
        if np.std(top_weights) == 0:
            return 0.0
        return float(np.mean(top_weights) / np.std(top_weights))
    
    def visualize_topics(self, topics: Dict, output_path: str = None):
        """Visualize topics as horizontal bar charts."""
        if not topics or 'model_config' not in topics:
            return
        
        n_topics = topics['model_config']['n_topics']
        
        # Setup the plot area
        fig, axes = plt.subplots(n_topics, 1, figsize=(10, 3.5 * n_topics), dpi=150)
        
        # Ensure axes is iterable even if only one topic exists
        if n_topics == 1:
            axes = [axes]
        
        # Use a professional style
        plt.style.use('ggplot')
        
        for idx, ax in enumerate(axes, 1):
            topic_key = f"topic_{idx}"
            if topic_key in topics:
                words = topics[topic_key]['words'][:10]
                weights = topics[topic_key]['weights'][:10]
                
                # Sort for visualization clarity
                words.reverse()
                weights.reverse()
                
                y_pos = np.arange(len(words))
                ax.barh(y_pos, weights, align='center', alpha=0.9, color=plt.cm.cividis(idx / n_topics))
                ax.set_yticks(y_pos)
                ax.set_yticklabels(words, fontsize=11)
                ax.set_xlabel('Relative Term Weight', fontsize=10)
                coherence = topics[topic_key].get('simplified_coherence', 'N/A')
                ax.set_title(f'Topic {idx} - Keywords (Coherence: {coherence:.2f})', fontsize=13, fontweight='bold')
                
        plt.tight_layout(pad=3.0)
        
        if output_path:
            fig.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
            logger.info(f"Saved topic visualization to {output_path}")


def main():
    """Main analysis pipeline."""
    # Configuration
    config = AnalysisConfig(
        data_file='data/bank_reviews_vader_analysis.csv',  # Now explicitly set to look inside the 'data/' folder
        n_topics=4,
        cache_dir='./cache',
        output_dir='./outputs'
    )
    
    try:
        # 1. Load data
        logger.info("--- STARTING DATA LOADING ---")
        loader = EfficientDataLoader(config)
        df = loader.load()
        
        # 2. Sentiment analysis
        logger.info("--- STARTING SENTIMENT ANALYSIS ---")
        analyzer = SentimentAnalyzer(df)
        analyzer.analyze_time_trends()
        analyzer.plot_trends(output_path=os.path.join(config.output_dir, 'sentiment_trends.png'))
        
        # 3. Topic modeling (on negative reviews)
        logger.info("--- STARTING TOPIC MODELING ---")
        modeler = LDATopicModeler(config)
        topics = modeler.extract_topics(df, use_cache=True)
        
        if topics:
            # Save topics to JSON
            topics_file = os.path.join(config.output_dir, 'topics.json')
            # Exclude large document distribution sample from main JSON output if needed, but keeping for now
            topics_to_save = topics.copy()
            if 'doc_topic_distribution_sample' in topics_to_save:
                del topics_to_save['doc_topic_distribution_sample']

            with open(topics_file, 'w') as f:
                json.dump(topics_to_save, f, indent=2)
            logger.info(f"Saved topics to {topics_file}")
            
            # Visualize topics
            modeler.visualize_topics(
                topics, 
                output_path=os.path.join(config.output_dir, 'topics_visualization.png')
            )
        else:
            logger.warning("Topic modeling skipped due to insufficient data or preprocessing issues.")
            
        logger.info("--- ANALYSIS COMPLETED SUCCESSFULLY ---")
        
    except FileNotFoundError as e:
        logger.error(f"FATAL ERROR: {e}. Please ensure the input data file exists.")
    except Exception as e:
        logger.error(f"Analysis failed due to an unexpected error: {e}", exc_info=True)
        # Re-raise to stop execution if main component failed
        raise


if __name__ == "__main__":
    # Test NLTK resources before starting
    logger.info("Verifying NLTK resources...")
    ensure_nltk_resources()
    
    main()