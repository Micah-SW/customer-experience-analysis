import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from nltk.corpus import stopwords
import time
import nltk
import warnings
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- Configuration and Setup ---

warnings.filterwarnings('ignore')

# Download VADER lexicon and stopwords if not already present
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("📥 Downloading NLTK resources...")
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('stopwords', quiet=True)

# Define file paths and constants
INPUT_FILE = 'bank_reviews_clean.csv' # Assuming this is the raw input file name
OUTPUT_VADER_DASHBOARD = 'vader_sentiment_dashboard.png'
OUTPUT_EXECUTIVE_CHART = 'professional_sentiment_comparison.png'

STOP_WORDS = set(stopwords.words('english'))
# Enhanced list of custom stopwords to improve relevancy for review analysis
CUSTOM_STOPWORDS = STOP_WORDS.union({
    'app', 'bank', 'mobile', 'get', 'can', 'will', 'like', 'one', 'new', 'need',
    'know', 'user', 'make', 'use', 'try', 'time', 'also', 'just', 'still', 'would',
    'said', 'much', 'way', 'got', 'even', 'please'
})

# --- VADER Sentiment Analyzer Class (For Comprehensive Analysis) ---

class BankSentimentAnalyzer:
    """Performs VADER-based sentiment analysis and generates a detailed 4-plot dashboard."""
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.sentiment_thresholds = {
            'positive': 0.05,
            'negative': -0.05
        }
        self.setup_plotting()

    def setup_plotting(self):
        """Configure professional plotting style for the VADER dashboard"""
        plt.style.use('seaborn-v0_8')
        self.colors = {
            'Positive': '#2E8B57',  # SeaGreen
            'Neutral': '#4682B4',   # SteelBlue
            'Negative': '#DC143C'   # Crimson
        }

    def load_data(self, filepath):
        """Load and validate data"""
        print("📊 Loading data...")
        try:
            df = pd.read_csv(filepath, low_memory=False)
            required_cols = ['bank', 'review_text']
            if not all(col in df.columns for col in required_cols):
                # Attempt to rename a common column if 'review_text' is missing
                if 'content' in df.columns:
                    df.rename(columns={'content': 'review_text'}, inplace=True)
                else:
                    raise ValueError(f"Missing required columns: {required_cols}")

            print(f"✅ Loaded {len(df):,} reviews across {df['bank'].nunique()} banks")
            return df
        except FileNotFoundError:
            print(f"❌ Error: Input file '{filepath}' not found.")
            return None
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None

    def analyze_sentiment_batch(self, texts):
        """Vectorized VADER sentiment analysis"""
        sentiments = []
        scores = []

        for text in texts:
            # Handle NaN or empty strings robustly
            if pd.isna(text) or not isinstance(text, str) or not text.strip():
                sentiments.append("Neutral")
                scores.append(0.0)
                continue

            try:
                score_dict = self.analyzer.polarity_scores(text)
                compound_score = score_dict['compound']
                scores.append(compound_score)

                if compound_score >= self.sentiment_thresholds['positive']:
                    sentiments.append("Positive")
                elif compound_score <= self.sentiment_thresholds['negative']:
                    sentiments.append("Negative")
                else:
                    sentiments.append("Neutral")
            except Exception as e:
                print(f"⚠️ Error analyzing text: {e}")
                sentiments.append("Neutral")
                scores.append(0.0)

        return sentiments, scores

    # --- Plotting functions for VADER Dashboard (omitted for brevity, assume they are fully implemented) ---
    def create_sentiment_dashboard(self, df):
        """Create comprehensive sentiment visualization dashboard"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🏦 Bank Sentiment Analysis Dashboard (VADER)', fontsize=16, fontweight='bold')

        # Plot 1: Sentiment Distribution by Bank (Stacked Bar)
        self._plot_sentiment_stacked_bar(df, axes[0, 0])

        # Plot 2: Sentiment Score Distribution
        self._plot_sentiment_distribution(df, axes[0, 1])

        # Plot 3: Sentiment Proportions (Pie Chart)
        self._plot_sentiment_proportions(df, axes[1, 0])

        # Plot 4: Average Sentiment by Bank
        self._plot_average_sentiment(df, axes[1, 1])

        plt.tight_layout()
        plt.savefig(OUTPUT_VADER_DASHBOARD, dpi=300, bbox_inches='tight')
        plt.close(fig) # Close figure to prevent display issues
        print(f"🖼️  VADER Dashboard saved as: {OUTPUT_VADER_DASHBOARD}")

    # Helper plotting methods (place all _plot methods here, as in the user's VADER code block)
    def _plot_sentiment_stacked_bar(self, df, ax):
        sentiment_summary = pd.crosstab(df['bank'], df['sentiment_label'])
        for sentiment in ['Positive', 'Neutral', 'Negative']:
            if sentiment not in sentiment_summary.columns: sentiment_summary[sentiment] = 0
        sentiment_summary = sentiment_summary[['Positive', 'Neutral', 'Negative']]

        sentiment_pct = sentiment_summary.div(sentiment_summary.sum(axis=1), axis=0) * 100

        bars = sentiment_pct.plot(kind='bar', stacked=True, ax=ax,
                                color=[self.colors[label] for label in sentiment_pct.columns],
                                edgecolor='black', linewidth=0.5)

        ax.set_title('📈 Sentiment Distribution by Bank (VADER)', fontweight='bold')
        ax.set_xlabel('Bank', fontweight='bold')
        ax.set_ylabel('Percentage (%)', fontweight='bold')
        ax.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=45)

        for container in bars.containers:
            ax.bar_label(container, fmt='%.1f%%', label_type='center', fontsize=8)

    def _plot_sentiment_distribution(self, df, ax):
        sns.violinplot(data=df, x='sentiment_label', y='sentiment_score',
                      ax=ax, palette=self.colors, order=['Positive', 'Neutral', 'Negative'])
        ax.set_title('🎻 VADER Sentiment Score Distribution', fontweight='bold')
        ax.set_xlabel('Sentiment Category', fontweight='bold')
        ax.set_ylabel('Compound Sentiment Score', fontweight='bold')

        for i, category in enumerate(['Positive', 'Neutral', 'Negative']):
            mean_score = df[df['sentiment_label'] == category]['sentiment_score'].mean()
            ax.scatter(i, mean_score, color='white', s=80, edgecolor='black', zorder=3, label=f'Mean: {mean_score:.3f}')

    def _plot_sentiment_proportions(self, df, ax):
        sentiment_counts = df['sentiment_label'].value_counts()
        colors = [self.colors[label] for label in sentiment_counts.index]

        wedges, texts, autotexts = ax.pie(sentiment_counts.values, labels=sentiment_counts.index,
                                         autopct='%1.1f%%', colors=colors, startangle=90,
                                         explode=[0.05, 0.02, 0.02])

        ax.set_title('🥧 Overall Sentiment Proportions (VADER)', fontweight='bold')

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

    def _plot_average_sentiment(self, df, ax):
        avg_sentiment = df.groupby('bank')['sentiment_score'].mean().sort_values()

        colors = [self.colors['Positive'] if x > 0.05 else self.colors['Negative'] if x < -0.05 else self.colors['Neutral'] for x in avg_sentiment.values]

        bars = ax.barh(range(len(avg_sentiment)), avg_sentiment.values, color=colors, alpha=0.7)
        ax.set_yticks(range(len(avg_sentiment)))
        ax.set_yticklabels(avg_sentiment.index)
        ax.set_title('📊 Average VADER Sentiment by Bank', fontweight='bold')
        ax.set_xlabel('Average Compound Score', fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)

        ax.axvline(x=0.05, color='green', linestyle='--', alpha=0.5, label='Positive Threshold')
        ax.axvline(x=-0.05, color='red', linestyle='--', alpha=0.5, label='Negative Threshold')

        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + (0.01 if width >= 0 else -0.03), bar.get_y() + bar.get_height()/2,
                   f'{width:.3f}', ha='left' if width >= 0 else 'right', va='center', fontweight='bold')

        ax.legend(loc='lower right')

    def generate_summary_report(self, df):
        """Generate comprehensive analysis report (printed to console)"""
        print("\n" + "="*60)
        print("📊 VADER SENTIMENT ANALYSIS REPORT")
        print("="*60)

        # Overall statistics
        total_reviews = len(df)
        sentiment_dist = df['sentiment_label'].value_counts()

        print(f"\n📈 OVERVIEW:")
        print(f"   • Total Reviews: {total_reviews:,}")
        print(f"   • Positive Reviews: {sentiment_dist.get('Positive', 0):,} ({sentiment_dist.get('Positive', 0)/total_reviews*100:.1f}%)")
        print(f"   • Negative Reviews: {sentiment_dist.get('Negative', 0):,} ({sentiment_dist.get('Negative', 0)/total_reviews*100:.1f}%)")

        # Bank-wise performance
        print(f"\n🏆 BANK PERFORMANCE RANKING (Based on Avg. Sentiment Score):")
        bank_sentiment = df.groupby('bank').agg({
            'sentiment_score': ['mean', 'count'],
            'sentiment_label': lambda x: (x == 'Positive').sum() / len(x) * 100
        }).round(3)

        bank_sentiment.columns = ['avg_sentiment', 'review_count', 'positive_pct']
        bank_sentiment = bank_sentiment.sort_values('avg_sentiment', ascending=False)

        for i, (bank, row) in enumerate(bank_sentiment.iterrows(), 1):
            sentiment_emoji = "🟢" if row['avg_sentiment'] > 0.05 else "🔴" if row['avg_sentiment'] < -0.05 else "🟡"
            print(f"   {i}. {bank}: {sentiment_emoji} Score={row['avg_sentiment']:.3f}, "
                  f"Positive={row['positive_pct']:.1f}%, Reviews={row['review_count']:,}")

    def run_analysis(self, filepath):
        """Main VADER analysis pipeline"""
        print("\n🧠 Starting VADER Sentiment Analysis...")
        
        df = self.load_data(filepath)
        if df is None: return None

        print("📝 Analyzing sentiment with VADER...")
        start_time = time.time()
        sentiments, scores = self.analyze_sentiment_batch(df['review_text'].values)
        df['sentiment_label'] = sentiments
        df['sentiment_score'] = scores
        print(f"✅ Sentiment analysis complete in {time.time() - start_time:.2f} seconds.")

        self.generate_summary_report(df)
        self.create_sentiment_dashboard(df)

        output_file = 'bank_reviews_vader_analysis.csv'
        df.to_csv(output_file, index=False)
        print(f"💾 Detailed VADER results saved to: {output_file}")
        
        return df

# --- Executive Reporting Functions (From Previous Iteration) ---

def get_top_keywords(df, sentiment_filter='Negative', top_n=10):
    """Extracts the most frequent non-stop words from a filtered set of reviews."""
    
    filtered_df = df[df['sentiment_label'] == sentiment_filter]
    
    if filtered_df.empty: return []
        
    all_text = ' '.join(filtered_df['review_text'].astype(str).str.lower())
    words = re.findall(r'\b[a-z]{3,}\b', all_text)
    
    # Filter using the global custom stopwords
    filtered_words = [word for word in words if word not in CUSTOM_STOPWORDS and len(word) > 2]
    
    word_counts = Counter(filtered_words)
    return word_counts.most_common(top_n)

def plot_professional_comparison(df):
    """Generates the professional, minimalist stacked bar chart for the executive report."""
    
    # Calculate sentiment percentages
    sentiment_counts = df.groupby('bank')['sentiment_label'].value_counts(normalize=True).mul(100).rename('percentage').reset_index()

    # Get total reviews for annotation
    total_counts = df['bank'].value_counts().reset_index()
    total_counts.columns = ['bank', 'total_reviews']

    # --- PLOT SETUP ---
    plt.style.use('default') 
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Custom, professional palette
    palette = {
        'Positive': '#00bfa5', # Success Green
        'Neutral': '#cccccc', # Light Gray
        'Negative': '#f44336' # Error Red
    }
    sentiment_order = ['Negative', 'Neutral', 'Positive'] 
    
    # Determine the plot order (banks sorted by their Positive Sentiment percentage)
    plot_order = sentiment_counts[sentiment_counts['sentiment_label'] == 'Positive'].sort_values('percentage', ascending=False)['bank']

    # Create the stacked bar chart (Horizontal)
    sns.barplot(
        x='percentage', 
        y='bank', 
        hue='sentiment_label', 
        data=sentiment_counts, 
        hue_order=sentiment_order,
        palette=palette,
        dodge=False,
        ax=ax,
        order=plot_order
    )
    
    # --- AESTHETIC ENHANCEMENTS ---
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_xlabel('') 
    ax.set_ylabel('') 

    # --- FIXED AND ROBUST LABELING ---
    plot_data = sentiment_counts.pivot(index='bank', columns='sentiment_label', values='percentage').fillna(0)
    plotted_bank_labels = [t.get_text() for t in ax.get_yticklabels()]
    plot_data = plot_data.loc[plotted_bank_labels] 

    # 1. CENTER LABELING (inside the bar segment)
    for i, bank_name in enumerate(plot_data.index):
        current_x = 0
        
        for sentiment in sentiment_order:
            percentage = plot_data.loc[bank_name, sentiment]
            
            if percentage > 2.5: 
                label_x = current_x + (percentage / 2)
                label_y = i 
                
                # Use white text for the dark Negative segment for readability
                color = 'white' if sentiment in ['Negative', 'Positive'] else 'black'
                
                ax.text(label_x, label_y, 
                        f'{percentage:.1f}%', 
                        ha='center', va='center', fontsize=9, color=color, weight='bold')
                        
            current_x += percentage

    # 2. TOTAL REVIEW COUNT (n= annotation on the far right)
    for i, bank_name in enumerate(plot_data.index):
        total_n = total_counts[total_counts['bank'] == bank_name]['total_reviews'].iloc[0]
        
        ax.text(102, 
                i, 
                f'(n={total_n})', 
                va='center', 
                ha='left', 
                fontsize=9, 
                color='gray')

    # Add custom Title and Legend
    ax.set_title('Executive Summary: App Satisfaction Comparison', fontsize=16, pad=20, loc='left', weight='bold')
    ax.legend(title='Sentiment', loc='upper left', bbox_to_anchor=(0.0, -0.05), ncol=3, frameon=False)
    ax.set_xlim(0, 105)

    plt.tight_layout()
    plt.savefig(OUTPUT_EXECUTIVE_CHART)
    plt.close(fig) # Close figure
    print(f"\n✅ Final Executive Chart saved as '{OUTPUT_EXECUTIVE_CHART}'")

# --- MAIN EXECUTION ---
def main():
    """Runs the full two-phase analysis pipeline."""
    full_start_time = time.time()
    
    # PHASE 1: VADER Analysis and Dashboard Generation
    analyzer = BankSentimentAnalyzer()
    
    # This phase loads raw data, calculates sentiment_label and sentiment_score, 
    # saves the detailed analysis CSV, and generates the 4-plot VADER dashboard.
    results_df = analyzer.run_analysis(INPUT_FILE) 
    
    if results_df is None:
        print("\n🛑 Cannot proceed to Executive Analysis. VADER Phase failed.")
        return

    # PHASE 2: Executive Visualization and Keyword Extraction
    print("\n" + "="*60)
    print("⭐ STARTING EXECUTIVE REPORTING PHASE")
    print("="*60)
    
    # 1. Generate the Professional Stacked Bar Chart
    plot_professional_comparison(results_df) 
    
    # 2. Extract Top Pain Points (using the sentiment labels generated by VADER)
    negative_count = len(results_df[results_df['sentiment_label'] == 'Negative'])
    print(f"\n--- Top 10 Urgent Pain Points (from {negative_count} Negative Reviews) ---")
    pain_points = get_top_keywords(results_df, sentiment_filter='Negative', top_n=10)

    if pain_points:
        for i, (word, count) in enumerate(pain_points):
            print(f"{i+1}. {word.capitalize()} (Mentions: {count})")
    
    full_end_time = time.time()
    print(f"\n⏱️ Total time for FULL analysis pipeline: {full_end_time - full_start_time:.2f} seconds.")

if __name__ == "__main__":
    main()