import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Download VADER lexicon if not already present
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    print("📥 Downloading VADER lexicon...")
    nltk.download('vader_lexicon')

class BankSentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.sentiment_thresholds = {
            'positive': 0.05,    # VADER recommended thresholds
            'negative': -0.05
        }
        self.setup_plotting()
    
    def setup_plotting(self):
        """Configure professional plotting style"""
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
            df = pd.read_csv(filepath)
            required_cols = ['bank', 'review_text']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Missing required columns: {required_cols}")
            
            print(f"✅ Loaded {len(df):,} reviews across {df['bank'].nunique()} banks")
            return df
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def analyze_sentiment_batch(self, texts):
        """Vectorized VADER sentiment analysis for better performance"""
        sentiments = []
        scores = []
        
        for text in texts:
            if pd.isna(text) or not isinstance(text, str) or not text.strip():
                sentiments.append("Neutral")
                scores.append(0.0)
                continue
                
            try:
                # VADER analysis - much better for reviews!
                score_dict = self.analyzer.polarity_scores(text)
                compound_score = score_dict['compound']
                scores.append(compound_score)
                
                # Use VADER's recommended thresholds
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
        plt.savefig('vader_sentiment_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_sentiment_stacked_bar(self, df, ax):
        """Stacked bar chart showing sentiment distribution per bank"""
        sentiment_summary = pd.crosstab(df['bank'], df['sentiment_label'])
        # Ensure all sentiment categories exist
        for sentiment in ['Positive', 'Neutral', 'Negative']:
            if sentiment not in sentiment_summary.columns:
                sentiment_summary[sentiment] = 0
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
        
        # Add value labels on bars
        for container in bars.containers:
            ax.bar_label(container, fmt='%.1f%%', label_type='center', fontsize=8)
    
    def _plot_sentiment_distribution(self, df, ax):
        """Violin plot showing sentiment score distribution"""
        sns.violinplot(data=df, x='sentiment_label', y='sentiment_score', 
                      ax=ax, palette=self.colors, order=['Positive', 'Neutral', 'Negative'])
        ax.set_title('🎻 VADER Sentiment Score Distribution', fontweight='bold')
        ax.set_xlabel('Sentiment Category', fontweight='bold')
        ax.set_ylabel('Compound Sentiment Score', fontweight='bold')
        
        # Add mean markers
        for i, category in enumerate(['Positive', 'Neutral', 'Negative']):
            mean_score = df[df['sentiment_label'] == category]['sentiment_score'].mean()
            ax.scatter(i, mean_score, color='white', s=80, edgecolor='black', zorder=3, label=f'Mean: {mean_score:.3f}')
    
    def _plot_sentiment_proportions(self, df, ax):
        """Pie chart showing overall sentiment proportions"""
        sentiment_counts = df['sentiment_label'].value_counts()
        colors = [self.colors[label] for label in sentiment_counts.index]
        
        wedges, texts, autotexts = ax.pie(sentiment_counts.values, labels=sentiment_counts.index,
                                         autopct='%1.1f%%', colors=colors, startangle=90,
                                         explode=[0.05, 0.02, 0.02])
        
        ax.set_title('🥧 Overall Sentiment Proportions (VADER)', fontweight='bold')
        
        # Style the percentage texts
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    
    def _plot_average_sentiment(self, df, ax):
        """Bar plot showing average sentiment score by bank"""
        avg_sentiment = df.groupby('bank')['sentiment_score'].mean().sort_values()
        
        colors = ['#2E8B57' if x > 0.05 else '#DC143C' if x < -0.05 else '#4682B4' for x in avg_sentiment.values]
        
        bars = ax.barh(range(len(avg_sentiment)), avg_sentiment.values, color=colors, alpha=0.7)
        ax.set_yticks(range(len(avg_sentiment)))
        ax.set_yticklabels(avg_sentiment.index)
        ax.set_title('📊 Average VADER Sentiment by Bank', fontweight='bold')
        ax.set_xlabel('Average Compound Score', fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add threshold lines
        ax.axvline(x=0.05, color='green', linestyle='--', alpha=0.5, label='Positive Threshold')
        ax.axvline(x=-0.05, color='red', linestyle='--', alpha=0.5, label='Negative Threshold')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + (0.01 if width >= 0 else -0.03), bar.get_y() + bar.get_height()/2,
                   f'{width:.3f}', ha='left' if width >= 0 else 'right', va='center', fontweight='bold')
        
        ax.legend()
    
    def analyze_sentiment_intensity(self, df):
        """Additional VADER-specific analysis"""
        print("\n🔥 VADER Intensity Analysis:")
        
        # Get most positive and negative reviews
        most_positive = df.nlargest(3, 'sentiment_score')[['bank', 'review_text', 'sentiment_score']]
        most_negative = df.nsmallest(3, 'sentiment_score')[['bank', 'review_text', 'sentiment_score']]
        
        print("\n🏆 MOST POSITIVE REVIEWS:")
        for idx, row in most_positive.iterrows():
            print(f"   Score: {row['sentiment_score']:.3f} | Bank: {row['bank']}")
            print(f"   Review: {row['review_text'][:100]}...")
            print()
        
        print("\n💔 MOST NEGATIVE REVIEWS:")
        for idx, row in most_negative.iterrows():
            print(f"   Score: {row['sentiment_score']:.3f} | Bank: {row['bank']}")
            print(f"   Review: {row['review_text'][:100]}...")
            print()
    
    def generate_summary_report(self, df):
        """Generate comprehensive analysis report"""
        print("\n" + "="*60)
        print("📊 VADER SENTIMENT ANALYSIS REPORT")
        print("="*60)
        
        # Overall statistics
        total_reviews = len(df)
        banks_count = df['bank'].nunique()
        sentiment_dist = df['sentiment_label'].value_counts()
        
        print(f"\n📈 OVERVIEW:")
        print(f"   • Total Reviews: {total_reviews:,}")
        print(f"   • Banks Analyzed: {banks_count}")
        print(f"   • Positive Reviews: {sentiment_dist.get('Positive', 0):,} ({sentiment_dist.get('Positive', 0)/total_reviews*100:.1f}%)")
        print(f"   • Neutral Reviews: {sentiment_dist.get('Neutral', 0):,} ({sentiment_dist.get('Neutral', 0)/total_reviews*100:.1f}%)")
        print(f"   • Negative Reviews: {sentiment_dist.get('Negative', 0):,} ({sentiment_dist.get('Negative', 0)/total_reviews*100:.1f}%)")
        
        # Bank-wise performance
        print(f"\n🏆 BANK PERFORMANCE RANKING:")
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
        
        # VADER-specific insights
        self.analyze_sentiment_intensity(df)
    
    def run_analysis(self, filepath):
        """Main analysis pipeline"""
        print("🧠 Starting VADER Sentiment Analysis...")
        print("💡 VADER is optimized for social media and review text!")
        
        # Load data
        df = self.load_data(filepath)
        if df is None:
            return
        
        # Analyze sentiment with VADER
        print("📝 Analyzing sentiment with VADER...")
        sentiments, scores = self.analyze_sentiment_batch(df['review_text'].values)
        df['sentiment_label'] = sentiments
        df['sentiment_score'] = scores
        
        # Generate insights
        self.generate_summary_report(df)
        
        # Create visualizations
        print("\n🎨 Generating professional VADER dashboard...")
        self.create_sentiment_dashboard(df)
        
        # Save results
        output_file = 'bank_reviews_vader_analysis.csv'
        df.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to: {output_file}")
        print("🖼️  Dashboard saved as: vader_sentiment_dashboard.png")
        
        return df

# Execute the analysis
if __name__ == "__main__":
    analyzer = BankSentimentAnalyzer()
    results_df = analyzer.run_analysis('bank_reviews_clean.csv')