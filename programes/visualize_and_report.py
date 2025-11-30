import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from nltk.corpus import stopwords
import time
import nltk

# NLTK setup (for consistency)
try:
    nltk.data.find('corpora/stopwords')
except:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords', quiet=True)

# Define the file paths
INPUT_FILE = 'reviews_with_sentiment.csv'
STOP_WORDS = set(stopwords.words('english'))

def get_top_keywords(df, sentiment_filter='Negative', top_n=10):
    """Extracts the most frequent non-stop words from a filtered set of reviews."""
    
    filtered_df = df[df['sentiment_label'] == sentiment_filter]
    
    if filtered_df.empty:
        return []
        
    all_text = ' '.join(filtered_df['review_text'].astype(str).str.lower())
    words = re.findall(r'\b[a-z]{3,}\b', all_text)
    
    # Enhanced list of custom stopwords to improve relevancy
    custom_stopwords = STOP_WORDS.union({'app', 'bank', 'mobile', 'get', 'can', 'will', 'like', 'one', 'new', 'need', 'know', 'user', 'make', 'use', 'try', 'time'})
    filtered_words = [word for word in words if word not in custom_stopwords and len(word) > 2]
    
    word_counts = Counter(filtered_words)
    return word_counts.most_common(top_n)

def plot_professional_comparison(df):
    """Generates a professional, minimalist stacked bar chart."""
    
    # Calculate sentiment percentages
    sentiment_counts = df.groupby('bank')['sentiment_label'].value_counts(normalize=True).mul(100).rename('percentage').reset_index()

    # Get total reviews for annotation
    total_counts = df['bank'].value_counts().reset_index()
    total_counts.columns = ['bank', 'total_reviews']

    # --- PLOT SETUP ---
    plt.style.use('default') 
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Custom, professional palette
    palette = {'Positive': '#00bfa5', # Success Green
               'Neutral': '#cccccc', # Light Gray
               'Negative': '#f44336'} # Error Red

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
        order=plot_order # Use the professional plot order
    )
    
    # --- AESTHETIC ENHANCEMENTS (Chart Junk Reduction) ---
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.set_xticks([]) # Remove X-axis ticks (we label directly)
    ax.set_xlabel('') 
    ax.set_ylabel('') 

    # --- FIXED AND ROBUST LABELING ---
    # Prepare data for robust internal labeling (Pivot the data)
    plot_data = sentiment_counts.pivot(index='bank', columns='sentiment_label', values='percentage').fillna(0)
    
    # Correctly extract the string labels from the Matplotlib Text objects
    # Note: We retrieve the labels *after* plotting to ensure they are in the correct order
    plotted_bank_labels = [t.get_text() for t in ax.get_yticklabels()]
    
    # Re-order the plot_data DataFrame rows to match the bar plot's (top-to-bottom) order
    plot_data = plot_data.loc[plotted_bank_labels[::-1]] 

    # 1. CENTER LABELING (inside the bar segment)
    # The iteration must be in reverse order to match the plotting position (index 0 is the bottom bar)
    for i, bank_name in enumerate(plot_data.index):
        current_x = 0 # Tracks the left edge of the segment for label positioning
        
        # Iterate through the sentiment segments for the current bank
        for sentiment in sentiment_order:
            percentage = plot_data.loc[bank_name, sentiment]
            
            # Only label non-zero segments large enough for text (e.g., > 2.5%)
            if percentage > 2.5: 
                label_x = current_x + (percentage / 2) # X position is the center of the segment
                label_y = i # Y position is the center of the bar
                
                # Use white text for the dark Negative segment for readability
                color = 'white' if sentiment == 'Negative' else 'black'
                
                ax.text(label_x, label_y, 
                        f'{percentage:.1f}%', 
                        ha='center', va='center', fontsize=9, color=color, weight='bold')
                        
            # Update the starting point for the next segment
            current_x += percentage

    # 2. TOTAL REVIEW COUNT (n= annotation on the far right)
    for i, bank_name in enumerate(plot_data.index):
        total_n = total_counts[total_counts['bank'] == bank_name]['total_reviews'].iloc[0]
        
        ax.text(102, # Far right of the chart
                i, # Center vertically
                f'(n={total_n})', 
                va='center', 
                ha='right', 
                fontsize=9, 
                color='gray')


    # Add custom Title and Legend
    ax.set_title('App Satisfaction: Positive Sentiment Comparison', fontsize=16, pad=20, loc='left', weight='bold')
    ax.legend(title='Sentiment', loc='upper left', bbox_to_anchor=(0.0, -0.05), ncol=3, frameon=False)
    ax.set_xlim(0, 105)

    plt.tight_layout()
    plt.savefig('professional_sentiment_comparison.png')
    print("\n✅ Final Professional Chart saved as 'professional_sentiment_comparison.png'")

# --- MAIN FUNCTION ---
def main():
    print("📈 Starting Visualization and Keyword Extraction...")
    start_time = time.time()
    
    # 1. Load Analyzed Data
    try:
        df = pd.read_csv(INPUT_FILE, low_memory=False)
    except FileNotFoundError:
        print(f"❌ Error: '{INPUT_FILE}' not found. Ensure sentiment analysis was run.")
        return

    # 2. Generate Professional Comparison Chart
    plot_professional_comparison(df)

    # 3. Extract Top Pain Points (from Negative Reviews)
    print("\n--- Top 10 Urgent Pain Points (from 510 Negative Reviews) ---")
    pain_points = get_top_keywords(df, sentiment_filter='Negative', top_n=10)

    if pain_points:
        for i, (word, count) in enumerate(pain_points):
            print(f"{i+1}. {word.capitalize()} (Mentions: {count})")
    
    end_time = time.time()
    print(f"\n⏱️ Total time for analysis and visualization: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()