import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time

# --- Configuration ---
INPUT_FILE = 'bank_reviews_clean.csv'
OUTPUT_FILE = 'reviews_with_sentiment.csv'
SENTIMENT_THRESHOLD = 0.05 # Standard VADER threshold

# 1. Setup VADER and NLTK Resources (FIXED FOR ERROR HANDLING)
def setup_vader():
    """Initializes VADER and downloads the lexicon if needed."""
    try:
        # Simple download call. NLTK handles if it's already present.
        print("Ensuring VADER lexicon is downloaded...")
        nltk.download('vader_lexicon', quiet=True) 
    except Exception as e:
        # Catching a general exception is safer here to prevent the AttributeError
        print(f"Warning: Automatic NLTK download failed but continuing. Error: {e}")
    
    return SentimentIntensityAnalyzer()

# 2. Optimized Sentiment Classification Function
def classify_sentiment(score):
    """Classifies sentiment based on the VADER compound score using a single check."""
    if score >= SENTIMENT_THRESHOLD:
        return "Positive"
    elif score <= -SENTIMENT_THRESHOLD:
        return "Negative"
    else:
        return "Neutral"

# 3. Main Analysis Function
def run_analysis():
    print(f"🚀 Starting VADER Sentiment Analysis on {INPUT_FILE}...")
    start_time = time.time()
    
    # Load Clean Data
    try:
        df = pd.read_csv(INPUT_FILE, low_memory=False)
        print(f"Loaded {len(df)} records.")
    except FileNotFoundError:
        print(f"❌ Error: '{INPUT_FILE}' not found. Ensure preprocessing was run.")
        return

    # Initialize VADER Analyzer
    analyzer = setup_vader()

    # Apply VADER (Optimized for performance)
    df['polarity_scores'] = df['review_text'].astype(str).apply(analyzer.polarity_scores)
    
    # Extract compound score in a vectorized way
    df['vader_compound_score'] = df['polarity_scores'].apply(lambda x: x['compound'])

    # Classify sentiment labels (using the vectorized score column)
    df['sentiment_label'] = df['vader_compound_score'].apply(classify_sentiment)

    # Drop the temporary 'polarity_scores' column
    df.drop(columns=['polarity_scores'], inplace=True)

    # 4. Save Final Analyzed Data
    df.to_csv(OUTPUT_FILE, index=False)
    
    end_time = time.time()
    
    print("\n--- Sentiment Summary per Bank (VADER) ---")
    summary = df.groupby(['bank', 'sentiment_label']).size().unstack(fill_value=0)
    print(summary)
    
    print(f"\n✅ Analysis complete. Saved {len(df)} records to '{OUTPUT_FILE}'")
    print(f"⏱️ Time taken: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    run_analysis()