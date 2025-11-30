import pandas as pd
import glob

# 1. Load the most recent raw csv
# (This finds the csv we just made)
list_of_files = glob.glob('raw_reviews_*.csv') 
latest_file = max(list_of_files, key=lambda x: x)
print(f"🧹 Processing file: {latest_file}")

df = pd.read_csv(latest_file)

# 2. Preprocessing Logic
original_count = len(df)

# A. Drop duplicates based on reviewId
df.drop_duplicates(subset=['reviewId'], inplace=True)

# B. Handle missing content
# We drop rows where 'content' (the review text) is missing
df.dropna(subset=['content'], inplace=True)

# C. Normalize Dates [cite: 82]
# Google Play scraper usually gives a datetime object, but pandas ensures it's standard
df['at'] = pd.to_datetime(df['at'])
df['date'] = df['at'].dt.strftime('%Y-%m-%d') # KPI Requirement

# D. Select and Rename Columns for the final schema [cite: 83]
# We map the scraper columns to the challenge requirements
clean_df = df[['content', 'score', 'date', 'bank', 'source', 'reviewId']].copy()
clean_df.rename(columns={
    'content': 'review_text',
    'score': 'rating',
    'reviewId': 'review_id'
}, inplace=True)

# 3. Quality Check (KPI Verification)
print("\n--- Data Quality Check ---")
print(f"Original Count: {original_count}")
print(f"Clean Count: {len(clean_df)}")
missing_percent = clean_df.isnull().mean().max() * 100
print(f"Missing Data: {missing_percent:.2f}% (KPI: <5%)")

# 4. Save Clean Data
clean_df.to_csv('bank_reviews_clean.csv', index=False)
print("✅ Cleaned data saved to 'bank_reviews_clean.csv'")