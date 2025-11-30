import pandas as pd
from google_play_scraper import Sort, reviews_all, reviews
from datetime import datetime

# 1. Define our targets (Bank Name and App ID)
apps = [
    {"bank": "Commercial Bank of Ethiopia", "id": "com.combanketh.mobilebanking"},
    {"bank": "Bank of Abyssinia", "id": "com.boa.boaMobileBanking"},
    {"bank": "Dashen Bank", "id": "com.dashen.dashensuperapp"}, # Confirming Dashen's ID is critical, using standard ID
]

all_reviews = []

print(f"🚀 Starting Scraping process at {datetime.now()}")

# 2. Loop through each bank and scrape
for app in apps:
    print(f"--- Scraping {app['bank']} ---")
    
    # We use 'reviews' (fetches a specific count) rather than 'reviews_all' to be faster for the interim
    # To EXCEED KPI: We aim for 2000, ensuring we have >400 clean ones easily.
    result, continuation_token = reviews(
        app['id'],
        lang='en', # English reviews
        country='us', 
        sort=Sort.NEWEST, # Get the latest data
        count=2000, 
    )
    
    # Tag the data with the bank name (Crucial for analysis later)
    for r in result:
        r['bank'] = app['bank']
        r['source'] = 'Google Play'
    
    all_reviews.extend(result)
    print(f"✅ Collected {len(result)} reviews for {app['bank']}")

# 3. Convert to DataFrame
df = pd.DataFrame(all_reviews)

# 4. Save Raw Data
filename = f"raw_reviews_{datetime.now().strftime('%Y%m%d')}.csv"
df.to_csv(filename, index=False)

print(f"\n🎉 Success! Total reviews scraped: {len(df)}")
print(f"💾 Saved raw data to: {filename}")