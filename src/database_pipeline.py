import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import logging
from contextlib import contextmanager
from typing import Generator
import os
import time
from dotenv import load_dotenv # ⬅️ NEW: Import for loading environment variables

# 1. Load Environment Variables
# This call looks for a .env file in the current directory and loads its contents
# into the environment, making them accessible via os.getenv().
load_dotenv()

# Setup logging - Professional touch for tracking progress
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
# 2. Fetch credentials securely from environment variables (No more hardcoding!)
DB_CONFIG = {
    'user': os.getenv('POSTGRES_USER'),
    'password': os.getenv('POSTGRES_PASSWORD'),
    'host': os.getenv('POSTGRES_HOST'),
    'port': os.getenv('POSTGRES_PORT'),
    'database': os.getenv('POSTGRES_DB')
}

# Ensure all config variables were loaded
if any(value is None for value in DB_CONFIG.values()):
    # We check if required variables are None, which happens if they aren't in .env or environment
    logger.critical("🚨 One or more database environment variables are missing (User, Password, Host, Port, or DB name). Please check your .env file.")
    # Show what we failed to get for easier debugging
    missing_vars = [k for k, v in DB_CONFIG.items() if v is None]
    logger.critical(f"Missing variables: {missing_vars}")
    exit(1)

# Build the connection URI using the securely loaded variables
DATABASE_URI = (
    f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
)

# Resolve CSV file path relative to script location
def get_csv_path() -> str:
    """Resolve the CSV file path relative to the script location."""
    # Get the directory where this script is located (src/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up one level to project root, then into data folder
    project_root = os.path.dirname(script_dir)
    csv_path = os.path.join(project_root, 'data', 'bank_reviews_vader_analysis.csv')
    
    return csv_path

CSV_FILE = get_csv_path()

@contextmanager
def db_connection() -> Generator:
    """
    Context manager for database connections with connection pooling.
    Handles commit/rollback automatically.
    """
    # Uses the DATABASE_URI built from environment variables
    engine = create_engine(
        DATABASE_URI,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        pool_recycle=3600
    )
    connection = None
    try:
        connection = engine.connect()
        yield connection
        connection.commit() # Auto-commit on success
    except SQLAlchemyError as e:
        if connection:
            connection.rollback() # Auto-rollback on error
        logger.error(f"Database transaction error: {e}")
        raise
    finally:
        if connection:
            connection.close()

def create_schema():
    """Creates the relational schema (Tables: banks, reviews) with Foreign Key."""
    logger.info("⏳ Preparing PostgreSQL schema...")
    
    with db_connection() as conn:
        # Drop tables in safe order (reviews first due to FK constraint)
        conn.execute(text("DROP TABLE IF EXISTS reviews CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS banks CASCADE;"))
        
        # 1. Create Banks Table (Parent Table)
        conn.execute(text("""
            CREATE TABLE banks (
                bank_id SERIAL PRIMARY KEY,
                bank_name VARCHAR(100) UNIQUE NOT NULL
            );
        """))
        
        # 2. Create Reviews Table (Child Table with Foreign Key)
        conn.execute(text("""
            CREATE TABLE reviews (
                review_id SERIAL PRIMARY KEY,
                bank_id INTEGER REFERENCES banks(bank_id) ON DELETE CASCADE,
                review_text TEXT NOT NULL,
                rating INTEGER,
                sentiment_label VARCHAR(20),
                sentiment_score FLOAT,
                review_date DATE, 
                source VARCHAR(50) DEFAULT 'Google Play Store'
            );
        """))
    logger.info("✅ Schema created successfully with FK relationship.")

def load_data_bulk(csv_file: str = None):
    """Load data using bulk operations and chunking for performance."""
    # Use provided csv_file or the resolved one
    if csv_file is None:
        csv_file = CSV_FILE
    
    # Log the path we're checking
    logger.info(f"Looking for CSV file at: {csv_file}")
    
    if not os.path.exists(csv_file):
        logger.error(f"❌ File not found: {csv_file}")
        logger.error("Please ensure:")
        logger.error("1. The CSV file exists in the data/ folder")
        logger.error("2. The file name is exactly: bank_reviews_vader_analysis.csv")
        logger.error(f"3. Current working directory: {os.getcwd()}")
        return
    
    try:
        # Log file size and metadata
        file_size = os.path.getsize(csv_file) / (1024 * 1024)  # MB
        logger.info(f"📊 Found CSV file: {csv_file} ({file_size:.2f} MB)")
        
        # Load data in chunks for memory efficiency (Scalability KPI)
        chunk_size = 5000
        chunks = pd.read_csv(csv_file, chunksize=chunk_size)
        total_rows = 0
        
        # We need to process banks first to build the Foreign Key map
        # Strategy: Read unique banks from the FULL file first (lightweight), then process chunks.
        
        # 1. Pre-process Banks
        logger.info("🏦 Processing bank entities...")
        full_df_banks = pd.read_csv(csv_file, usecols=['bank'])
        unique_banks = full_df_banks['bank'].unique()
        logger.info(f"Found {len(unique_banks)} unique banks: {list(unique_banks)}")
        
        with db_connection() as conn:
            # Bulk insert banks
            bank_records = [{'bank_name': bank} for bank in unique_banks]
            if bank_records:
                for bank_record in bank_records:
                    conn.execute(
                        text("INSERT INTO banks (bank_name) VALUES (:bank_name) ON CONFLICT (bank_name) DO NOTHING"),
                        bank_record
                    )
                logger.info(f"✅ Inserted/updated {len(bank_records)} banks")
            
            # Get Bank IDs mapping
            result = conn.execute(text("SELECT bank_name, bank_id FROM banks"))
            bank_map = {row.bank_name: row.bank_id for row in result}
            logger.info(f"📋 Bank ID mapping created with {len(bank_map)} entries")
        
        # 2. Process Reviews in Chunks
        logger.info("📝 Starting bulk review insertion...")
        
        # Re-initialize chunk iterator
        chunks = pd.read_csv(csv_file, chunksize=chunk_size)
        
        with db_connection() as conn:
            chunk_count = 0
            for chunk in chunks:
                chunk_count += 1
                logger.info(f"Processing chunk {chunk_count}...")
                
                # Prepare chunk data
                # Handle date conversion safely
                try:
                    chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce').dt.date
                except Exception as e:
                    logger.warning(f"Date conversion issue in chunk {chunk_count}: {e}")
                    # Create a dummy date column if conversion fails
                    chunk['date'] = pd.Timestamp.now().date()
                
                # Clean review text
                chunk = chunk.dropna(subset=['review_text'])
                if len(chunk) == 0:
                    logger.warning(f"Skipping empty chunk {chunk_count}")
                    continue
                    
                # Fill missing ratings with 0 and convert to int
                chunk['rating'] = chunk['rating'].fillna(0).astype(int)
                
                # Map bank names to bank IDs
                chunk['bank_id'] = chunk['bank'].map(bank_map)
                
                # Filter out rows where bank mapping failed
                chunk = chunk.dropna(subset=['bank_id'])
                chunk['bank_id'] = chunk['bank_id'].astype(int)
                
                if len(chunk) == 0:
                    logger.warning(f"All rows in chunk {chunk_count} had invalid bank mapping")
                    continue
                
                # Filter and Rename for SQL
                db_chunk = chunk[['bank_id', 'review_text', 'rating', 'sentiment_label', 'sentiment_score', 'date', 'source']].copy()
                db_chunk.rename(columns={'date': 'review_date'}, inplace=True)
                
                # Bulk Insert using pandas to_sql
                db_chunk.to_sql('reviews', conn, if_exists='append', index=False, method='multi')
                total_rows += len(db_chunk)
                logger.info(f"✅ Inserted chunk {chunk_count} with {len(db_chunk)} rows (Total: {total_rows})")

        logger.info(f"🎉 Successfully inserted {total_rows} total reviews into PostgreSQL.")
        
        # Create indexes for better query performance
        with db_connection() as conn:
            logger.info("🔧 Creating performance indexes...")
            conn.execute(text("CREATE INDEX idx_reviews_bank_id ON reviews(bank_id);"))
            conn.execute(text("CREATE INDEX idx_reviews_sentiment_score ON reviews(sentiment_score);"))
            conn.execute(text("CREATE INDEX idx_reviews_review_date ON reviews(review_date);"))
            conn.execute(text("CREATE INDEX idx_reviews_rating ON reviews(rating);"))
            logger.info("✅ Indexes created for optimal query performance")
        
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}", exc_info=True)
        raise

def verify_database_data():
    """Verify the data was loaded correctly."""
    try:
        with db_connection() as conn:
            # Check banks table
            bank_count = conn.execute(text("SELECT COUNT(*) FROM banks")).scalar()
            logger.info(f"📊 Banks table has {bank_count} records")
            
            # Check reviews table
            review_count = conn.execute(text("SELECT COUNT(*) FROM reviews")).scalar()
            logger.info(f"📊 Reviews table has {review_count} records")
            
            # Check by bank
            result = conn.execute(text("""
                SELECT b.bank_name, COUNT(r.review_id) as review_count,
                       AVG(r.sentiment_score) as avg_sentiment
                FROM banks b
                LEFT JOIN reviews r ON b.bank_id = r.bank_id
                GROUP BY b.bank_name
                ORDER BY review_count DESC
            """))
            
            logger.info("📋 Review distribution by bank:")
            for row in result:
                logger.info(f"   {row.bank_name}: {row.review_count} reviews, avg sentiment: {row.avg_sentiment:.3f}")
                
    except Exception as e:
        logger.error(f"Verification error: {e}")

if __name__ == "__main__":
    # Log the CSV file path we're using
    logger.info(f"📁 Using CSV file: {CSV_FILE}")
    
    # Check if CSV exists before attempting anything
    if not os.path.exists(CSV_FILE):
        logger.error(f"❌ CSV file not found at: {CSV_FILE}")
        logger.error("Please ensure:")
        logger.error("1. Your project structure is: week2_challenge/data/bank_reviews_vader_analysis.csv")
        logger.error("2. You're running this from the week2_challenge directory")
        logger.error(f"3. Current directory: {os.getcwd()}")
        exit(1)
    
    # Retry logic included in main execution flow
    max_retries = 3
    success = False
    
    for attempt in range(max_retries):
        try:
            logger.info(f"🚀 Attempt {attempt + 1} of {max_retries}")
            
            # Test database connection first
            logger.info("🔌 Testing database connection...")
            with db_connection() as conn:
                result = conn.execute(text("SELECT version()"))
                db_version = result.fetchone()[0]
                logger.info(f"✅ Connected to PostgreSQL: {db_version[:50]}...")
            
            # Create schema
            create_schema()
            
            # Load data
            load_data_bulk()
            
            # Verify data
            verify_database_data()
            
            success = True
            break
            
        except Exception as e:
            logger.warning(f"⚠️ Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                logger.info(f"⏳ Retrying in {2 ** attempt} seconds...")  # Exponential backoff
                time.sleep(2 ** attempt)
            else:
                logger.critical("🚨 Pipeline failed after max retries.")
    
    if success:
        logger.info("🎉 Database pipeline completed successfully!")
    else:
        exit(1)