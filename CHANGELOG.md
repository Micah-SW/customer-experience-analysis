# Changelog
All major changes to this project will be documented here, following a structured
and transparent versioning convention.

## [1.4.0] - 2025-11-30
### Added
- Finalized **Interim Report** (condensed 4-page version).
- Added sentiment visualization images including:
  - `professional_sentiment_comparison.png`
  - `vader_sentiment_dashboard.png`
- Completed TF-IDF keyword extraction and early theme clustering.
- Added polished CSV files:
  - `bank_reviews_clean.csv`
  - `bank_reviews_vader_analysis.csv`
  - `reviews_with_sentiment.csv`
- Added scripts for advanced analysis:
  - `visualize_and_report.py`
  - `deeper_theme_analysis.py`

### Improved
- Major rewrite of interim report for clarity, structure, and professional tone.
- Strengthened theme grouping logic and mapped keywords to severity levels.
- Improved folder organization (`csv files/`, `images/`, `programs/`).
- Enhanced data validation for review cleaning pipeline.

---

## [1.3.0] - 2025-11-29
### Added
- Implemented **sentiment analysis pipeline** using VADER.
- Script added: `sentiment_analysis.py`.
- Generated `bank_reviews_vader_analysis.csv` containing:
  - compound scores
  - sentiment labels (Positive/Neutral/Negative)
  - cleaned text
- Added rating/sentiment correlation checks.

### Improved
- Improved preprocessing robustness:
  - better handling of non-English text segments
  - improved punctuation/emoji stripping
- Added safeguards to prevent misclassification of short reviews.
- Updated README to reflect sentiment analysis workflow.

---

## [1.2.0] - 2025-11-28
### Added
- Implemented **review preprocessing** pipeline:
  - deduplication
  - date normalization (YYYY-MM-DD)
  - whitespace & unicode cleanup
  - minimum-length filtering
  - Amharic/English heuristic tagging
- Script added: `preprocess_reviews.py`
- Exported dataset: `bank_reviews_clean.csv`

### Improved
- Updated repository documentation:
  - Added pipeline explanation to README
  - Included troubleshooting notes for scraped data issues

### Fixed
- Removed >2.6% duplicates detected across raw datasets.
- Fixed issue causing incorrect parsing of timestamps in certain review entries.

---

## [1.1.0] - 2025-11-27
### Added
- Implemented **scraping pipeline** (`scrape_reviews.py`):
  - fetched reviews for CBE, Dashen Bank, and Bank of Abyssinia
  - standardized to 400+ reviews per bank as required
  - stored raw dataset into `raw_reviews_20251130.csv`
- Added initial exploratory notebook and quick sentiment checks.

### Improved
- Set up Git repo structure following best practices.
- Documented scraping methodology and dataset fields in README.
- Added `requirements.txt` for reproducibility.

### Fixed
- Bug where some long reviews were truncated due to scraper default limits.

---

## [1.0.0] - 2025-11-26
### Added
- Initial project setup for **10 Academy Week 2 Challenge**.
- Created project directories:
  - `programs/`
  - `csv files/`
  - `images/`
- Added initial README with:
  - challenge description
  - objectives
  - planned tasks
- Environment created and initialized:
  - `.venv/`
  - initial dependency installation
- Started repo with:
  - `Analysis_Report.md`
  - base folder skeleton

---

## Legend
- **Added**: new functionality
- **Improved**: enhancements/refactors
- **Fixed**: bug fixes or corrections
- **Deprecated**: features to be removed (none yet)
- **Removed**: features deleted (none yet)

