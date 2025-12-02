# Changelog

All major changes to this project are documented here, following structured versioning conventions.

---

## [1.5.0] - 2025-12-02

### **Added**
- **Database Schema Implementation**: Created four core relational tables (banks, reviews, topics, topic_assignments) for analytical reporting
- **Final Analytical Reporting SQL**: Added comprehensive report file (`src/final_bank_reports.sql`) containing five definitive analytical queries
- **Project Summary Report**: Generated `PROJECT_SUMMARY.md` file compiling all analytical findings including bank rankings, health status, and urgent alerts
- **SQL Execution Script**: Added dependency `src/query_executor.py` for automated query execution

### **Improved**
- **Execution Reliability**: Refactored final query execution process to focus exclusively on pure analytical SELECT statements, achieving 100% success rate
- **Reporting Structure**: Consolidated key analytical insights (QoQ trends, Negative Review Share, Real-Time Alerts) for executive review
- **Code Optimization**: Streamlined project structure for better maintainability

### **Removed**
- **Complex DDL Components**: Removed multi-line DDL (CREATE FUNCTION, CREATE MATERIALIZED VIEW) that caused parser errors
- **Redundant Scripts**: Eliminated outdated scripts to focus on core analytical functionality

---

## [1.4.0] - 2025-11-30

### **Added**
- **Finalized Interim Report**: Created condensed 4-page professional version
- **Sentiment Visualizations**: Added professional sentiment analysis images:
  - `professional_sentiment_comparison.png`
  - `vader_sentiment_dashboard.png`
- **Advanced Analytics**: Completed TF-IDF keyword extraction and early theme clustering
- **Polished CSV Datasets**:
  - `bank_reviews_clean.csv`
  - `bank_reviews_vader_analysis.csv`
  - `reviews_with_sentiment.csv`
- **Advanced Analysis Scripts**:
  - `visualize_and_report.py`
  - `deeper_theme_analysis.py`

### **Improved**
- **Report Clarity**: Major rewrite of interim report for improved structure and professional tone
- **Theme Analysis**: Strengthened theme grouping logic with keyword-to-severity level mapping
- **Project Organization**: Enhanced folder structure (`csv files/`, `images/`, `programs/`)
- **Data Validation**: Improved review cleaning pipeline with robust validation

---

## [1.3.0] - 2025-11-29

### **Added**
- **Sentiment Analysis Pipeline**: Implemented VADER-based sentiment scoring
- **Analysis Script**: Added `sentiment_analysis.py` for automated sentiment processing
- **Enhanced Dataset**: Generated `bank_reviews_vader_analysis.csv` containing:
  - Compound sentiment scores
  - Sentiment labels (Positive/Neutral/Negative)
  - Cleaned review text
- **Quality Metrics**: Added rating/sentiment correlation checks

### **Improved**
- **Preprocessing Robustness**: Enhanced handling of non-English text segments
- **Text Cleaning**: Improved punctuation and emoji stripping
- **Classification Safeguards**: Added protection against misclassification of short reviews
- **Documentation**: Updated README with detailed sentiment analysis workflow

---

## [1.2.0] - 2025-11-28

### **Added**
- **Review Preprocessing Pipeline**: Implemented comprehensive data cleaning system:
  - Deduplication
  - Date normalization (YYYY-MM-DD format)
  - Whitespace and Unicode cleanup
  - Minimum-length filtering
  - Amharic/English language tagging
- **Processing Script**: Added `preprocess_reviews.py` for automated cleaning
- **Clean Dataset**: Exported validated dataset as `bank_reviews_clean.csv`

### **Improved**
- **Repository Documentation**: Updated README with detailed pipeline explanations
- **Troubleshooting**: Added notes for addressing scraped data issues
- **Data Quality**: Improved handling of various review formats

### **Fixed**
- **Duplicate Removal**: Eliminated >2.6% duplicates detected across raw datasets
- **Timestamp Parsing**: Fixed incorrect parsing of timestamps in certain review entries

---

## [1.1.0] - 2025-11-27

### **Added**
- **Scraping Pipeline**: Implemented `scrape_reviews.py` for automated data collection
- **Dataset Collection**: Fetched reviews for three major banks (CBE, Dashen Bank, Bank of Abyssinia)
- **Data Standardization**: Maintained consistent 400+ reviews per bank as required
- **Raw Dataset**: Stored initial data in `raw_reviews_20251130.csv`
- **Exploratory Analysis**: Added initial notebook with quick sentiment checks

### **Improved**
- **Repository Structure**: Set up Git repository following industry best practices
- **Methodology Documentation**: Documented scraping methodology and dataset fields in README
- **Reproducibility**: Added `requirements.txt` for dependency management

### **Fixed**
- **Review Truncation**: Resolved bug where long reviews were truncated due to scraper default limits
- **Data Collection**: Improved robustness of review fetching across different bank platforms

---

## [1.0.0] - 2025-11-26

### **Added**
- **Project Initialization**: Set up 10 Academy Week 2 Challenge project structure
- **Directory Organization**: Created core project directories:
  - `programs/`
  - `csv files/`
  - `images/`
- **Project Documentation**: Added initial README with:
  - Challenge description
  - Objectives and goals
  - Planned tasks and timeline
- **Development Environment**:
  - Created `.venv/` virtual environment
  - Installed initial dependencies
  - Set up Git repository
- **Initial Artifacts**:
  - `Analysis_Report.md` (preliminary report)
  - Base folder structure

---

## Legend

| Term | Meaning |
|------|---------|
| **Added** | New functionality, features, or components |
| **Improved** | Enhancements, optimizations, or refactors |
| **Fixed** | Bug fixes, corrections, or resolution of issues |
| **Deprecated** | Features marked for future removal (none yet) |
| **Removed** | Features or components deleted (none yet) |

---

## Versioning Convention

This project follows a structured versioning approach:

### **Version Format: MAJOR.MINOR.PATCH**
- **MAJOR**: Significant milestones or major feature additions
- **MINOR**: Important feature additions or improvements
- **PATCH**: Bug fixes or minor improvements

### **Release Frequency**
- Major releases: End of each significant project phase
- Minor releases: Upon completion of key deliverables
- Patch releases: As needed for bug fixes

---

**Last Updated**: December 2, 2025  
**Maintained By**: Mikiyas getnet 
**Document Version**: 1.5.0