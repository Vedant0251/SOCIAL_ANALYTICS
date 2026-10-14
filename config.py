"""BrandPulse AI - Central Workspace Configuration.

Defines all paths, verified Kaggle dataset slugs, SNAP graph endpoints,
model identifiers, random seeds, hardware policies, and aspect mappings.
Follows Rule 4 (Security) by reading secrets exclusively from environment
variables or .env files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

# Load local environment if available
load_dotenv()

# =====================================================================
# 1. DIRECTORY STRUCTURE & PATHS
# =====================================================================
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_MANUAL_ZIPS_DIR = DATA_DIR / "manual_zips"
DATA_MANUAL_LABELS_DIR = DATA_DIR / "manual_labels"

CACHE_DIR = PROJECT_ROOT / "cache"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
DATA_GAPS_FILE = REPORTS_DIR / "data_gaps.md"
DOCS_DIR = PROJECT_ROOT / "docs"

SCRAPY_DIR = PROJECT_ROOT / "scrapy_project"
MANIFEST_FILE = DATA_RAW_DIR / "MANIFEST.json"

for directory in [
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    DATA_MANUAL_ZIPS_DIR,
    DATA_MANUAL_LABELS_DIR,
    CACHE_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    DOCS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# =====================================================================
# 2. VERIFIED KAGGLE SLUGS & DATASET SCHEMAS (Rule 2)
# =====================================================================
# Slugs are confirmed via Kaggle CLI: `kaggle datasets list -s ...`
KAGGLE_DATASETS: dict[str, dict[str, Any]] = {
    "airline_sentiment": {
        "slug": "crowdflower/twitter-airline-sentiment",
        "primary_file": "Tweets.csv",
        "license": "CC BY-NC-SA 4.0",
        "description": "Twitter US Airline Sentiment dataset with tweet text, sentiments, reasons, coordinates, and timestamps.",
        "expected_columns": [
            "tweet_id",
            "airline_sentiment",
            "airline_sentiment_confidence",
            "negativereason",
            "negativereason_confidence",
            "airline",
            "text",
            "tweet_coord",
            "tweet_created",
            "tweet_location",
            "user_timezone",
            "retweet_count",
        ],
    },
    "amazon_reviews": {
        "slug": "arhamrumi/amazon-product-reviews",
        "primary_file": "Reviews.csv",
        "license": "CC0: Public Domain",
        "description": "Amazon fine food and consumer product reviews with text, helpfulness, and star ratings.",
        "expected_columns": ["Id", "ProductId", "UserId", "Score", "Time", "Summary", "Text"],
    },
    "shopping_apps": {
        "slug": "jocelyndumlao/shoppingappreviews-dataset",
        "primary_file": "ShoppingAppReviews Dataset/ShoppingAppReviews/csv/Amazon shopping.csv",
        "secondary_file": "ShoppingAppReviews Dataset/ShoppingAppReviews/csv/Myntra.csv",
        "license": "CC0: Public Domain",
        "description": "Comparative e-commerce app reviews including Amazon and Myntra.",
        "expected_columns": ["userName", "content", "score"],
    },
    "indian_airlines": {
        "slug": "jagathratchakan/indian-airlines-customer-reviews",
        "primary_file": "Indian_Domestic_Airline.csv",
        "license": "Unknown / Community",
        "description": "Customer reviews for Indian domestic carriers including IndiGo, Air India, and SpiceJet.",
        "expected_columns": ["Airline", "Review", "Rating"],
    },
    "phone_reviews": {
        "slug": "grikomsn/amazon-cell-phones-reviews",
        "primary_file": "20191226-reviews.csv",
        "items_file": "20191226-items.csv",
        "license": "CC0: Public Domain",
        "description": "Cell phone consumer reviews comparing Apple iPhone and Samsung Galaxy brands.",
        "expected_columns": ["asin", "name", "rating", "title", "body"],
    },
    "cyberbullying": {
        "slug": "saurabhshahane/cyberbullying-dataset",
        "primary_file": "twitter_parsed_dataset.csv",
        "toxicity_file": "toxicity_parsed_dataset.csv",
        "license": "CC0: Public Domain",
        "description": "Multi-class tweet cyberbullying and toxicity dataset for rigorous NLP benchmark evaluation.",
        "expected_columns": ["Text", "oh_label"],
    },
    "website_traffic": {
        "slug": "thedevastator/the-top-websites-in-the-world",
        "primary_file": "df_1.csv",
        "license": "Open Database License (ODbL)",
        "description": "Top visited global domains and web traffic metrics for competitor bench analytics.",
        "expected_columns": ["Domain", "Monthly Visits"],
    },
    "hospital_reviews": {
        "slug": "junaid6731/hospital-reviews-dataset",
        "primary_file": "hospital.csv",
        "license": "CC0: Public Domain",
        "description": "Hospital customer feedback and healthcare satisfaction ratings.",
        "expected_columns": ["Review", "Rating"],
    },
}

# =====================================================================
# 3. SNAP NETWORK DATASETS
# =====================================================================
SNAP_BASE_URL = "https://snap.stanford.edu/data"
SNAP_DATASETS: dict[str, dict[str, str]] = {
    "twitter_combined": {
        "url": f"{SNAP_BASE_URL}/twitter_combined.txt.gz",
        "filename": "twitter_combined.txt.gz",
        "type": "ego-network",
        "description": "Stanford SNAP Twitter ego-networks combined graph (edges: 1,768,149, nodes: 81,306).",
    },
    "higgs_retweet": {
        "url": f"{SNAP_BASE_URL}/higgs-retweet_network.edgelist.gz",
        "filename": "higgs-retweet_network.edgelist.gz",
        "type": "retweet-network",
        "description": "Higgs Twitter dataset retweet network during discovery announcement.",
    },
    "higgs_mention": {
        "url": f"{SNAP_BASE_URL}/higgs-mention_network.edgelist.gz",
        "filename": "higgs-mention_network.edgelist.gz",
        "type": "mention-network",
        "description": "Higgs Twitter dataset user-to-user mention network.",
    },
}

# =====================================================================
# 4. COMPETITOR BENCHMARK PAIRS
# =====================================================================
COMPETITOR_PAIRS: list[dict[str, Any]] = [
    {
        "pair_id": "ecommerce",
        "entity_a": "Amazon",
        "entity_b": "Myntra",
        "category": "E-Commerce",
        "dataset_key": "shopping_apps",
        "notes": "Verified real app review data from ShoppingAppReviews dataset.",
    },
    {
        "pair_id": "aviation_india",
        "entity_a": "IndiGo",
        "entity_b": "Air India",
        "category": "Aviation (India)",
        "dataset_key": "indian_airlines",
        "notes": "Real Indian Domestic Airlines reviews; US Crowdflower airline data is strictly segregated.",
    },
    {
        "pair_id": "tech_giants",
        "entity_a": "Apple",
        "entity_b": "Samsung",
        "category": "Consumer Tech",
        "dataset_key": "phone_reviews",
        "notes": "Real Amazon smartphone reviews partitioned by brand entity.",
    },
    {
        "pair_id": "healthcare",
        "entity_a": "Mayo Clinic",
        "entity_b": "Johns Hopkins",
        "category": "Healthcare",
        "dataset_key": "hospital_reviews",
        "notes": "Healthcare reviews dataset; entity filtering applies if present, else logged as data gap.",
    },
    {
        "pair_id": "higher_ed",
        "entity_a": "MIT",
        "entity_b": "Stanford",
        "category": "Higher Education",
        "dataset_key": None,
        "notes": "Subject to data availability search; flagged in data gaps if no organic reviews found.",
    },
]

# =====================================================================
# 5. MACHINE LEARNING & HARDWARE SETTINGS (Rule 5 & 6)
# =====================================================================
RANDOM_SEED = 42

SENTIMENT_MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
ZERO_SHOT_MODEL_ID = "facebook/bart-large-mnli"
ABSA_MODEL_ID = "yangheng/deberta-v3-base-absa-v1.1"

# Batch sizing with dynamic OOM fallback
INFERENCE_BATCH_SIZES = [32, 16, 8, 4, 1]
USE_FP16 = True

# Aspect Definition Taxonomy for Social Brand Monitoring
ASPECT_TAXONOMY: list[str] = [
    "Logistics/Delivery",
    "Customer Support",
    "Product/Service Quality",
    "Price/Value",
    "Flight Delays",
    "Baggage",
    "Booking/Cancellations",
    "Cabin Experience",
]

# Mapping proxy negative reasons from US Airlines dataset to aspects
NEGATIVE_REASON_ASPECT_MAP: dict[str, str] = {
    "Bad Flight": "Cabin Experience",
    "Can't Tell": "Customer Support",
    "Late Flight": "Flight Delays",
    "Customer Service Issue": "Customer Support",
    "Flight Booking Problems": "Booking/Cancellations",
    "Lost Luggage": "Baggage",
    "Flight Attendant Complaints": "Cabin Experience",
    "Cancelled Flight": "Booking/Cancellations",
    "Damaged Luggage": "Baggage",
    "longlines": "Customer Support",
}

# =====================================================================
# 6. EXTERNAL API CONFIGURATIONS
# =====================================================================
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "BrandPulseAI/1.0")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")

HACKER_NEWS_API_BASE = "https://hacker-news.firebaseio.com/v0"
WIKIPEDIA_PAGEVIEWS_BASE = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
LEGAL_SCRAPING_TARGETS = [
    "https://books.toscrape.com",
    "https://quotes.toscrape.com",
]
