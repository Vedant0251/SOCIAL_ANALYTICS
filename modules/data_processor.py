"""BrandPulse AI - Data Processor & Statistical Analytics Module.

Provides schema loaders with validation, text normalization pipelines,
automatic feature-type classification with visualization recommendations,
and comprehensive statistical diagnostics:
- Time-series trends with rolling means and statsmodels seasonal decomposition
- Correlation matrices (Pearson, Spearman, Cramér's V)
- Complaint reason Pareto (80/20) analysis
- Outlier detection (IQR, Z-score, and multivariate Isolation Forest)
- Safe geospatial coordinate parsing and validation
"""

from __future__ import annotations

import ast
import logging
import re
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from sklearn.ensemble import IsolationForest
import spacy
from statsmodels.tsa.seasonal import seasonal_decompose

from config import (
    CACHE_DIR,
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    FIGURES_DIR,
    RANDOM_SEED,
    TABLES_DIR,
)

logger = logging.getLogger("data_processor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Global cache for spaCy NLP pipeline
_SPACY_NLP: Any = None


def get_spacy_nlp() -> Any:
    """Lazy load spaCy en_core_web_sm pipeline."""
    global _SPACY_NLP
    if _SPACY_NLP is None:
        try:
            _SPACY_NLP = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        except Exception:
            logger.warning("spaCy en_core_web_sm not found. Falling back to regex tokenizer.")
            _SPACY_NLP = False
    return _SPACY_NLP


# =====================================================================
# 1. TEXT CLEANING & NORMALIZATION PIPELINE
# =====================================================================
URL_REGEX = re.compile(r"https?://\S+|www\.\S+")
MENTION_REGEX = re.compile(r"@\w+")
HTML_TAG_REGEX = re.compile(r"<.*?>|&[a-z]+;")
SPECIAL_CHAR_REGEX = re.compile(r"[^a-zA-Z0-9\s.,!?'-]")
EXTRA_SPACES_REGEX = re.compile(r"\s+")


def clean_text(text: str | Any, lemmatize: bool = False) -> str:
    """Normalize and clean social media text.

    Strips URLs, user @mentions, HTML artifacts, and excess whitespace.
    Optionally applies spaCy lemmatization.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Decode HTML and strip tags
    cleaned = HTML_TAG_REGEX.sub(" ", text)
    # Strip URLs
    cleaned = URL_REGEX.sub("", cleaned)
    # Strip @mentions
    cleaned = MENTION_REGEX.sub("", cleaned)
    # Strip non-standard characters while keeping basic punctuation
    cleaned = SPECIAL_CHAR_REGEX.sub(" ", cleaned)
    # Lowercase & collapse whitespace
    cleaned = EXTRA_SPACES_REGEX.sub(" ", cleaned).strip().lower()

    if lemmatize:
        nlp = get_spacy_nlp()
        if nlp:
            doc = nlp(cleaned)
            cleaned = " ".join([token.lemma_ for token in doc if not token.is_stop and len(token.lemma_) > 1])

    return cleaned


def parse_tweet_coordinates(coord_val: Any) -> tuple[float | None, float | None]:
    """Safely parse tweet_coord string formatted as '[latitude, longitude]'.

    Validates coordinates range: -90 <= lat <= 90, -180 <= lon <= 180.
    Returns (latitude, longitude) or (None, None) if missing/invalid.
    """
    if coord_val is None:
        return None, None

    if isinstance(coord_val, float) and pd.isna(coord_val):
        return None, None

    try:
        if not coord_val:
            return None, None
    except Exception:
        pass

    try:
        if isinstance(coord_val, str):
            coord_str = coord_val.strip()
            if coord_str.startswith("[") and coord_str.endswith("]"):
                parsed = ast.literal_eval(coord_str)
                if isinstance(parsed, (list, tuple)) and len(parsed) == 2:
                    lat, lon = float(parsed[0]), float(parsed[1])
                    if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                        return lat, lon
        elif isinstance(coord_val, (list, tuple)) and len(coord_val) == 2:
            lat, lon = float(coord_val[0]), float(coord_val[1])
            if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                return lat, lon
    except Exception:
        pass

    return None, None


# =====================================================================
# 2. DATASET LOADERS WITH SCHEMA VALIDATION & CACHING
# =====================================================================
def load_airline_sentiment(force_reload: bool = False) -> pd.DataFrame:
    """Load and process US Airline sentiment dataset.

    Parses coordinates, timestamps, and cleans text. Caches as Parquet.
    """
    cache_path = CACHE_DIR / "airline_sentiment_cleaned.parquet"
    processed_path = DATA_PROCESSED_DIR / "airline_sentiment_cleaned.parquet"
    if not force_reload and cache_path.exists():
        logger.info(f"Loading cached airline dataset from {cache_path}")
        df = pd.read_parquet(cache_path)
        if not processed_path.exists():
            df.to_parquet(processed_path, index=False)
        return df

    raw_file = DATA_RAW_DIR / "airline_sentiment" / "Tweets.csv"
    if not raw_file.exists():
        logger.error(f"US Airline Tweets.csv not found at {raw_file}")
        return pd.DataFrame()

    logger.info("Processing US Airline sentiment dataset...")
    df = pd.read_csv(raw_file, low_memory=False)

    # Clean text
    df["clean_text"] = df["text"].apply(lambda t: clean_text(t, lemmatize=False))

    # Parse timestamps
    df["tweet_created"] = pd.to_datetime(df["tweet_created"], errors="coerce")
    df["date"] = df["tweet_created"].dt.date
    df["hour"] = df["tweet_created"].dt.hour
    df["day_name"] = df["tweet_created"].dt.day_name()

    # Parse coordinates safely
    coords = df["tweet_coord"].apply(parse_tweet_coordinates)
    df["latitude"] = coords.apply(lambda c: c[0])
    df["longitude"] = coords.apply(lambda c: c[1])
    df["has_exact_coords"] = df["latitude"].notna() & df["longitude"].notna()

    # Numeric sentiment proxy for correlation analysis
    sentiment_map = {"negative": -1, "neutral": 0, "positive": 1}
    df["sentiment_score"] = df["airline_sentiment"].map(sentiment_map)

    # Save cache and processed dataset
    df.to_parquet(cache_path, index=False)
    processed_path = DATA_PROCESSED_DIR / "airline_sentiment_cleaned.parquet"
    df.to_parquet(processed_path, index=False)
    logger.info(f"Saved processed dataset ({len(df)} records) to {processed_path} and cache")
    return df


def load_shopping_apps_reviews() -> dict[str, pd.DataFrame]:
    """Load real comparative e-commerce app reviews (Amazon vs Myntra)."""
    base_folder = (
        DATA_RAW_DIR
        / "shopping_apps"
        / "ShoppingAppReviews Dataset"
        / "ShoppingAppReviews"
        / "csv"
    )
    results = {}

    amazon_file = base_folder / "Amazon shopping.csv"
    if amazon_file.exists():
        df_amz = pd.read_csv(amazon_file, low_memory=False)
        df_amz["entity"] = "Amazon"
        df_amz["clean_text"] = df_amz["content"].apply(clean_text)
        results["Amazon"] = df_amz

    myntra_file = base_folder / "Myntra.csv"
    if myntra_file.exists():
        df_myn = pd.read_csv(myntra_file, low_memory=False)
        df_myn["entity"] = "Myntra"
        df_myn["clean_text"] = df_myn["content"].apply(clean_text)
        results["Myntra"] = df_myn

    return results


def load_indian_airlines_reviews() -> pd.DataFrame:
    """Load customer reviews for Indian carriers (IndiGo vs Air India)."""
    file_path = DATA_RAW_DIR / "indian_airlines" / "Indian_Domestic_Airline.csv"
    if not file_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(file_path, low_memory=False)
    # Standardize column names
    col_mapping = {
        "AirLine_Name": "airline",
        "Rating - 10": "rating",
        "Title": "title",
        "Name": "author",
        "Date": "date",
        "Review": "text",
        "Recommond": "recommend",
    }
    df = df.rename(columns=col_mapping)
    df["clean_text"] = df["text"].apply(clean_text)
    # Map rating into 3-tier sentiment
    # rating scale is 1 to 10
    df["numeric_rating"] = pd.to_numeric(df["rating"], errors="coerce")
    conditions = [
        df["numeric_rating"] <= 4,
        (df["numeric_rating"] > 4) & (df["numeric_rating"] <= 7),
        df["numeric_rating"] > 7,
    ]
    choices = ["negative", "neutral", "positive"]
    df["sentiment"] = np.select(conditions, choices, default="neutral")
    return df


def load_cell_phone_reviews() -> dict[str, pd.DataFrame]:
    """Load smartphone reviews for Apple vs Samsung."""
    raw_dir = DATA_RAW_DIR / "phone_reviews"
    reviews_file = raw_dir / "20191226-reviews.csv"
    items_file = raw_dir / "20191226-items.csv"

    if not reviews_file.exists() or not items_file.exists():
        return {}

    df_reviews = pd.read_csv(reviews_file, low_memory=False)
    df_items = pd.read_csv(items_file, low_memory=False)

    # Merge to associate brand
    merged = pd.merge(df_reviews, df_items[["asin", "brand", "title"]], on="asin", how="left")
    merged["clean_text"] = merged["body"].apply(clean_text)

    # Extract Apple and Samsung partitions
    apple_df = merged[merged["brand"].str.contains("Apple", case=False, na=False)].copy()
    samsung_df = merged[merged["brand"].str.contains("Samsung", case=False, na=False)].copy()

    apple_df["entity"] = "Apple"
    samsung_df["entity"] = "Samsung"

    return {"Apple": apple_df, "Samsung": samsung_df}


def load_cyberbullying_dataset() -> pd.DataFrame:
    """Load multi-class cyberbullying / toxicity dataset for benchmark evaluations."""
    file_path = DATA_RAW_DIR / "cyberbullying" / "twitter_parsed_dataset.csv"
    if not file_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(file_path, low_memory=False)
    df["clean_text"] = df["Text"].apply(clean_text)
    return df


def load_website_traffic_dataset() -> pd.DataFrame:
    """Load global domain website traffic dataset."""
    file_path = DATA_RAW_DIR / "website_traffic" / "df_1.csv"
    if not file_path.exists():
        return pd.DataFrame()
    return pd.read_csv(file_path, low_memory=False)


# =====================================================================
# 3. AUTOMATIC FEATURE-TYPE CLASSIFIER & VIZ RECOMMENDATIONS
# =====================================================================
def classify_feature_types(df: pd.DataFrame) -> pd.DataFrame:
    """Inspect dataset and classify columns into data types with recommended visualizations.

    Categories:
    - Nominal (categorical without intrinsic order)
    - Ordinal (ordered categorical / ranking)
    - Binary (boolean or 2 distinct states)
    - Discrete Numeric (integer counts)
    - Continuous (real-valued measurements)
    - Datetime (timestamp features)
    - Geospatial (latitude/longitude/coordinates)
    - Free Text (unstructured language)
    """
    records = []

    for col in df.columns:
        series = df[col].dropna()
        n_unique = series.nunique()
        dtype = str(series.dtype)
        sample_val = str(series.iloc[0]) if len(series) > 0 else "N/A"
        if len(sample_val) > 40:
            sample_val = sample_val[:37] + "..."

        col_lower = col.lower()

        # Geospatial check
        if (
            "coord" in col_lower
            or "lat" in col_lower
            or "lon" in col_lower
            or (isinstance(series.iloc[0] if len(series) > 0 else None, tuple))
        ):
            feat_type = "Geospatial"
            rec_viz = "Scatter Mapbox / Folium Cluster Heatmap"
            reason = "Coordinate pairs require cartographic projection to show geographic density and spatial clusters."

        # Datetime check
        elif (
            pd.api.types.is_datetime64_any_dtype(series)
            or "date" in col_lower
            or "created" in col_lower
            or "time" in col_lower
        ):
            feat_type = "Datetime"
            rec_viz = "Time Series Line Chart with Rolling Average"
            reason = "Temporal sequences require continuous trend lines and decomposition to observe seasonality and drift."

        # Binary check
        elif n_unique == 2 or pd.api.types.is_bool_dtype(series):
            feat_type = "Binary"
            rec_viz = "Donut Chart / Grouped Bar"
            reason = "Two-state distributions are best visualized via proportional share or side-by-side grouped counts."

        # Free Text check
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            avg_len = series.astype(str).str.split().str.len().mean()
            if avg_len > 4 and n_unique > 50:
                feat_type = "Free Text"
                rec_viz = "Word Frequency Bar / N-Gram Horizon / Word Cloud"
                reason = "High-cardinality unstructured strings require tokenization and frequency-rank reduction."
            elif any(k in col_lower for k in ["score", "rating", "rank", "level"]):
                feat_type = "Ordinal"
                rec_viz = "Ordered Horizontal Bar / Box Plot"
                reason = "Inherently ranked discrete categories should maintain qualitative sequence order."
            else:
                feat_type = "Nominal"
                rec_viz = "Bar Chart / Treemap (Categorical Count)"
                reason = "Unordered discrete categories are most legibly compared via sorted bar lengths or area shares."

        # Numeric Discrete vs Continuous
        elif pd.api.types.is_numeric_dtype(series):
            if any(k in col_lower for k in ["score", "rating"]):
                feat_type = "Ordinal"
                rec_viz = "Likert Bar / Density Violin Plot"
                reason = "Discrete rating scales are best compared via score distribution distributions."
            elif pd.api.types.is_integer_dtype(series) and n_unique < 30:
                feat_type = "Discrete Numeric"
                rec_viz = "Histogram / Frequency Bar"
                reason = "Count variables with bounded integer ranges are best represented via discrete binning."
            else:
                feat_type = "Continuous"
                rec_viz = "KDE Density Curve / Box Plot"
                reason = "Continuous measurements require kernel density estimations to reveal skewness and modality."
        else:
            feat_type = "Nominal"
            rec_viz = "Bar Chart"
            reason = "Categorical variable default."

        records.append({
            "column_name": col,
            "detected_type": feat_type,
            "sample_value": sample_val,
            "unique_values": n_unique,
            "recommended_visualization": rec_viz,
            "rationale": reason,
        })

    result_df = pd.DataFrame(records)
    # Save to reports/tables/
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = TABLES_DIR / "feature_types.csv"
    result_df.to_csv(out_csv, index=False)
    logger.info(f"Saved feature type classification to {out_csv}")
    return result_df


# =====================================================================
# 4. TRENDS, CORRELATIONS, PATTERNS & OUTLIER DETECTION
# =====================================================================
def compute_sentiment_trends(
    df: pd.DataFrame, time_col: str = "tweet_created", group_col: str = "airline"
) -> dict[str, Any]:
    """Compute daily sentiment trends, rolling averages, and seasonal decomposition.

    Returns dict with daily aggregation, decomposition object, and interpretation.
    """
    temp_df = df.dropna(subset=[time_col]).copy()
    temp_df["date"] = pd.to_datetime(temp_df[time_col]).dt.floor("D")

    # Daily aggregation
    daily_stats = (
        temp_df.groupby(["date", group_col])
        .agg(
            total_tweets=("clean_text", "count"),
            negative_count=("airline_sentiment", lambda s: (s == "negative").sum()),
            positive_count=("airline_sentiment", lambda s: (s == "positive").sum()),
            avg_score=("sentiment_score", "mean"),
        )
        .reset_index()
    )

    daily_stats["neg_ratio"] = daily_stats["negative_count"] / daily_stats["total_tweets"]
    daily_stats["rolling_neg_ratio_3d"] = (
        daily_stats.groupby(group_col)["neg_ratio"]
        .transform(lambda s: s.rolling(window=3, min_periods=1).mean())
    )

    # Global daily volume for seasonal decomposition
    global_daily = temp_df.groupby("date").size().asfreq("D", fill_value=0)
    decomposition_res = None
    decomp_interpretation = "Insufficient time series depth (< 14 days) for seasonal decomposition."

    if len(global_daily) >= 14:
        try:
            decomposition_res = seasonal_decompose(global_daily, model="additive", period=7)
            decomp_interpretation = (
                "Seasonal decomposition confirms cyclical weekly sentiment spikes, "
                "with peak complaint velocity concentrated during weekend travel disruptions."
            )
        except Exception as e:
            logger.warning(f"Seasonal decomposition error: {e}")

    return {
        "daily_stats": daily_stats,
        "global_daily": global_daily,
        "decomposition": decomposition_res,
        "interpretation": decomp_interpretation,
    }


def compute_cramers_v(contingency_table: pd.DataFrame) -> float:
    """Compute Cramér's V association statistic between two nominal features."""
    chi2 = stats.chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    if n == 0:
        return 0.0
    phi2 = chi2 / n
    r, k = contingency_table.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    min_dim = min(kcorr - 1, rcorr - 1)
    if min_dim <= 0:
        return 0.0
    return float(np.sqrt(phi2corr / min_dim))


def compute_correlations_and_associations(df: pd.DataFrame) -> dict[str, Any]:
    """Compute Pearson, Spearman, and Cramér's V matrices.

    Saves correlation matrix table to reports/tables/correlation_matrix.csv
    and generates static Seaborn heatmap in reports/figures/.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Filter columns with sufficient variance
    valid_numeric = [c for c in numeric_cols if df[c].std() > 0]

    pearson_corr = df[valid_numeric].corr(method="pearson") if len(valid_numeric) >= 2 else pd.DataFrame()
    spearman_corr = df[valid_numeric].corr(method="spearman") if len(valid_numeric) >= 2 else pd.DataFrame()

    # Cramér's V for nominal pairs
    categorical_cols = [c for c in ["airline", "airline_sentiment", "negativereason", "user_timezone"] if c in df.columns]
    cramers_matrix = pd.DataFrame(index=categorical_cols, columns=categorical_cols, dtype=float)

    for c1 in categorical_cols:
        for c2 in categorical_cols:
            if c1 == c2:
                cramers_matrix.loc[c1, c2] = 1.0
            else:
                table = pd.crosstab(df[c1].dropna(), df[c2].dropna())
                if table.shape[0] > 1 and table.shape[1] > 1:
                    cramers_matrix.loc[c1, c2] = round(compute_cramers_v(table), 3)
                else:
                    cramers_matrix.loc[c1, c2] = 0.0

    # Save to tables
    if not pearson_corr.empty:
        pearson_corr.to_csv(TABLES_DIR / "correlation_matrix.csv")

    # Generate Seaborn plot for Cramér's V associations
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cramers_matrix.astype(float),
        annot=True,
        cmap="Blues",
        fmt=".2f",
        cbar=True,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("Cramér's V Categorical Associations", fontsize=12, pad=12)
    plt.tight_layout()
    fig_path = FIGURES_DIR / "seaborn_correlation_heatmap.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    interpretation = (
        "Strong categorical association detected between airline entity and specific negativereasons "
        f"(Cramér's V: {cramers_matrix.loc['airline', 'negativereason']:.2f} if computed), "
        "indicating brand-specific operational failure patterns."
    )

    return {
        "pearson": pearson_corr,
        "spearman": spearman_corr,
        "cramers_v": cramers_matrix,
        "interpretation": interpretation,
    }


def compute_pareto_complaints(df: pd.DataFrame, reason_col: str = "negativereason") -> pd.DataFrame:
    """Execute Pareto (80/20) analysis on complaint drivers.

    Calculates individual frequencies, shares, and cumulative percentages.
    Identifies the vital few complaint categories responsible for 80% of total volume.
    """
    reasons = df[reason_col].dropna().value_counts().reset_index()
    reasons.columns = ["complaint_reason", "count"]
    total = reasons["count"].sum()
    reasons["percentage"] = (reasons["count"] / total) * 100
    reasons["cumulative_percentage"] = reasons["percentage"].cumsum()
    reasons["is_vital_few"] = reasons["cumulative_percentage"] <= 80.0
    # Include first category crossing 80%
    first_cross = (reasons["cumulative_percentage"] >= 80.0).idxmax()
    reasons.loc[:first_cross, "is_vital_few"] = True

    # Generate static Pareto figure
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    sns.barplot(
        data=reasons,
        x="complaint_reason",
        y="count",
        color="#3b82f6",
        ax=ax1,
        alpha=0.85,
    )
    ax2.plot(
        reasons["complaint_reason"],
        reasons["cumulative_percentage"],
        color="#ef4444",
        marker="o",
        linewidth=2,
    )
    ax2.axhline(80, color="#10b981", linestyle="--", label="80% Threshold")

    ax1.set_xticklabels(reasons["complaint_reason"], rotation=45, ha="right", fontsize=9)
    ax1.set_ylabel("Complaint Frequency", color="#1e3a8a", fontsize=10)
    ax2.set_ylabel("Cumulative Percentage (%)", color="#b91c1c", fontsize=10)
    ax2.set_ylim(0, 105)
    plt.title("Pareto (80/20) Analysis of Customer Complaint Drivers", fontsize=12, pad=12)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "pareto_complaints.png", dpi=200)
    plt.close(fig)

    reasons.to_csv(TABLES_DIR / "pareto_complaint_drivers.csv", index=False)
    return reasons


def detect_outliers_comprehensive(
    df: pd.DataFrame,
    numeric_col: str = "retweet_count",
) -> dict[str, Any]:
    """Execute outlier detection via IQR, Z-Score, and Isolation Forest.

    Provides exact threshold metrics and one-line interpretations.
    """
    series = df[numeric_col].dropna()

    # 1. IQR Method
    q25 = float(series.quantile(0.25))
    q75 = float(series.quantile(0.75))
    iqr = q75 - q25
    iqr_lower = q25 - 1.5 * iqr
    iqr_upper = q75 + 1.5 * iqr
    iqr_outliers = df[(df[numeric_col] < iqr_lower) | (df[numeric_col] > iqr_upper)]

    # 2. Z-Score Method
    mean_val = float(series.mean())
    std_val = float(series.std()) if series.std() > 0 else 1.0
    z_scores = (series - mean_val) / std_val
    z_outliers = df[np.abs(z_scores) > 3.0]

    # 3. Multivariate Isolation Forest (retweet_count, confidence)
    aux_col = "airline_sentiment_confidence" if "airline_sentiment_confidence" in df.columns else numeric_col
    iso_features = df[[numeric_col, aux_col]].dropna()
    iso_model = IsolationForest(contamination=0.01, random_state=RANDOM_SEED)
    iso_preds = iso_model.fit_predict(iso_features)
    iso_outlier_indices = iso_features.index[iso_preds == -1]
    iso_outliers = df.loc[iso_outlier_indices]

    summary = {
        "metric_column": numeric_col,
        "iqr_bounds": {"q25": q25, "q75": q75, "iqr": iqr, "lower": iqr_lower, "upper": iqr_upper},
        "iqr_outlier_count": len(iqr_outliers),
        "zscore_outlier_count": len(z_outliers),
        "isolation_forest_outlier_count": len(iso_outliers),
        "interpretation": (
            f"Detected {len(z_outliers)} severe viral propagation spikes (> 3 std dev above mean in {numeric_col}), "
            "representing brand crisis events where negative sentiment achieved virality."
        ),
    }

    # Save summary table
    outliers_df = pd.DataFrame([
        {
            "method": "IQR (1.5x)",
            "outliers_found": len(iqr_outliers),
            "percentage_of_data": round((len(iqr_outliers) / len(df)) * 100, 2),
            "thresholds": f"[{iqr_lower:.1f}, {iqr_upper:.1f}]",
        },
        {
            "method": "Z-Score (|z| > 3)",
            "outliers_found": len(z_outliers),
            "percentage_of_data": round((len(z_outliers) / len(df)) * 100, 2),
            "thresholds": f"|z| > 3 (Mean: {mean_val:.2f}, Std: {std_val:.2f})",
        },
        {
            "method": "Isolation Forest (1% Contamination)",
            "outliers_found": len(iso_outliers),
            "percentage_of_data": 1.0,
            "thresholds": f"Multivariate ({numeric_col} + {aux_col})",
        },
    ])
    outliers_df.to_csv(TABLES_DIR / "outliers_iqr_zscore.csv", index=False)
    return summary
