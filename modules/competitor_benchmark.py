"""BrandPulse AI - Competitor Benchmarking & Statistical Significance Engine.

Conducts rigorous cross-brand comparative analysis for 5 target industry pairs:
1. E-Commerce: Amazon vs Myntra (ShoppingAppReviews real data)
2. Aviation (India): IndiGo vs Air India (Indian Domestic Airlines real data)
3. Consumer Electronics: Apple vs Samsung (Amazon Cell Phone Reviews real data)
4. Healthcare: Mayo Clinic vs Johns Hopkins (Data Gap audited per Rule 1)
5. Higher Education: MIT vs Stanford (Data Gap audited per Rule 1)

Comparative Metrics:
- Volume & Voice Share
- Sentiment Distribution & Net Sentiment Score (NSS = Pos% - Neg%)
- Statistical Significance:
  - Chi-Square Test of Independence for sentiment categorical distributions (with Cramér's V effect size)
  - Mann-Whitney U Test for ordinal sentiment scores (with rank-biserial effect size)
- Aspect-Level Decompositions and Complaint Divergence
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scipy import stats

from config import (
    CACHE_DIR,
    COMPETITOR_PAIRS,
    FIGURES_DIR,
    TABLES_DIR,
)
from modules.absa_engine import detect_aspects_lexicon
from modules.data_processor import (
    load_cell_phone_reviews,
    load_indian_airlines_reviews,
    load_shopping_apps_reviews,
)

logger = logging.getLogger("competitor_benchmark")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def compare_two_entities(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    name_a: str,
    name_b: str,
    category: str,
    sentiment_col_a: str = "sentiment",
    sentiment_col_b: str = "sentiment",
    score_col_a: str | None = None,
    score_col_b: str | None = None,
) -> dict[str, Any]:
    """Execute generic, statistically rigorous comparative benchmark between two entities."""
    n_a = len(df_a)
    n_b = len(df_b)
    total_vol = n_a + n_b

    # Sentiment distribution
    sent_counts_a = df_a[sentiment_col_a].value_counts()
    sent_counts_b = df_b[sentiment_col_b].value_counts()

    pos_a = int(sent_counts_a.get("positive", 0))
    neg_a = int(sent_counts_a.get("negative", 0))
    neu_a = int(sent_counts_a.get("neutral", 0))

    pos_b = int(sent_counts_b.get("positive", 0))
    neg_b = int(sent_counts_b.get("negative", 0))
    neu_b = int(sent_counts_b.get("neutral", 0))

    # Net Sentiment Scores
    nss_a = round(((pos_a - neg_a) / max(1, n_a)) * 100, 2)
    nss_b = round(((pos_b - neg_b) / max(1, n_b)) * 100, 2)
    nss_diff = round(nss_a - nss_b, 2)

    # 1. Chi-Square Test of Independence
    contingency = np.array([
        [pos_a, neu_a, neg_a],
        [pos_b, neu_b, neg_b],
    ])
    # Avoid zero rows
    valid_contingency = contingency[:, contingency.sum(axis=0) > 0]
    if valid_contingency.shape[1] > 1:
        chi2_stat, p_val_chi2, dof, _ = stats.chi2_contingency(valid_contingency)
        # Cramér's V effect size
        n_obs = contingency.sum()
        cramers_v = float(np.sqrt(chi2_stat / (n_obs * (min(valid_contingency.shape) - 1))))
    else:
        chi2_stat, p_val_chi2, cramers_v = 0.0, 1.0, 0.0

    # 2. Mann-Whitney U Test on sentiment scores
    score_map = {"negative": -1, "neutral": 0, "positive": 1}
    scores_a = (
        df_a[score_col_a].dropna().values
        if score_col_a and score_col_a in df_a.columns
        else df_a[sentiment_col_a].map(score_map).dropna().values
    )
    scores_b = (
        df_b[score_col_b].dropna().values
        if score_col_b and score_col_b in df_b.columns
        else df_b[sentiment_col_b].map(score_map).dropna().values
    )

    if len(scores_a) > 0 and len(scores_b) > 0:
        u_stat, p_val_u = stats.mannwhitneyu(scores_a, scores_b, alternative="two-sided")
        # Rank-biserial correlation effect size
        rank_biserial = float(1.0 - (2.0 * u_stat) / (len(scores_a) * len(scores_b)))
    else:
        u_stat, p_val_u, rank_biserial = 0.0, 1.0, 0.0

    # 3. Aspect-level comparison
    text_col_a = "clean_text" if "clean_text" in df_a.columns else "text"
    text_col_b = "clean_text" if "clean_text" in df_b.columns else "text"

    aspects_a = df_a[text_col_a].dropna().head(1000).apply(detect_aspects_lexicon).explode().value_counts()
    aspects_b = df_b[text_col_b].dropna().head(1000).apply(detect_aspects_lexicon).explode().value_counts()

    top_aspects = list(set(aspects_a.head(5).index).union(set(aspects_b.head(5).index)))
    aspect_comp = []
    for asp in top_aspects:
        share_a = round((aspects_a.get(asp, 0) / max(1, aspects_a.sum())) * 100, 1)
        share_b = round((aspects_b.get(asp, 0) / max(1, aspects_b.sum())) * 100, 1)
        aspect_comp.append({"aspect": asp, f"{name_a}_share": share_a, f"{name_b}_share": share_b})

    is_significant = p_val_chi2 < 0.05
    winner = name_a if nss_a > nss_b else name_b

    interpretation = (
        f"Statistically {'significant' if is_significant else 'insignificant'} difference detected between "
        f"{name_a} (NSS: {nss_a:+.1f}) and {name_b} (NSS: {nss_b:+.1f}) "
        f"(Chi² = {chi2_stat:.2f}, p = {p_val_chi2:.4e}, Cramér's V = {cramers_v:.3f}). "
        f"{winner} holds a net sentiment advantage of {abs(nss_diff):.1f} percentage points."
    )

    return {
        "status": "available",
        "entity_a": name_a,
        "entity_b": name_b,
        "category": category,
        "volume_a": n_a,
        "volume_b": n_b,
        "voice_share_a": round((n_a / max(1, total_vol)) * 100, 1),
        "voice_share_b": round((n_b / max(1, total_vol)) * 100, 1),
        "net_sentiment_a": nss_a,
        "net_sentiment_b": nss_b,
        "nss_difference": nss_diff,
        "distribution_a": {
            "positive": round((pos_a / max(1, n_a)) * 100, 1),
            "neutral": round((neu_a / max(1, n_a)) * 100, 1),
            "negative": round((neg_a / max(1, n_a)) * 100, 1),
        },
        "distribution_b": {
            "positive": round((pos_b / max(1, n_b)) * 100, 1),
            "neutral": round((neu_b / max(1, n_b)) * 100, 1),
            "negative": round((neg_b / max(1, n_b)) * 100, 1),
        },
        "chi2_statistic": round(float(chi2_stat), 3),
        "chi2_p_value": float(p_val_chi2),
        "cramers_v": round(float(cramers_v), 3),
        "mann_whitney_u": round(float(u_stat), 1),
        "mann_whitney_p": float(p_val_u),
        "rank_biserial_effect": round(float(rank_biserial), 3),
        "is_statistically_significant": is_significant,
        "interpretation": interpretation,
        "aspect_comparison": aspect_comp,
    }


def execute_all_competitor_benchmarks() -> dict[str, Any]:
    """Run comparative benchmark across all 5 candidate competitor pairs.

    Returns available results for verified real datasets, and standardized
    'data unavailable' objects for unverified datasets per Rule 1.
    """
    benchmarks: dict[str, Any] = {}
    sig_records = []

    # 1. Amazon vs Myntra
    apps_data = load_shopping_apps_reviews()
    if "Amazon" in apps_data and "Myntra" in apps_data:
        df_amz = apps_data["Amazon"].copy()
        df_myn = apps_data["Myntra"].copy()

        # Score map: score 1-2: negative, 3: neutral, 4-5: positive
        for d in [df_amz, df_myn]:
            d["sentiment"] = np.select(
                [d["score"] <= 2, d["score"] == 3, d["score"] >= 4],
                ["negative", "neutral", "positive"],
                default="neutral",
            )
        res_ecom = compare_two_entities(
            df_amz, df_myn, "Amazon", "Myntra", "E-Commerce", score_col_a="score", score_col_b="score"
        )
        benchmarks["ecommerce"] = res_ecom
    else:
        benchmarks["ecommerce"] = {
            "status": "unavailable",
            "entity_a": "Amazon",
            "entity_b": "Myntra",
            "reason": "ShoppingAppReviews dataset missing locally.",
        }

    # 2. IndiGo vs Air India (Indian carriers only)
    indian_airlines_df = load_indian_airlines_reviews()
    if not indian_airlines_df.empty:
        df_indigo = indian_airlines_df[
            indian_airlines_df["airline"].str.contains("IndiGo|Indigo", case=False, na=False)
        ].copy()
        df_airindia = indian_airlines_df[
            indian_airlines_df["airline"].str.contains("Air India", case=False, na=False)
        ].copy()

        if len(df_indigo) >= 10 and len(df_airindia) >= 10:
            res_air = compare_two_entities(
                df_indigo,
                df_airindia,
                "IndiGo",
                "Air India",
                "Aviation (India)",
                score_col_a="numeric_rating",
                score_col_b="numeric_rating",
            )
            benchmarks["aviation_india"] = res_air
        else:
            benchmarks["aviation_india"] = {
                "status": "unavailable",
                "entity_a": "IndiGo",
                "entity_b": "Air India",
                "reason": "Insufficient review volume for entity separation in Indian airlines dataset.",
            }
    else:
        benchmarks["aviation_india"] = {
            "status": "unavailable",
            "entity_a": "IndiGo",
            "entity_b": "Air India",
            "reason": "Indian domestic airlines dataset missing.",
        }

    # 3. Apple vs Samsung
    phone_data = load_cell_phone_reviews()
    if "Apple" in phone_data and "Samsung" in phone_data:
        df_apple = phone_data["Apple"].copy()
        df_samsung = phone_data["Samsung"].copy()

        for d in [df_apple, df_samsung]:
            d["sentiment"] = np.select(
                [d["rating"] <= 2, d["rating"] == 3, d["rating"] >= 4],
                ["negative", "neutral", "positive"],
                default="neutral",
            )
        res_tech = compare_two_entities(
            df_apple,
            df_samsung,
            "Apple",
            "Samsung",
            "Consumer Tech",
            score_col_a="rating",
            score_col_b="rating",
        )
        benchmarks["tech_giants"] = res_tech
    else:
        benchmarks["tech_giants"] = {
            "status": "unavailable",
            "entity_a": "Apple",
            "entity_b": "Samsung",
            "reason": "Phone reviews dataset missing locally.",
        }

    # 4. Mayo Clinic vs Johns Hopkins (Data Gap per Rule 1)
    benchmarks["healthcare"] = {
        "status": "unavailable",
        "entity_a": "Mayo Clinic",
        "entity_b": "Johns Hopkins",
        "category": "Healthcare",
        "reason": (
            "No verified public Kaggle patient review or Twitter sentiment dataset with discrete "
            "entity separation for Mayo Clinic vs Johns Hopkins was verified. Flagged as Data Gap per Rule 1."
        ),
    }

    # 5. MIT vs Stanford (Data Gap per Rule 1)
    benchmarks["higher_ed"] = {
        "status": "unavailable",
        "entity_a": "MIT",
        "entity_b": "Stanford",
        "category": "Higher Education",
        "reason": (
            "No public Kaggle consumer review/tweet dataset with isolated student sentiment for "
            "MIT vs Stanford was verified. Flagged as Data Gap per Rule 1."
        ),
    }

    # Collect significance records for summary table
    for pair_id, bench in benchmarks.items():
        if bench.get("status") == "available":
            sig_records.append({
                "Category": bench["category"],
                "Entity_A": bench["entity_a"],
                "Entity_B": bench["entity_b"],
                "Volume_A": bench["volume_a"],
                "Volume_B": bench["volume_b"],
                "NSS_A": bench["net_sentiment_a"],
                "NSS_B": bench["net_sentiment_b"],
                "Chi2_Stat": bench["chi2_statistic"],
                "Chi2_p_val": f"{bench['chi2_p_value']:.4e}",
                "Cramers_V": bench["cramers_v"],
                "Mann_Whitney_U": bench["mann_whitney_u"],
                "MW_p_val": f"{bench['mann_whitney_p']:.4e}",
                "Significant": bench["is_statistically_significant"],
            })

    if sig_records:
        sig_df = pd.DataFrame(sig_records)
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        sig_df.to_csv(TABLES_DIR / "competitor_significance_tests.csv", index=False)

    return benchmarks
