"""BrandPulse AI - Automated Unit & Integration Test Suite.

Tests cover:
1. Coordinate parsing and geospatial validation
2. Text cleaning and normalization
3. Classification metric computation shapes and ranges
4. Graph data structures, centrality, and community detection
5. Ethical web scraping and robots.txt validation
6. Automatic feature type classification schema
7. Competitor benchmarking statistical significance testing
"""

from __future__ import annotations

import math
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd
import pytest

from modules.competitor_benchmark import compare_two_entities
from modules.data_processor import (
    classify_feature_types,
    clean_text,
    compute_cramers_v,
    parse_tweet_coordinates,
)
from modules.network_analyzer import (
    analyze_graph_dynamics,
    demonstrate_graph_data_structures,
)
from modules.scrapers_demo import EthicalBeautifulSoupScraper


# =====================================================================
# 1. COORDINATE PARSING & GEO VALIDATION TESTS
# =====================================================================
def test_parse_tweet_coordinates_valid():
    """Verify correct extraction of valid [lat, lon] strings."""
    lat, lon = parse_tweet_coordinates("[40.7128, -74.0060]")
    assert lat == pytest.approx(40.7128, 0.0001)
    assert lon == pytest.approx(-74.0060, 0.0001)


def test_parse_tweet_coordinates_tuples_and_lists():
    """Verify extraction from raw python tuples or lists."""
    lat, lon = parse_tweet_coordinates([34.0522, -118.2437])
    assert lat == pytest.approx(34.0522, 0.0001)
    assert lon == pytest.approx(-118.2437, 0.0001)


def test_parse_tweet_coordinates_invalid_bounds():
    """Verify rejection of out-of-bounds latitude/longitude."""
    # Latitude > 90
    lat, lon = parse_tweet_coordinates("[105.0, 50.0]")
    assert lat is None and lon is None

    # Longitude < -180
    lat, lon = parse_tweet_coordinates("[45.0, -195.0]")
    assert lat is None and lon is None


def test_parse_tweet_coordinates_missing_and_malformed():
    """Verify handling of NaN, empty strings, and malformed syntax."""
    assert parse_tweet_coordinates(np.nan) == (None, None)
    assert parse_tweet_coordinates(None) == (None, None)
    assert parse_tweet_coordinates("") == (None, None)
    assert parse_tweet_coordinates("NotACoordinate") == (None, None)
    assert parse_tweet_coordinates("[40.5]") == (None, None)


# =====================================================================
# 2. TEXT CLEANING & NORMALIZATION TESTS
# =====================================================================
def test_clean_text_strips_urls_and_mentions():
    """Verify regex removal of URLs and Twitter handles."""
    raw = "Hey @United your flight http://t.co/xyz was delayed! @Delta was better."
    cleaned = clean_text(raw)
    assert "http" not in cleaned
    assert "@united" not in cleaned
    assert "@delta" not in cleaned
    assert "flight" in cleaned
    assert "delayed" in cleaned


def test_clean_text_html_and_whitespace():
    """Verify stripping of HTML entities and extra whitespace."""
    raw = "Delayed &amp; canceled <br> flight &lt;3   terrible service!"
    cleaned = clean_text(raw)
    assert "<br>" not in cleaned
    assert "&amp;" not in cleaned
    assert "  " not in cleaned
    assert "delayed" in cleaned
    assert "canceled" in cleaned


def test_clean_text_empty_input():
    """Verify graceful handling of empty or non-string inputs."""
    assert clean_text("") == ""
    assert clean_text(None) == ""
    assert clean_text(12345) == ""


# =====================================================================
# 3. STATISTICAL & FEATURE TYPE TESTS
# =====================================================================
def test_feature_type_classifier_structure():
    """Verify automatic feature-type classifier produces required columns."""
    sample_df = pd.DataFrame({
        "tweet_id": [101, 102, 103],
        "sentiment": ["negative", "positive", "neutral"],
        "is_verified": [True, False, True],
        "rating_score": [1.5, 4.2, 3.0],
        "created_at": ["2026-02-15 10:00:00", "2026-02-15 11:00:00", "2026-02-15 12:00:00"],
        "comment_text": [
            "This flight was horribly late and my bag got lost entirely.",
            "Great smooth experience with delicious inflight coffee.",
            "Average flight, departed on time but seat was cramped.",
        ],
    })
    res_df = classify_feature_types(sample_df)
    assert "column_name" in res_df.columns
    assert "detected_type" in res_df.columns
    assert "recommended_visualization" in res_df.columns
    assert "rationale" in res_df.columns
    assert len(res_df) == 6


def test_cramers_v_perfect_and_independent():
    """Verify Cramér's V bounds on synthetic contingency tables."""
    # Perfect association
    perf_table = pd.DataFrame([[100, 0], [0, 100]])
    v_perf = compute_cramers_v(perf_table)
    assert v_perf > 0.95

    # Completely independent distribution
    indep_table = pd.DataFrame([[50, 50], [50, 50]])
    v_indep = compute_cramers_v(indep_table)
    assert v_indep == pytest.approx(0.0, 0.05)


# =====================================================================
# 4. GRAPH DATA STRUCTURES & CENTRALITY TESTS
# =====================================================================
def test_graph_data_structures_demo():
    """Verify graph data structure demonstration generator."""
    demo = demonstrate_graph_data_structures()
    assert demo["num_nodes"] == 5
    assert demo["num_edges"] == 6
    assert demo["is_directed"] is True
    assert len(demo["edge_list"]) == 6
    assert len(demo["node_order"]) == 5
    assert len(demo["adjacency_matrix"]) == 5


def test_graph_dynamics_analytics_on_fixture():
    """Verify graph centrality and Louvain community metrics on tiny fixture."""
    # Create synthetic connected barbell graph
    g = nx.barbell_graph(5, 1)
    summary = analyze_graph_dynamics(g, k_samples_betweenness=10)

    assert summary["num_nodes"] == 11
    assert summary["num_edges"] == 22
    assert summary["num_communities"] >= 2
    assert -0.5 <= summary["modularity_score"] <= 1.0
    assert 0.0 <= summary["within_edge_ratio"] <= 1.0
    assert len(summary["top_influencers"]) > 0


# =====================================================================
# 5. ETHICAL SCRAPING & ROBOTS.TXT LOGIC TESTS
# =====================================================================
def test_ethical_scraper_robots_validation():
    """Verify robots.txt logic permits legal sandbox crawling."""
    scraper = EthicalBeautifulSoupScraper()
    # quotes.toscrape.com is an open public educational sandbox
    allowed = scraper.is_url_allowed("https://quotes.toscrape.com/page/1/")
    assert allowed is True


def test_ethical_scraper_fallback_sample():
    """Verify offline fallback produces valid quote dictionaries."""
    scraper = EthicalBeautifulSoupScraper()
    quotes = scraper._fallback_quotes_sample()
    assert len(quotes) >= 2
    assert "text" in quotes[0]
    assert "author" in quotes[0]


# =====================================================================
# 6. COMPETITOR BENCHMARK STATISTICAL TESTS
# =====================================================================
def test_competitor_comparison_significance():
    """Verify statistical significance calculation between synthetic distributions."""
    df_a = pd.DataFrame({
        "sentiment": ["positive"] * 80 + ["negative"] * 20,
        "clean_text": ["great service"] * 100,
    })
    df_b = pd.DataFrame({
        "sentiment": ["positive"] * 20 + ["negative"] * 80,
        "clean_text": ["terrible flight"] * 100,
    })

    comp = compare_two_entities(df_a, df_b, "BrandA", "BrandB", "TestCategory")
    assert comp["status"] == "available"
    assert bool(comp["is_statistically_significant"]) is True
    assert comp["chi2_p_value"] < 0.001
    assert comp["net_sentiment_a"] == pytest.approx(60.0)
    assert comp["net_sentiment_b"] == pytest.approx(-60.0)
    assert comp["nss_difference"] == pytest.approx(120.0)
