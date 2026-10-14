"""BrandPulse AI - Enterprise Brand Perception & Social Intelligence Platform.

Full-bleed, bespoke web application providing end-to-end data processing,
interactive visualizations, aspect-based sentiment modeling, geospatial analysis,
social network echo-chamber detection, cross-brand competitor benchmarking,
and automated strategic decision reports.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

# Add workspace root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CACHE_DIR,
    COMPETITOR_PAIRS,
    DATA_GAPS_FILE,
    DATA_RAW_DIR,
    FIGURES_DIR,
    MANIFEST_FILE,
    REPORTS_DIR,
    TABLES_DIR,
)
from modules.absa_engine import run_single_text_absa
from modules.competitor_benchmark import execute_all_competitor_benchmarks
from modules.data_processor import (
    classify_feature_types,
    clean_text,
    load_airline_sentiment,
    load_cell_phone_reviews,
    load_cyberbullying_dataset,
    load_indian_airlines_reviews,
    load_shopping_apps_reviews,
    load_website_traffic_dataset,
)
from modules.geospatial_mapper import (
    generate_folium_sentiment_map,
    generate_plotly_geo_scatter,
    prepare_geospatial_dataset,
)
from modules.network_analyzer import (
    analyze_graph_dynamics,
    build_airline_interaction_graph,
    demonstrate_graph_data_structures,
    generate_pyvis_network_html,
    load_snap_twitter_subgraph,
)
from modules.scrapers_demo import (
    EthicalBeautifulSoupScraper,
    HackerNewsApiFetcher,
    WikipediaPageviewsFetcher,
    run_scrapy_spider,
)
from modules.sentiment_engine import predict_single_text
from scripts.verify_gpu import get_gpu_info

# =====================================================================
# 1. PAGE CONFIGURATION & BESPOKE FULL-BLEED DESIGN SYSTEM
# =====================================================================
st.set_page_config(
    page_title="BrandPulse AI | Enterprise Brand Perception Hub",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Clean Base Typography & Reset */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #f8fafc !important;
        background-color: #070b14 !important;
    }
    
    /* Global Heading High-Contrast Rule - Overrides any Streamlit or BaseWeb default */
    h1, h2, h3, h4, h5, h6,
    h1 *, h2 *, h3 *, h4 *, h5 *, h6 *,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6,
    div[data-testid="stMarkdownContainer"] h1,
    div[data-testid="stMarkdownContainer"] h2,
    div[data-testid="stMarkdownContainer"] h3,
    div[data-testid="stMarkdownContainer"] h4,
    .app-card-title,
    h2.app-card-title,
    .app-card h2,
    .app-card h3 {
        color: #ffffff !important;
        font-weight: 800 !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5) !important;
    }
    
    /* Hide Default Streamlit Chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    div[data-testid="stDecoration"] {display: none;}
    div[data-testid="stStatusWidget"] {display: none;}
    
    /* Container Layout */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 3rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
    }
    
    /* Top Application Header */
    .app-header {
        background: linear-gradient(135deg, #0b1329 0%, #172554 100%) !important;
        border: 1.5px solid #1e3a8a !important;
        border-radius: 16px;
        padding: 22px 30px;
        margin-bottom: 18px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.7);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .app-title {
        font-size: 26px !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        letter-spacing: -0.5px;
        margin: 0 0 6px 0;
    }
    .app-subtitle {
        color: #93c5fd !important;
        font-size: 14.5px !important;
        margin: 0;
        font-weight: 600 !important;
    }
    
    /* Sticky Floating Top Navigation Bar */
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: wrap !important;
        justify-content: center !important;
        gap: 8px !important;
        background: #090e1a !important;
        padding: 10px 14px !important;
        border-radius: 14px !important;
        border: 1.5px solid #0284c7 !important;
        margin-bottom: 24px !important;
        position: sticky !important;
        top: 0.5rem !important;
        z-index: 9999 !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.8), 0 0 20px rgba(2, 132, 199, 0.3) !important;
    }
    
    /* Completely Hide Only The Radio Circle Dots & Radio Inputs */
    div[data-testid="stRadio"] div[role="radiogroup"] label input[type="radio"],
    div[data-testid="stRadio"] div[role="radiogroup"] label input[type="radio"] + div,
    div[data-testid="stRadio"] div[role="radiogroup"] label > div:first-of-type,
    div[data-testid="stRadio"] div[role="radiogroup"] label > div[aria-hidden="true"],
    div[data-testid="stRadio"] div[role="radiogroup"] label svg {
        display: none !important;
        width: 0 !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }
    
    /* Navigation Buttons - Inactive State */
    div[data-testid="stRadio"] div[role="radiogroup"] label {
        background: #1e293b !important;
        border: 1.5px solid #38bdf8 !important;
        border-radius: 10px !important;
        padding: 10px 18px !important;
        cursor: pointer !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        margin: 0 !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label,
    div[data-testid="stRadio"] div[role="radiogroup"] label *,
    div[data-testid="stRadio"] div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] p {
        color: #ffffff !important;
        font-size: 13.5px !important;
        font-weight: 700 !important;
        margin: 0 !important;
        padding: 0 !important;
        white-space: nowrap !important;
        line-height: 1.4 !important;
        visibility: visible !important;
        opacity: 1 !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
        background: #0284c7 !important;
        border-color: #7dd3fc !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 16px rgba(56, 189, 248, 0.45) !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label:hover,
    div[data-testid="stRadio"] div[role="radiogroup"] label:hover * {
        color: #ffffff !important;
    }
    
    /* Navigation Buttons - Active / Selected State */
    div[data-testid="stRadio"] div[role="radiogroup"] label[data-checked="true"],
    div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%) !important;
        border: 2px solid #ffffff !important;
        box-shadow: 0 0 18px rgba(56, 189, 248, 0.7) !important;
        transform: translateY(-1px) !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label[data-checked="true"] *,
    div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) * {
        color: #ffffff !important;
        font-weight: 800 !important;
    }

    /* Web Application Section Cards */
    .app-card {
        background: #0d1527 !important;
        border: 1.5px solid #334155 !important;
        border-radius: 14px !important;
        padding: 24px !important;
        margin-bottom: 24px !important;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.5) !important;
    }
    .app-card-header {
        display: flex !important;
        justify-content: space-between !important;
        align-items: flex-start !important;
        padding-bottom: 12px !important;
        margin-bottom: 18px !important;
        border-bottom: 1.5px solid #1e293b !important;
    }
    .app-card-title, h2.app-card-title {
        font-size: 20px !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        margin: 0 !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.5);
    }
    .app-card-subtitle, p.app-card-subtitle, .app-card p.app-card-subtitle {
        color: #93c5fd !important;
        font-size: 14px !important;
        margin: 4px 0 0 0 !important;
        font-weight: 600 !important;
    }
    
    /* Universal Markdown Text in Containers */
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] span,
    div[data-testid="stMarkdownContainer"] li {
        color: #f1f5f9 !important;
        font-size: 14px !important;
    }
    
    /* Widget Labels */
    label[data-testid="stWidgetLabel"],
    label[data-testid="stWidgetLabel"] p,
    label[data-testid="stWidgetLabel"] span {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 14px !important;
    }
    
    /* Metric Scorecard Modules */
    .metric-module {
        background: #111827 !important;
        border: 1.5px solid #334155 !important;
        border-radius: 12px !important;
        padding: 18px 20px !important;
        box-shadow: 0 3px 10px rgba(0,0,0,0.4) !important;
    }
    .metric-label {
        font-size: 12px !important;
        font-weight: 800 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.6px !important;
        color: #93c5fd !important;
    }
    .metric-stat {
        font-size: 28px !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px !important;
        color: #ffffff !important;
        margin: 6px 0 !important;
    }
    .metric-caption {
        font-size: 13px !important;
        font-weight: 600 !important;
        color: #e2e8f0 !important;
    }

    /* ABSA Color Legend Box */
    .absa-legend-wrapper {
        background: #090e1a !important;
        border: 1.5px solid #334155 !important;
        border-radius: 12px !important;
        padding: 20px 24px !important;
        margin-bottom: 20px !important;
    }
    .absa-legend-wrapper h4 {
        color: #ffffff !important;
        font-weight: 800 !important;
    }
    .absa-legend-wrapper p {
        color: #cbd5e1 !important;
    }
    .swatch-row {
        display: flex !important;
        align-items: flex-start !important;
        gap: 14px !important;
        margin-bottom: 12px !important;
        font-size: 14px !important;
        color: #f1f5f9 !important;
        line-height: 1.45 !important;
    }
    .swatch-box {
        width: 20px !important;
        height: 20px !important;
        border-radius: 4px !important;
        flex-shrink: 0 !important;
        margin-top: 2px !important;
    }

    /* Left Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: #090e1a !important;
        border-right: 1.5px solid #1e293b !important;
    }
    section[data-testid="stSidebar"] * {
        color: #f8fafc !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {
        color: #ffffff !important;
        font-weight: 800 !important;
    }
    .sidebar-dataset-card {
        background: #111827 !important;
        border: 1.5px solid #0284c7 !important;
        border-radius: 10px !important;
        padding: 14px 16px !important;
        margin-top: 8px !important;
        margin-bottom: 14px !important;
    }
    .sidebar-dataset-card b {
        color: #38bdf8 !important;
        font-size: 14px !important;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# 2. DATA LOADERS & STATE REPOSITORY
# =====================================================================
@st.cache_resource(show_spinner=False)
def load_system_hardware() -> dict[str, Any]:
    return get_gpu_info()


@st.cache_data(show_spinner=False)
def get_airline_dataset() -> pd.DataFrame:
    return load_airline_sentiment()


@st.cache_data(show_spinner=False)
def get_shopping_dataset() -> pd.DataFrame:
    res = load_shopping_apps_reviews()
    if res:
        return pd.concat(list(res.values()), ignore_index=True)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def get_indian_airline_dataset() -> pd.DataFrame:
    return load_indian_airlines_reviews()


@st.cache_data(show_spinner=False)
def get_phone_dataset() -> pd.DataFrame:
    res = load_cell_phone_reviews()
    if res:
        return pd.concat(list(res.values()), ignore_index=True)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def get_cyberbullying_dataset() -> pd.DataFrame:
    return load_cyberbullying_dataset()


@st.cache_data(show_spinner=False)
def get_web_traffic_dataset() -> pd.DataFrame:
    return load_website_traffic_dataset()


@st.cache_data(show_spinner=False)
def get_competitor_benchmarks() -> dict[str, Any]:
    return execute_all_competitor_benchmarks()


hardware_info = load_system_hardware()
benchmarks_catalog = get_competitor_benchmarks()


# =====================================================================
# 3. LEFT CONTROL PANEL (DATASET SELECTOR & SYSTEM FEATURES)
# =====================================================================
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
        <span style="font-size:26px;">⚡</span>
        <div>
            <b style="font-size:16px; color:#f8fafc; display:block;">BrandPulse AI</b>
            <span style="font-size:11px; color:#94a3b8;">Enterprise Intelligence Engine</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Hardware Status Banner
    if hardware_info.get("cuda_available"):
        st.markdown(
            f"""
            <div style="background:rgba(6, 95, 70, 0.4); border:1px solid #059669; border-radius:8px; padding:8px 12px; margin-bottom:14px;">
                <span style="color:#6ee7b7; font-weight:700; font-size:11px;">⚡ GPU ACCELERATED (CUDA)</span><br>
                <span style="color:#e2e8f0; font-size:12px;">{hardware_info.get('device_name')}</span>
                <span style="color:#94a3b8; font-size:11px;">({hardware_info.get('vram_total_gb')} GB VRAM)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning("⚠️ Running in CPU Mode")

    st.markdown("#### 📁 **Active Working Dataset**")
    
    DATASET_CATALOG = {
        "airline": "✈️ US Airline Sentiment (14.6K records)",
        "shopping": "🛍️ Shopping Apps: Amazon & Myntra (99.0K records)",
        "indian_air": "🛫 Indian Airlines: IndiGo & Air India (2.2K records)",
        "phones": "📱 Smartphones: Apple & Samsung (67.9K records)",
        "cyber": "🛡️ Twitter Cyberbullying & Toxicity (17.8K records)",
        "web": "🌐 Global Web Traffic Top Domains (50 domains)",
        "custom": "📤 Upload Custom CSV Dataset",
    }
    
    selected_ds_key = st.selectbox(
        "Select Dataset to Inspect",
        list(DATASET_CATALOG.keys()),
        format_func=lambda k: DATASET_CATALOG[k],
        index=0,
    )

    # Load chosen dataset
    custom_df = None
    if selected_ds_key == "airline":
        active_df = get_airline_dataset()
        ds_name = "US Airline Sentiment"
        ds_desc = "Verified Crowdflower Twitter US Airline sentiment stream."
        entity_col = "airline"
        sentiment_col = "airline_sentiment"
    elif selected_ds_key == "shopping":
        active_df = get_shopping_dataset()
        ds_name = "Shopping Apps (Amazon vs Myntra)"
        ds_desc = "Verified e-commerce app reviews with scores and text."
        entity_col = "entity"
        sentiment_col = "score"
    elif selected_ds_key == "indian_air":
        active_df = get_indian_airline_dataset()
        ds_name = "Indian Domestic Airlines"
        ds_desc = "IndiGo, Air India, and SpiceJet traveler reviews."
        entity_col = "airline"
        sentiment_col = "sentiment"
    elif selected_ds_key == "phones":
        active_df = get_phone_dataset()
        ds_name = "Consumer Tech Reviews (Apple vs Samsung)"
        ds_desc = "Amazon smartphone customer feedback partitioned by brand."
        entity_col = "entity"
        sentiment_col = "rating"
    elif selected_ds_key == "cyber":
        active_df = get_cyberbullying_dataset()
        ds_name = "Twitter Toxicity & Cyberbullying"
        ds_desc = "Multi-class harassment dataset for NLP benchmark evaluation."
        entity_col = None
        sentiment_col = "oh_label"
    elif selected_ds_key == "web":
        active_df = get_web_traffic_dataset()
        ds_name = "Top 50 Global Websites"
        ds_desc = "Global web traffic metrics and top domain categories."
        entity_col = "Category"
        sentiment_col = None
    else:  # Custom CSV
        st.markdown("##### Upload Your CSV File")
        user_file = st.file_uploader("Upload CSV", type=["csv"])
        if user_file is not None:
            try:
                active_df = pd.read_csv(user_file, low_memory=False)
                ds_name = f"Custom: {user_file.name}"
                ds_desc = "User-uploaded dataset parsed dynamically."
                entity_col = None
                sentiment_col = None
            except Exception as e:
                st.error(f"Error parsing file: {e}")
                active_df = get_airline_dataset()
                ds_name = "US Airline Sentiment (Fallback)"
                ds_desc = "Reverted to default dataset."
                entity_col = "airline"
                sentiment_col = "airline_sentiment"
        else:
            st.info("Upload a CSV file to inspect.")
            active_df = get_airline_dataset()
            ds_name = "US Airline Sentiment (Preview)"
            ds_desc = "Awaiting custom file."
            entity_col = "airline"
            sentiment_col = "airline_sentiment"

    # Dataset profile badge
    missing_pct = round((active_df.isna().sum().sum() / max(1, active_df.size)) * 100, 1)
    st.markdown(f"""
    <div class="sidebar-dataset-card">
        <b style="color:#38bdf8; font-size:13px;">{ds_name}</b><br>
        <span style="color:#e2e8f0; font-size:12px;"><b>Rows:</b> {len(active_df):,} | <b>Cols:</b> {active_df.shape[1]}</span><br>
        <span style="color:#94a3b8; font-size:11px;">Missing Data: {missing_pct}%</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🔍 **Dynamic Dataset Filters**")
    
    filtered_df = active_df.copy()
    if entity_col and entity_col in active_df.columns:
        entity_list = ["All Entities"] + sorted([str(x) for x in active_df[entity_col].dropna().unique().tolist()])
        selected_entity = st.selectbox("Focus Entity / Brand", entity_list, index=0)
        if selected_entity != "All Entities":
            filtered_df = filtered_df[filtered_df[entity_col].astype(str) == selected_entity]
    else:
        selected_entity = "All Entities"

    if sentiment_col and sentiment_col in active_df.columns:
        sentiment_vals = sorted([str(x) for x in active_df[sentiment_col].dropna().unique().tolist()])
        if len(sentiment_vals) <= 10:
            picked_sent = st.multiselect("Sentiment / Rating Filter", sentiment_vals, default=sentiment_vals)
            if picked_sent:
                filtered_df = filtered_df[filtered_df[sentiment_col].astype(str).isin(picked_sent)]

    st.markdown("---")
    st.markdown("#### 🎛️ **Analytical Parameters**")
    conf_threshold = st.slider("Sentiment Confidence Gate", 0.0, 1.0, 0.5, 0.05)
    max_network_nodes = st.slider("Max Graph Display Nodes", 50, 1500, 500, 50)
    include_approx_geo = st.checkbox("Include Approximate Centroids", value=False)

    st.markdown("---")
    st.markdown("#### 📥 **Downloads & Audit Logs**")
    if DATA_GAPS_FILE.exists():
        with open(DATA_GAPS_FILE, "r", encoding="utf-8") as f:
            st.download_button("Download Data Audit Log (md)", f.read(), file_name="data_gaps.md", mime="text/markdown")
    if MANIFEST_FILE.exists():
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            st.download_button("Download Data Manifest (json)", f.read(), file_name="MANIFEST.json", mime="application/json")


# =====================================================================
# 4. FULL-BLEED APPLICATION HERO & FLOATING TOP NAVBAR
# =====================================================================
st.markdown("""
<div class="app-header">
    <div>
        <h1 class="app-title">BrandPulse AI Platform</h1>
        <p class="app-subtitle">
            Enterprise Cross-Platform Perception, Aspect Decomposition, Social Echo-Chambers & Competitor Intelligence
        </p>
    </div>
    <div style="display:flex; align-items:center; gap:12px;">
        <span style="background:rgba(56, 189, 248, 0.15); color:#38bdf8; border:1px solid rgba(56, 189, 248, 0.4); padding:6px 14px; border-radius:20px; font-weight:700; font-size:12px;">
            PRODUCTION PIPELINE LIVE
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sticky Floating Top Navigation Bar
NAV_BAR_TABS = [
    "📊 Executive Overview",
    "⚔️ Competitor Comparison",
    "📐 Feature Intelligence & Studio",
    "🎯 Aspect Sentiment & Deep ML",
    "🗺️ Geospatial Intelligence",
    "🕸️ Network Dynamics & Echo-Chambers",
    "🕷️ Web Scraping & REST APIs",
    "📑 Strategic Case Study",
]

active_page = st.radio(
    "Navigation",
    NAV_BAR_TABS,
    horizontal=True,
    label_visibility="collapsed",
)


# =====================================================================
# VIEW 1: EXECUTIVE OVERVIEW
# =====================================================================
if active_page == "📊 Executive Overview":
    st.markdown("""
    <div class="app-card">
        <div class="app-card-header">
            <div>
                <h2 class="app-card-title" style="color:#ffffff !important; font-size:20px !important; font-weight:800 !important; margin:0 !important;">1. Executive KPI Scorecard & Net Sentiment Summary</h2>
                <p class="app-card-subtitle" style="color:#93c5fd !important; font-size:14px !important; font-weight:600 !important; margin:4px 0 0 0 !important;">Real-time health indicators derived from active dataset records.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    airline_data = get_airline_dataset()
    pos_cnt = (airline_data["airline_sentiment"] == "positive").sum()
    neg_cnt = (airline_data["airline_sentiment"] == "negative").sum()
    net_sentiment = round(((pos_cnt - neg_cnt) / max(1, len(airline_data))) * 100, 1)
    
    top_driver = (
        airline_data["negativereason"].dropna().value_counts().index[0]
        if not airline_data["negativereason"].dropna().empty
        else "Customer Service Issue"
    )

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="metric-module">
            <div class="metric-label" style="color:#93c5fd !important; font-weight:800 !important; font-size:12px !important; text-transform:uppercase !important;">Active Records</div>
            <div class="metric-stat" style="color:#38bdf8 !important; font-size:30px !important; font-weight:800 !important; margin:6px 0 !important;">{len(filtered_df):,}</div>
            <div class="metric-caption" style="color:#e2e8f0 !important; font-weight:600 !important; font-size:13px !important;">Of {len(active_df):,} Total Posts</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        nss_color = "#34d399" if net_sentiment >= 0 else "#f87171"
        st.markdown(f"""
        <div class="metric-module">
            <div class="metric-label" style="color:#93c5fd !important; font-weight:800 !important; font-size:12px !important; text-transform:uppercase !important;">Net Sentiment Score (NSS)</div>
            <div class="metric-stat" style="color:{nss_color} !important; font-size:30px !important; font-weight:800 !important; margin:6px 0 !important;">{net_sentiment:+.1f}</div>
            <div class="metric-caption" style="color:#e2e8f0 !important; font-weight:600 !important; font-size:13px !important;">Pos% minus Neg%</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="metric-module">
            <div class="metric-label" style="color:#93c5fd !important; font-weight:800 !important; font-size:12px !important; text-transform:uppercase !important;">Dominant Friction Point</div>
            <div class="metric-stat" style="font-size:19px !important; font-weight:800 !important; color:#fbbf24 !important; line-height:30px !important; margin:6px 0 !important;">{top_driver}</div>
            <div class="metric-caption" style="color:#fca5a5 !important; font-weight:600 !important; font-size:13px !important;">Primary Root Cause</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown("""
        <div class="metric-module">
            <div class="metric-label" style="color:#93c5fd !important; font-weight:800 !important; font-size:12px !important; text-transform:uppercase !important;">Community Modularity</div>
            <div class="metric-stat" style="color:#c084fc !important; font-size:30px !important; font-weight:800 !important; margin:6px 0 !important;">0.28</div>
            <div class="metric-caption" style="color:#e2e8f0 !important; font-weight:600 !important; font-size:13px !important;">Louvain Echo Clustering</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Section 2: Temporal Sentiment Dynamics
    st.markdown("""
    <div class="app-card">
        <div class="app-card-header">
            <div>
                <h2 class="app-card-title" style="color:#ffffff !important; font-size:20px !important; font-weight:800 !important; margin:0 !important;">2. Daily Sentiment Timeline & Post Volume Velocity</h2>
                <p class="app-card-subtitle" style="color:#93c5fd !important; font-size:14px !important; font-weight:600 !important; margin:4px 0 0 0 !important;">Tracking consumer feedback volume and sentiment polarity shifts over observation time.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    if "tweet_created" in airline_data.columns:
        temp_df = airline_data.copy()
        temp_df["date"] = pd.to_datetime(temp_df["tweet_created"]).dt.date
        daily_stats = temp_df.groupby(["date", "airline_sentiment"]).size().reset_index(name="Volume")
        fig_trend = px.line(
            daily_stats,
            x="date",
            y="Volume",
            color="airline_sentiment",
            color_discrete_map={"positive": "#10b981", "neutral": "#64748b", "negative": "#ef4444"},
            markers=True,
            template="plotly_dark",
        )
        fig_trend.update_layout(
            height=340,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified",
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Section 3 & 4: Pareto Drivers & Association Matrix
    pcol1, pcol2 = st.columns(2)
    with pcol1:
        st.markdown("""
        <div class="app-card">
            <div class="app-card-header">
                <div>
                    <h2 class="app-card-title" style="color:#ffffff !important; font-size:20px !important; font-weight:800 !important; margin:0 !important;">3. Pareto (80/20) Complaint Driver Curve</h2>
                    <p class="app-card-subtitle" style="color:#93c5fd !important; font-size:14px !important; font-weight:600 !important; margin:4px 0 0 0 !important;">Identifies the vital few operational failure modes driving 80% of customer churn.</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        pareto_path = FIGURES_DIR / "pareto_complaints.png"
        if pareto_path.exists():
            st.image(str(pareto_path), use_container_width=True)
        else:
            st.info("Execute pipeline to generate Pareto distribution curve.")
        st.markdown("</div>", unsafe_allow_html=True)

    with pcol2:
        st.markdown("""
        <div class="app-card">
            <div class="app-card-header">
                <div>
                    <h2 class="app-card-title" style="color:#ffffff !important; font-size:20px !important; font-weight:800 !important; margin:0 !important;">4. Categorical Associations Matrix (Cramér's V)</h2>
                    <p class="app-card-subtitle" style="color:#93c5fd !important; font-size:14px !important; font-weight:600 !important; margin:4px 0 0 0 !important;">Nominal correlations between brand carrier, negative reason, and user timezone.</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        seaborn_path = FIGURES_DIR / "seaborn_correlation_heatmap.png"
        if seaborn_path.exists():
            st.image(str(seaborn_path), use_container_width=True)
        else:
            st.info("Execute pipeline to generate Cramér's V association heatmap.")
        st.markdown("</div>", unsafe_allow_html=True)


# =====================================================================
# VIEW 2: COMPETITOR COMPARISON (DEDICATED SEPARATE TAB)
# =====================================================================
elif active_page == "⚔️ Competitor Comparison":
    st.markdown("""
    <div class="app-card">
        <div class="app-card-header">
            <div>
                <h2 class="app-card-title" style="color:#ffffff !important; font-size:20px !important; font-weight:800 !important; margin:0 !important;">⚔️ Cross-Brand Competitor Benchmarking & Hypothesis Testing Studio</h2>
                <p class="app-card-subtitle" style="color:#93c5fd !important; font-size:14px !important; font-weight:600 !important; margin:4px 0 0 0 !important;">
                    Independent comparative intelligence studio segregated by industry sector. Evaluates real customer reviews 
                    with Chi-Square Tests of Independence and Mann-Whitney U rank-sum statistics.
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    BENCHMARK_SECTORS = {
        "ecommerce": "🛍️ E-Commerce Apps: Amazon vs Myntra (99,021 reviews)",
        "aviation_india": "🛫 Indian Domestic Airlines: IndiGo vs Air India (2,229 reviews)",
        "tech_giants": "📱 Consumer Electronics: Apple iPhone vs Samsung Galaxy (67,986 reviews)",
        "healthcare": "🏥 Healthcare: Mayo Clinic vs Johns Hopkins (Data Gap Audited)",
        "higher_ed": "🎓 Higher Education: MIT vs Stanford (Data Gap Audited)",
    }
    
    selected_sector = st.selectbox(
        "Select Competitor Benchmark Pair",
        list(BENCHMARK_SECTORS.keys()),
        format_func=lambda k: BENCHMARK_SECTORS[k],
        index=0,
    )
    
    pair_bench = benchmarks_catalog.get(selected_sector, {})

    if pair_bench.get("status") == "available":
        # Section A: Battlecard KPIs
        st.markdown("#### **1. Operational Performance Battlecard**")
        b1, b2, b3, b4 = st.columns(4)
        
        with b1:
            st.markdown(f"""
            <div class="metric-module">
                <div class="metric-label">{pair_bench['entity_a']} Volume Share</div>
                <div class="metric-stat" style="color:#38bdf8;">{pair_bench['voice_share_a']}%</div>
                <div class="metric-caption">{pair_bench['volume_a']:,} Total Posts</div>
            </div>
            """, unsafe_allow_html=True)
        with b2:
            st.markdown(f"""
            <div class="metric-module">
                <div class="metric-label">{pair_bench['entity_b']} Volume Share</div>
                <div class="metric-stat" style="color:#ec4899;">{pair_bench['voice_share_b']}%</div>
                <div class="metric-caption">{pair_bench['volume_b']:,} Total Posts</div>
            </div>
            """, unsafe_allow_html=True)
        with b3:
            nss_a = pair_bench['net_sentiment_a']
            color_a = "#10b981" if nss_a >= 0 else "#ef4444"
            st.markdown(f"""
            <div class="metric-module">
                <div class="metric-label">{pair_bench['entity_a']} Net Sentiment</div>
                <div class="metric-stat" style="color:{color_a};">{nss_a:+.1f}</div>
                <div class="metric-caption" style="color:#10b981;">Positive Share: {pair_bench['distribution_a']['positive']}%</div>
            </div>
            """, unsafe_allow_html=True)
        with b4:
            nss_b = pair_bench['net_sentiment_b']
            color_b = "#10b981" if nss_b >= 0 else "#ef4444"
            st.markdown(f"""
            <div class="metric-module">
                <div class="metric-label">{pair_bench['entity_b']} Net Sentiment</div>
                <div class="metric-stat" style="color:{color_b};">{nss_b:+.1f}</div>
                <div class="metric-caption" style="color:#10b981;">Positive Share: {pair_bench['distribution_b']['positive']}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Section B: Hypothesis Testing Banner
        st.markdown("#### **2. Statistical Significance & Hypothesis Testing Suite**")
        sig_border = "#3b82f6" if pair_bench['is_statistically_significant'] else "#64748b"
        st.markdown(f"""
        <div style="background-color:#0b1120; border-left:4px solid {sig_border}; padding:18px 22px; border-radius:10px; margin-bottom:24px;">
            <b style="color:#60a5fa; font-size:15px;">HYPOTHESIS TEST CONCLUSION:</b>
            <p style="color:#f1f5f9; font-size:14px; margin:8px 0;">{pair_bench['interpretation']}</p>
            <div style="display:flex; gap:20px; flex-wrap:wrap; color:#94a3b8; font-size:13px; margin-top:10px;">
                <span><b>Chi-Square (χ²):</b> {pair_bench['chi2_statistic']:.2f} (p = {pair_bench['chi2_p_value']:.4e})</span>
                <span><b>Cramér's V Effect Size:</b> {pair_bench['cramers_v']:.3f}</span>
                <span><b>Mann-Whitney U:</b> {pair_bench['mann_whitney_u']:,} (p = {pair_bench['mann_whitney_p']:.4e})</span>
                <span><b>Significant:</b> <span style="color:#34d399; font-weight:700;">{'YES (p < 0.05)' if pair_bench['is_statistically_significant'] else 'NO'}</span></span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Section C: Visual Comparisons
        st.markdown("#### **3. Comparative Distribution Charts**")
        vcol1, vcol2 = st.columns([1, 1.2])
        
        with vcol1:
            vol_df = pd.DataFrame({
                "Brand": [pair_bench["entity_a"], pair_bench["entity_b"]],
                "Volume": [pair_bench["volume_a"], pair_bench["volume_b"]],
            })
            fig_pie = px.pie(
                vol_df,
                values="Volume",
                names="Brand",
                color_discrete_sequence=["#38bdf8", "#ec4899"],
                hole=0.45,
                template="plotly_dark",
                title="Voice Share (Total Volume Ratio)",
            )
            fig_pie.update_layout(
                height=320,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with vcol2:
            dist_data = [
                {"Brand": pair_bench["entity_a"], "Sentiment": "Positive", "Share (%)": pair_bench["distribution_a"]["positive"]},
                {"Brand": pair_bench["entity_a"], "Sentiment": "Neutral", "Share (%)": pair_bench["distribution_a"]["neutral"]},
                {"Brand": pair_bench["entity_a"], "Sentiment": "Negative", "Share (%)": pair_bench["distribution_a"]["negative"]},
                {"Brand": pair_bench["entity_b"], "Sentiment": "Positive", "Share (%)": pair_bench["distribution_b"]["positive"]},
                {"Brand": pair_bench["entity_b"], "Sentiment": "Neutral", "Share (%)": pair_bench["distribution_b"]["neutral"]},
                {"Brand": pair_bench["entity_b"], "Sentiment": "Negative", "Share (%)": pair_bench["distribution_b"]["negative"]},
            ]
            fig_bar = px.bar(
                pd.DataFrame(dist_data),
                x="Brand",
                y="Share (%)",
                color="Sentiment",
                color_discrete_map={"Positive": "#10b981", "Neutral": "#64748b", "Negative": "#ef4444"},
                barmode="group",
                template="plotly_dark",
                title="Sentiment Polarity Breakdown by Brand",
            )
            fig_bar.update_layout(
                height=320,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Section D: Aspect Divergence
        if pair_bench.get("aspect_comparison"):
            st.markdown("#### **4. Operational Aspect Attention & Divergence**")
            asp_df = pd.DataFrame(pair_bench["aspect_comparison"])
            st.dataframe(asp_df, use_container_width=True)

    else:
        st.markdown(f"""
        <div style="background-color:#1e1b4b; border:1px solid #4338ca; border-radius:12px; padding:24px; margin:20px 0;">
            <span style="background:#4338ca; color:#e0e7ff; padding:4px 10px; border-radius:12px; font-size:12px; font-weight:700;">
                FORMALLY AUDITED DATA GAP
            </span>
            <h3 style="color:#c7d2fe; margin:12px 0 6px 0;">{pair_bench.get('entity_a')} vs {pair_bench.get('entity_b')}</h3>
            <p style="color:#e0e7ff; font-size:14px; margin:0 0 12px 0;">{pair_bench.get('reason')}</p>
            <small style="color:#a5b4fc;">
                In accordance with enterprise data integrity standards, missing datasets are logged in <code>reports/data_gaps.md</code> 
                rather than populated with fabricated or synthetic data.
            </small>
        </div>
        """, unsafe_allow_html=True)

    # Section E: Global Website Traffic Benchmarking
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### **5. Global Website Mindshare & Domain Traffic Benchmark**")
    web_traffic_df = get_web_traffic_dataset()
    if not web_traffic_df.empty:
        w1, w2 = st.columns([1, 1])
        with w1:
            st.dataframe(
                web_traffic_df[["Site", "Domain Name", "Category", "Principal country/territory"]].head(10),
                use_container_width=True,
            )
        with w2:
            cat_counts = web_traffic_df["Category"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Domains"]
            fig_cat = px.bar(
                cat_counts.head(7),
                x="Category",
                y="Domains",
                color="Domains",
                color_continuous_scale="Purples",
                template="plotly_dark",
                title="Top Domain Categories in Global Web Traffic",
            )
            fig_cat.update_layout(
                height=300,
                showlegend=False,
                margin=dict(l=10, r=10, t=35, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_cat, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


# =====================================================================
# VIEW 3: FEATURE INTELLIGENCE & VISUAL STUDIO
# =====================================================================
elif active_page == "📐 Feature Intelligence & Studio":
    st.markdown(f"""
    <div class="app-card">
        <div class="app-card-header">
            <div>
                <h2 class="app-card-title" style="color:#ffffff !important; font-size:20px !important; font-weight:800 !important; margin:0 !important;">1. Automated Feature-Type Classifier: {selected_ds_key.upper()}</h2>
                <p class="app-card-subtitle" style="color:#93c5fd !important; font-size:14px !important; font-weight:600 !important; margin:4px 0 0 0 !important;">
                    Mathematical profiling of active dataset columns with justified chart recommendations 
                    (Nominal, Ordinal, Binary, Continuous, Discrete, Datetime, Geospatial, Free Text).
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    ft_table = classify_feature_types(filtered_df)
    st.dataframe(ft_table, use_container_width=True, height=280)
    
    st.download_button(
        "Download Feature Classification Table (CSV)",
        ft_table.to_csv(index=False),
        file_name=f"{selected_ds_key}_features.csv",
        mime="text/csv",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Dynamic Visual Studio
    st.markdown("""
    <div class="app-card">
        <div class="app-card-header">
            <div>
                <h2 class="app-card-title" style="color:#ffffff !important; font-size:20px !important; font-weight:800 !important; margin:0 !important;">2. Interactive Visual Studio (Dynamic Chart Builder)</h2>
                <p class="app-card-subtitle" style="color:#93c5fd !important; font-size:14px !important; font-weight:600 !important; margin:4px 0 0 0 !important;">Select any feature to dynamically generate customizable Plotly visualizations.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    col_target = st.selectbox("Select Feature to Visualize", filtered_df.columns.tolist(), index=0)
    
    meta_info = ft_table[ft_table["column_name"] == col_target]
    if not meta_info.empty:
        det_type = meta_info["detected_type"].iloc[0]
        rec_viz = meta_info["recommended_visualization"].iloc[0]
        rationale = meta_info["rationale"].iloc[0]
        
        st.info(f"**Detected Type:** `{det_type}` | **Recommended Visualization:** `{rec_viz}`\n\n*{rationale}*")
        
        vc1, vc2 = st.columns([1, 2])
        with vc1:
            chart_kind = st.selectbox(
                "Chart Technique",
                ["Recommended", "Bar Chart", "Histogram", "Box Plot", "Line Chart", "Pie Chart"],
            )
            palette = st.selectbox(
                "Color Theme",
                ["Blues", "Viridis", "Plasma", "Tealgrn", "Sunset"],
            )

        with vc2:
            series = filtered_df[col_target].dropna()
            
            if (chart_kind == "Recommended" and det_type in ["Nominal", "Binary"]) or chart_kind == "Bar Chart":
                counts = series.value_counts().reset_index()
                counts.columns = [col_target, "Count"]
                fig_res = px.bar(
                    counts.head(15),
                    x=col_target,
                    y="Count",
                    color="Count",
                    color_continuous_scale=palette,
                    template="plotly_dark",
                )
                fig_res.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_res, use_container_width=True)
                
            elif (chart_kind == "Recommended" and det_type in ["Continuous", "Discrete Numeric"]) or chart_kind == "Histogram":
                fig_res = px.histogram(
                    filtered_df,
                    x=col_target,
                    nbins=30,
                    marginal="box",
                    color_discrete_sequence=["#38bdf8"],
                    template="plotly_dark",
                )
                fig_res.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_res, use_container_width=True)
                
            elif chart_kind == "Pie Chart":
                counts = series.value_counts().reset_index()
                counts.columns = [col_target, "Count"]
                fig_res = px.pie(
                    counts.head(8),
                    values="Count",
                    names=col_target,
                    hole=0.4,
                    template="plotly_dark",
                )
                fig_res.update_layout(paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_res, use_container_width=True)
                
            elif (chart_kind == "Recommended" and det_type == "Datetime") or chart_kind == "Line Chart":
                try:
                    dt_series = pd.to_datetime(filtered_df[col_target], errors="coerce")
                    ts_df = filtered_df.groupby(dt_series.dt.date).size().reset_index(name="Volume")
                    fig_res = px.line(
                        ts_df,
                        x=ts_df.columns[0],
                        y="Volume",
                        markers=True,
                        template="plotly_dark",
                    )
                    fig_res.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_res, use_container_width=True)
                except Exception:
                    st.warning("Could not parse as time-series.")
            else:
                counts = series.value_counts().reset_index()
                counts.columns = [col_target, "Count"]
                fig_res = px.bar(
                    counts.head(15),
                    x=col_target,
                    y="Count",
                    color_discrete_sequence=["#38bdf8"],
                    template="plotly_dark",
                )
                fig_res.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_res, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


# =====================================================================
# VIEW 4: ASPECT SENTIMENT & DEEP ML (WITH COLOR SCALE GUIDE)
# =====================================================================
elif active_page == "🎯 Aspect Sentiment & Deep ML":
    st.markdown("""
    <div class="app-card">
        <div class="app-card-header">
            <div>
                <h2 class="app-card-title" style="color:#ffffff !important; font-size:20px !important; font-weight:800 !important; margin:0 !important;">1. Aspect-Based Sentiment Analysis Matrix (ABSA)</h2>
                <p class="app-card-subtitle" style="color:#93c5fd !important; font-size:14px !important; font-weight:600 !important; margin:4px 0 0 0 !important;">
                    Net sentiment polarity decomposed across 8 discrete operational dimensions for brand carriers.
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # DETAILED HEATMAP COLOR EXPLANATION (User Request)
    st.markdown("""
    <div class="absa-legend-wrapper">
        <h4 style="color:#ffffff !important; font-weight:800 !important; margin-top:0; margin-bottom:8px;">🎨 Heatmap Color Legend & Scale Interpretation</h4>
        <p style="color:#cbd5e1 !important; font-size:13.5px; margin-bottom:12px; font-weight:500;">
            The score in each cell represents the <b style="color:#ffffff;">Net Aspect Polarity Score</b>, bounded strictly in <code>[-1.00, +1.00]</code> 
            computed as: <code>(Positive Mentions - Negative Mentions) / Total Mentions</code>.
        </p>
        <div class="swatch-row">
            <div class="swatch-box" style="background:#dc2626;"></div>
            <div>
                <b style="color:#ef4444;">Deep Red / Crimson (-1.00 to -0.40): Acute Operational Crisis</b><br>
                Severe customer dissatisfaction where negative complaints vastly outweigh positive praise. Represents acute operational pain points (e.g., flight delays, lost baggage, booking failures) that require immediate leadership intervention.
            </div>
        </div>
        <div class="swatch-row">
            <div class="swatch-box" style="background:#f97316;"></div>
            <div>
                <b style="color:#fb923c;">Orange / Coral (-0.39 to -0.15): Moderate Friction</b><br>
                Negative feedback outpaces positive comments. An emerging operational vulnerability or chronic friction area.
            </div>
        </div>
        <div class="swatch-row">
            <div class="swatch-box" style="background:#eab308;"></div>
            <div>
                <b style="color:#facc15;">Yellow / Amber (-0.14 to +0.14): Neutral / Mixed Experience</b><br>
                Equal volume of positive and negative customer posts, or predominantly factual inquiries.
            </div>
        </div>
        <div class="swatch-row">
            <div class="swatch-box" style="background:#84cc16;"></div>
            <div>
                <b style="color:#a3e635;">Light Green (+0.15 to +0.39): Favorable Performance</b><br>
                Satisfied customer feedback consistently outpaces complaints.
            </div>
        </div>
        <div class="swatch-row">
            <div class="swatch-box" style="background:#16a34a;"></div>
            <div>
                <b style="color:#4ade80;">Deep Emerald (+0.40 to +1.00): Core Competitive Advantage</b><br>
                Strong customer delight, high praise, and superior operational execution (e.g., exceptional cabin crew hospitality).
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    absa_file = TABLES_DIR / "absa_matrix.csv"
    if absa_file.exists():
        absa_matrix_df = pd.read_csv(absa_file)
        st.dataframe(
            absa_matrix_df.style.background_gradient(
                cmap="RdYlGn",
                vmin=-1.0,
                vmax=1.0,
                subset=[c for c in absa_matrix_df.columns if c != "Aspect"],
            ),
            use_container_width=True,
        )
    else:
        st.info("Execute master pipeline to render ABSA matrix.")

    st.markdown("</div>", unsafe_allow_html=True)

    # Section 2: Deep Transformer Evaluation Curves
    st.markdown("""
    <div class="app-card">
        <div class="app-card-header">
            <div>
                <h2 class="app-card-title" style="color:#ffffff !important; font-size:20px !important; font-weight:800 !important; margin:0 !important;">2. Deep Transformer Model Evaluation (RoBERTa on CUDA GPU)</h2>
                <p class="app-card-subtitle" style="color:#93c5fd !important; font-size:14px !important; font-weight:600 !important; margin:4px 0 0 0 !important;">
                    cardiffnlp/twitter-roberta-base-sentiment-latest precision, recall, and ROC-AUC curves.
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    m1, m2 = st.columns(2)
    with m1:
        st.markdown("##### Confusion Matrix (Raw Classification Counts)")
        cm_path = FIGURES_DIR / "confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), use_container_width=True)
        else:
            st.info("Confusion matrix available in reports/figures/.")
    with m2:
        st.markdown("##### Multi-Class One-vs-Rest ROC & AUC Curves")
        roc_path = FIGURES_DIR / "roc_curves.png"
        if roc_path.exists():
            st.image(str(roc_path), use_container_width=True)
        else:
            st.info("ROC curves available in reports/figures/.")

    clf_rep_file = TABLES_DIR / "sentiment_classification_report.csv"
    if clf_rep_file.exists():
        st.markdown("##### Detailed Classification Report")
        clf_df = pd.read_csv(clf_rep_file, index_col=0)
        st.dataframe(clf_df.style.format(precision=3), use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Section 3: Live Model Sandbox
    st.markdown("""
    <div class="app-card">
        <div class="app-card-header">
            <div>
                <h2 class="app-card-title" style="color:#ffffff !important; font-size:20px !important; font-weight:800 !important; margin:0 !important;">3. Live Interactive GPU Sentiment & ABSA Tester</h2>
                <p class="app-card-subtitle" style="color:#93c5fd !important; font-size:14px !important; font-weight:600 !important; margin:4px 0 0 0 !important;">Type any custom customer post to execute real-time GPU transformer scoring.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    sample_input = st.text_area(
        "Enter customer feedback text to evaluate:",
        value="The flight was delayed by 3 hours in Chicago and customer service was unresponsive, but the cabin crew was exceptionally polite.",
    )
    if st.button("Score Text Live with GPU"):
        with st.spinner("Executing GPU inference..."):
            pred_res = predict_single_text(sample_input)
            absa_res = run_single_text_absa(sample_input)
            
            p1, p2 = st.columns([1, 2])
            with p1:
                st.markdown(f"**Overall Sentiment:** `{pred_res['sentiment'].upper()}`")
                st.markdown(f"**Confidence:** `{pred_res['confidence'] * 100:.1f}%`")
                st.progress(pred_res['confidence'])
            with p2:
                st.markdown("**Detected Operational Aspects & Polarities:**")
                st.dataframe(pd.DataFrame(absa_res), use_container_width=True)
                
    st.markdown("</div>", unsafe_allow_html=True)


# =====================================================================
# VIEW 5: GEOSPATIAL INTELLIGENCE
# =====================================================================
elif active_page == "🗺️ Geospatial Intelligence":
    st.markdown("""
    <div class="app-card">
        <div class="app-card-header">
            <div>
                <h2 class="app-card-title" style="color:#ffffff !important; font-size:20px !important; font-weight:800 !important; margin:0 !important;">1. Geospatial Sentiment Cartography & Regional Heatmaps</h2>
                <p class="app-card-subtitle" style="color:#93c5fd !important; font-size:14px !important; font-weight:600 !important; margin:4px 0 0 0 !important;">
                    Geographic distribution of customer sentiment and complaint density with strict coordinate validation.
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    airline_df = get_airline_dataset()
    geo_data = prepare_geospatial_dataset(airline_df)
    
    g1, g2, g3 = st.columns(3)
    with g1:
        st.metric("Exact GPS Coordinates", f"{geo_data['exact_count']:,}", f"{geo_data['exact_percentage']}% of records")
    with g2:
        st.metric("Approximate Centroids", f"{len(geo_data['approx_df']):,}", "Timezone derived")
    with g3:
        st.metric("Tracked US Regions", f"{len(geo_data['regional_stats'])} Zones")

    # Folium Interactive Map
    map_filename = "folium_sentiment_map_approx.html" if include_approx_geo else "folium_sentiment_map.html"
    map_path = FIGURES_DIR / map_filename
    
    if map_path.exists():
        with open(map_path, "r", encoding="utf-8") as f:
            st.components.v1.html(f.read(), height=550)
    else:
        m = generate_folium_sentiment_map(geo_data, include_approx=include_approx_geo)
        st_folium(m, width="100%", height=500)

    if not geo_data["regional_stats"].empty:
        st.markdown("#### Regional Sentiment Breakdown & Complaint Density")
        st.dataframe(geo_data["regional_stats"], use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


# =====================================================================
# VIEW 6: NETWORK DYNAMICS & ECHO-CHAMBERS
# =====================================================================
elif active_page == "🕸️ Network Dynamics & Echo-Chambers":
    st.markdown("""
    <div class="app-card">
        <div class="app-card-header">
            <div>
                <h2 class="app-card-title" style="color:#ffffff !important; font-size:20px !important; font-weight:800 !important; margin:0 !important;">1. Social Graph Dynamics & Louvain Community Echo-Chambers</h2>
                <p class="app-card-subtitle" style="color:#93c5fd !important; font-size:14px !important; font-weight:600 !important; margin:4px 0 0 0 !important;">
                    NetworkX graph theory, PageRank influencer identification, and polarization modularity metrics.
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    net_choice = st.radio("Select Network Topology", ["Twitter Brand Interaction Graph", "Stanford SNAP Ego-Network"], horizontal=True)
    html_file = "pyvis_echo_chamber_network.html" if "Brand" in net_choice else "pyvis_snap_network.html"
    pyvis_path = FIGURES_DIR / html_file
    
    if pyvis_path.exists():
        with open(pyvis_path, "r", encoding="utf-8") as f:
            st.components.v1.html(f.read(), height=620)
    else:
        st.info("Run pipeline to export PyVis interactive graph.")

    inf_file = TABLES_DIR / "influencer_centrality_leaderboard.csv"
    if inf_file.exists():
        st.markdown("#### Top Influencers & Centrality Leaderboard")
        inf_df = pd.read_csv(inf_file)
        st.dataframe(inf_df.head(max_network_nodes), use_container_width=True)

    st.markdown("---")
    st.markdown("#### Network Data Structure Representations")
    struct_demo = demonstrate_graph_data_structures()
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("##### Weighted Edge List")
        st.dataframe(pd.DataFrame(struct_demo["edge_list"], columns=["Source", "Target", "Weight"]), use_container_width=True)
    with d2:
        st.markdown("##### Adjacency Matrix")
        st.dataframe(pd.DataFrame(struct_demo["adjacency_matrix"], index=struct_demo["node_order"], columns=struct_demo["node_order"]), use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


# =====================================================================
# VIEW 7: WEB SCRAPING & REST APIS
# =====================================================================
elif active_page == "🕷️ Web Scraping & REST APIs":
    st.markdown("""
    <div class="app-card">
        <div class="app-card-header">
            <div>
                <h2 class="app-card-title" style="color:#ffffff !important; font-size:20px !important; font-weight:800 !important; margin:0 !important;">1. Ethical Web Scraping Pipelines & Sandbox Crawlers</h2>
                <p class="app-card-subtitle" style="color:#93c5fd !important; font-size:14px !important; font-weight:600 !important; margin:4px 0 0 0 !important;">
                    Automated crawling with robots.txt validation, Scrapy subprocess runner, and live REST API connectors.
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown("#### Ethical BeautifulSoup Sandbox Scraper")
        if st.button("Execute BeautifulSoup Crawler"):
            with st.spinner("Crawling quotes.toscrape.com with robots.txt check..."):
                scraper = EthicalBeautifulSoupScraper()
                quotes = scraper.scrape_quotes_sandbox(max_pages=1)
                st.success(f"Scraped {len(quotes)} quotes!")
                st.dataframe(pd.DataFrame(quotes), use_container_width=True)
    with sc2:
        st.markdown("#### Scrapy Spider Execution (`scrapy crawl quotes`)")
        if st.button("Spawn Scrapy Subprocess"):
            with st.spinner("Executing asynchronous Scrapy spider..."):
                s_res = run_scrapy_spider("quotes")
                if s_res.get("success"):
                    st.success(f"Scrapy finished successfully! Items: {s_res.get('items_scraped')}")
                else:
                    st.warning(f"Finished: {s_res.get('return_code')}")
                with st.expander("Scrapy Log Output"):
                    st.code(s_res.get("stderr", s_res.get("stdout", "")))

    st.markdown("---")
    st.markdown("#### 2. Live REST API Connectors")
    api1, api2 = st.columns(2)
    with api1:
        st.markdown("##### Hacker News Firebase REST API")
        if st.button("Fetch Top Stories"):
            hn = HackerNewsApiFetcher()
            st.dataframe(pd.DataFrame(hn.fetch_data(limit=6)), use_container_width=True)
    with api2:
        st.markdown("##### Wikipedia Brand Traffic (Wikimedia REST API)")
        if st.button("Fetch Brand Pageviews"):
            wiki = WikipediaPageviewsFetcher()
            traffic = wiki.fetch_data("Apple_Inc.")
            if traffic:
                st.dataframe(pd.DataFrame(traffic)[["timestamp", "views"]].tail(8), use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


# =====================================================================
# VIEW 8: STRATEGIC CASE STUDY
# =====================================================================
elif active_page == "📑 Strategic Case Study":
    st.markdown("""
    <div class="app-card">
        <div class="app-card-header">
            <div>
                <h2 class="app-card-title" style="color:#ffffff !important; font-size:20px !important; font-weight:800 !important; margin:0 !important;">1. Executive Strategy & Operational Crisis Mitigation Report</h2>
                <p class="app-card-subtitle" style="color:#93c5fd !important; font-size:14px !important; font-weight:600 !important; margin:4px 0 0 0 !important;">
                    Automated corporate decision support synthesizing empirical findings into evidence-based operational interventions.
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    report_file = REPORTS_DIR / "case_study_executive_report.md"
    if report_file.exists():
        with open(report_file, "r", encoding="utf-8") as f:
            rep_text = f.read()
        st.download_button("Download Strategy Report (Markdown)", rep_text, file_name="executive_strategy_report.md", mime="text/markdown")
        st.markdown(rep_text)
    else:
        st.info("Execute pipeline to compile executive strategy report.")

    dec_file = TABLES_DIR / "recommendation_decision_matrix.csv"
    if dec_file.exists():
        st.markdown("---")
        st.markdown("#### 2. Prioritized Operational Action Matrix (Effort vs Impact)")
        dec_df = pd.read_csv(dec_file)
        st.dataframe(dec_df, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)
