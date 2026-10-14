# BrandPulse AI: Cross-Platform Brand Perception & Echo-Chamber Engine

**BrandPulse AI** is an enterprise-grade social listening, brand perception, and network dynamics platform powered by PyTorch (CUDA GPU), HuggingFace Transformers, NetworkX, and Streamlit. It delivers a full working production pipeline where multi-source social data is ingested, normalized, modeled, and converted into rich interactive visualizations, statistical tests, and strategic executive intelligence.

---

## 1. System Architecture & End-to-End Pipeline

```
Raw Multi-Platform Data Ingestion (Airlines, E-Commerce, Tech Reviews, Social Graphs)
                                      │
                                      ▼
             Data Preprocessing, Cleaning & Coordinate Validation
                        (data/processed/ & cache/)
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         ▼                            ▼                            ▼
Feature Profiling &          GPU Deep Learning &          Social Network Dynamics &
Exploratory Visualizations   Aspect-Based Sentiment       Echo-Chamber Homophily
• Auto Data Type Classifier  • RoBERTa FP16 Inference     • PageRank & Centrality
• Cramér's V Heatmaps        • 8-Dimension ABSA Matrix    • Louvain Community Detection
• Pareto 80/20 Drivers       • Multi-Class ROC Curves     • Interactive PyVis Topology
• IQR & Z-score Outliers     • Live Interactive Tester    • Echo-Chamber Homophily
         │                            │                            │
         └────────────────────────────┼────────────────────────────┘
                                      │
         ┌────────────────────────────┴────────────────────────────┐
         ▼                                                         ▼
Geospatial Intelligence & Mapping                        Competitor Benchmarking & Strategy
• Folium Exact GPS MarkerClusters                        • Cross-Brand Sentiment Differential
• Complaint Density HeatMaps                             • Chi-Square & Mann-Whitney U Tests
• Regional Vulnerability Aggregation                     • Executive Prioritized Action Matrix
                                      │
                                      ▼
           Interactive Streamlit Workspace & Visual Studio (app.py)
```

---

## 2. Hardware & System Requirements
- **OS**: Windows 10/11 or Linux
- **Python**: 3.10+ (Tested on Python 3.12.5)
- **GPU Acceleration**: NVIDIA GPU with CUDA support (e.g., RTX 4050 Laptop GPU, 6GB VRAM, Compute Capability 8.9).
  - Automatically falls back to CPU if no CUDA GPU is detected.
  - Automatic FP16 mixed-precision and dynamic batch-size fallback (`32 -> 16 -> 8 -> 4 -> 1`) prevents CUDA OOM.

---

## 3. Quickstart & Launch Sequence

### Step 1: Environment Setup & Dependencies
```bash
# Create virtual environment
python -m venv .venv

# Activate environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1
# (Linux/macOS: source .venv/bin/activate)

# Install CUDA-enabled PyTorch & Torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install all workspace dependencies
pip install -r requirements.txt

# Download spaCy English linguistic model
python -m spacy download en_core_web_sm
```

### Step 2: Verify Hardware Acceleration
```bash
python scripts/verify_gpu.py
```
*Confirms GPU name, available VRAM, CUDA capability, and FP16 Tensor Core acceleration.*

### Step 3: Data Ingestion (Real Datasets)
```bash
# Download verified Kaggle datasets (idempotent, validates schemas, writes MANIFEST.json)
python scripts/download_kaggle_data.py

# Download Stanford SNAP Twitter graphs (streams and extracts twitter_combined & Higgs networks)
python scripts/download_snap_graph.py
```

### Step 4: Run Master Analytics Pipeline
```bash
python scripts/run_pipeline.py
```
*Executes all data processing, machine learning models, statistical tests, graph algorithms, and exports all visualizations to `reports/figures/`, tables to `reports/tables/`, and processed datasets to `data/processed/`.*

### Step 5: Run Automated Tests
```bash
pytest
```
*Validates coordinate parsing, text cleaning, metric shapes, graph analysis, and scraper robots.txt logic.*

### Step 6: Launch Interactive Dashboard Studio
```bash
streamlit run app.py --server.port 8501
```
Open **http://localhost:8501** in your browser to interact with the full analytics suite.

---

## 4. Pipeline Modules & Output Artifacts

| Analytical Phase | Core Python Module | Streamlit Dashboard Tab | Generated Visualizations & Tables |
| :--- | :--- | :--- | :--- |
| **Feature Intelligence & Profiling** | [`modules/data_processor.py`](modules/data_processor.py) | **Tab 2: Feature Intelligence & Visual Studio** | `reports/tables/feature_types.csv`<br/>`reports/tables/correlation_matrix.csv`<br/>`reports/figures/seaborn_correlation_heatmap.png` |
| **Statistical Diagnostics & Pareto** | [`modules/data_processor.py`](modules/data_processor.py) | **Tab 1 & Tab 2** | `reports/tables/pareto_complaint_drivers.csv`<br/>`reports/tables/outliers_iqr_zscore.csv`<br/>`reports/figures/pareto_complaints.png` |
| **Deep Transformer Sentiment (GPU)** | [`modules/sentiment_engine.py`](modules/sentiment_engine.py) | **Tab 3: Aspect Sentiment & Deep ML Models** | `reports/tables/sentiment_classification_report.csv`<br/>`reports/figures/confusion_matrix.png`<br/>`reports/figures/roc_curves.png`<br/>`reports/figures/precision_recall_curves.png` |
| **Aspect-Based Sentiment (ABSA)** | [`modules/absa_engine.py`](modules/absa_engine.py) | **Tab 3: Aspect Sentiment & Deep ML Models** | `reports/tables/absa_matrix.csv`<br/>`reports/tables/absa_summary_metrics.csv`<br/>Interactive Aspect Heatmap |
| **Social Graph Dynamics & Echo-Chambers** | [`modules/network_analyzer.py`](modules/network_analyzer.py) | **Tab 5: Social Network Dynamics** | `reports/tables/influencer_centrality_leaderboard.csv`<br/>`reports/tables/community_echo_metrics.csv`<br/>`reports/figures/pyvis_echo_chamber_network.html`<br/>`reports/figures/pyvis_snap_network.html` |
| **Geospatial Intelligence** | [`modules/geospatial_mapper.py`](modules/geospatial_mapper.py) | **Tab 4: Geospatial Sentiment & Regional Maps** | `reports/tables/regional_sentiment_stats.csv`<br/>`reports/figures/folium_sentiment_map.html`<br/>`reports/figures/folium_sentiment_map_approx.html` |
| **Competitor Benchmarking** | [`modules/competitor_benchmark.py`](modules/competitor_benchmark.py) | **Tab 1: Executive Overview** | `reports/tables/competitor_significance_tests.csv`<br/>Differential Net Sentiment cards |
| **Web Scraping & REST APIs** | [`modules/scrapers_demo.py`](modules/scrapers_demo.py) | **Tab 6: Web Scraping & REST APIs** | `reports/tables/scraped_sandbox_quotes.csv`<br/>Live Hacker News REST feed |
| **Executive Decision Support** | [`modules/case_study.py`](modules/case_study.py) | **Tab 7: Strategic Case Study** | `reports/tables/recommendation_decision_matrix.csv`<br/>`reports/case_study_executive_report.md` |

---

## 5. Verified Datasets & Data Manifest

| Domain | Dataset Identifier | Source / License | Volume / Records | Key Attributes |
| :--- | :--- | :--- | :--- | :--- |
| **US Aviation Sentiment** | `crowdflower/twitter-airline-sentiment` | CC BY-NC-SA 4.0 | 14,640 rows | Sentiments, negative reasons, coordinates, timezones |
| **Comparative Shopping** | `jocelyndumlao/shoppingappreviews-dataset` | CC0 Public Domain | 99,021 rows | Amazon vs Myntra app ratings and reviews |
| **Indian Domestic Airlines** | `jagathratchakan/indian-airlines-customer-reviews` | Community Open | 2,229 rows | IndiGo vs Air India service reviews |
| **Consumer Electronics** | `grikomsn/amazon-cell-phones-reviews` | CC0 Public Domain | 67,986 rows | Apple iPhone vs Samsung Galaxy reviews |
| **Cyberbullying Multi-Class** | `saurabhshahane/cyberbullying-dataset` | CC0 Public Domain | 17,823 rows | Multi-class NLP toxicity benchmark |
| **Global Website Traffic** | `thedevastator/the-top-websites-in-the-world` | ODbL | 50 domains | Domain visit volumes and traffic trends |
| **Stanford Twitter Graph** | `twitter_combined.txt.gz` (SNAP) | Stanford University | 2,420,766 edges | Multi-ego social interaction topology |
| **Higgs Retweet & Mention** | `higgs-*.edgelist.gz` (SNAP) | Stanford University | 478,950 edges | Social news diffusion and retweet networks |

---

## 6. Enterprise Data Integrity & Audit Guarantees
1. **Real Data Guarantee**: Zero synthetic or fabricated metrics. If a data source has gaps (e.g., healthcare reviews or higher education firehoses), they are formally audited and logged in `reports/data_gaps.md` rather than guessed.
2. **Geospatial Rigor**: Exact GPS coordinates (~7.0% of airline records) are strictly segregated from timezone-derived regional centroids with user toggle controls in the UI.
3. **User Control**: Full interactive controls including custom CSV upload, live model inference on arbitrary text, adjustable sentiment confidence thresholds, and network node sizing.
4. **Ethical Web Crawling**: Scrapers strictly enforce `robots.txt` compliance, rate limiting, and execute only on legal sandbox environments.
