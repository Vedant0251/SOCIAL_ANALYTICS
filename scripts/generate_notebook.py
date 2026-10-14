"""BrandPulse AI - Master Interactive Notebook Generator."""

import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell("""# BrandPulse AI: Cross-Platform Brand Perception & Echo-Chamber Engine
**Master Interactive Research & Analytics Notebook**

End-to-end data science and social analytics pipeline powered by verified real-world datasets, GPU-accelerated NLP, and social graph analytics."""),
    nbf.v4.new_markdown_cell("""## 0. Hardware & CUDA Verification
Detects NVIDIA GPU and confirms FP16 Tensor Core acceleration support."""),
    nbf.v4.new_code_cell("""import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path.cwd()))

from scripts.verify_gpu import get_gpu_info
info = get_gpu_info()
for k, v in info.items():
    print(f"{k:25s}: {v}")"""),
    nbf.v4.new_markdown_cell("""## Stage 1: Automatic Feature-Type Classifier & Recommended Visualizations
Classifies features into Nominal, Ordinal, Binary, Continuous, Datetime, Geospatial, Free Text."""),
    nbf.v4.new_code_cell("""from modules.data_processor import load_airline_sentiment, classify_feature_types
df_airline = load_airline_sentiment()
feature_types = classify_feature_types(df_airline)
feature_types.head(10)"""),
    nbf.v4.new_markdown_cell("""## Stage 2: Trends, Correlations, Patterns & Outlier Diagnostics
Computes seasonal decomposition, Cramér's V associations, Pareto 80/20 analysis, and IQR/Z-score/Isolation Forest outliers."""),
    nbf.v4.new_code_cell("""from modules.data_processor import (
    compute_sentiment_trends,
    compute_correlations_and_associations,
    compute_pareto_complaints,
    detect_outliers_comprehensive
)

trends = compute_sentiment_trends(df_airline)
print('Trend Interpretation:', trends['interpretation'])

corrs = compute_correlations_and_associations(df_airline)
print('Associations Interpretation:', corrs['interpretation'])

pareto = compute_pareto_complaints(df_airline)
print('Vital few complaints causing 80% volume:')
display(pareto[pareto['is_vital_few']])

outliers = detect_outliers_comprehensive(df_airline)
print('Outlier Diagnostics:', outliers['interpretation'])"""),
    nbf.v4.new_markdown_cell("""## Stage 3: Statistical Visualizations & Dashboards
Correlation heatmaps, Pareto distributions, and visual diagnostics."""),
    nbf.v4.new_code_cell("""import matplotlib.pyplot as plt
import matplotlib.image as mpimg

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.imshow(mpimg.imread('reports/figures/seaborn_correlation_heatmap.png'))
ax1.axis('off')
ax1.set_title("Cramer's V Associations")

ax2.imshow(mpimg.imread('reports/figures/pareto_complaints.png'))
ax2.axis('off')
ax2.set_title("Pareto (80/20) Complaints")
plt.tight_layout()
plt.show()"""),
    nbf.v4.new_markdown_cell("""## Stage 4: Geospatial Location Sentiment & Regional Heatmaps
Separates exact GPS coordinates (~7.0%) from timezone-derived regional centroids."""),
    nbf.v4.new_code_cell("""from modules.geospatial_mapper import prepare_geospatial_dataset
geo_data = prepare_geospatial_dataset(df_airline)
print(f"Exact GPS records: {geo_data['exact_count']:,} ({geo_data['exact_percentage']}%) | Approx centroids: {len(geo_data['approx_df']):,}")
display(geo_data['regional_stats'])"""),
    nbf.v4.new_markdown_cell("""## Stage 5: Competitor Benchmarking & Statistical Significance
Cross-brand comparisons with Chi-Square tests of independence and Mann-Whitney U rank-biserial tests."""),
    nbf.v4.new_code_cell("""from modules.competitor_benchmark import execute_all_competitor_benchmarks
benchmarks = execute_all_competitor_benchmarks()
for pair_id, bench in benchmarks.items():
    print(f"=== {pair_id.upper()}: {bench.get('entity_a')} vs {bench.get('entity_b')} ===")
    if bench.get('status') == 'available':
        print(f"  Voice Share: {bench['voice_share_a']}% vs {bench['voice_share_b']}%")
        print(f"  NSS: {bench['net_sentiment_a']:+.1f} vs {bench['net_sentiment_b']:+.1f}")
        print(f"  Chi2 = {bench['chi2_statistic']} (p = {bench['chi2_p_value']:.4e}) | Sig: {bench['is_statistically_significant']}")
    else:
        print(f"  Status: {bench.get('status')} | Reason: {bench.get('reason')}")"""),
    nbf.v4.new_markdown_cell("""## Stage 6: Web Scraping Pipelines & Ethical Ingestion
Demonstrating robots.txt compliance on sandbox targets, Scrapy, and REST API fetchers."""),
    nbf.v4.new_code_cell("""from modules.scrapers_demo import EthicalBeautifulSoupScraper, HackerNewsApiFetcher
scraper = EthicalBeautifulSoupScraper()
quotes = scraper.scrape_quotes_sandbox(max_pages=1)
print(f"Scraped {len(quotes)} quotes from quotes.toscrape.com")

hn = HackerNewsApiFetcher()
stories = hn.fetch_data(limit=3)
print(f"Fetched {len(stories)} stories from Hacker News REST API")"""),
    nbf.v4.new_markdown_cell("""## Stage 7: Aspect-Based Sentiment Analysis & Deep Transformer Inference
cardiffnlp RoBERTa on NVIDIA CUDA GPU with Precision, Recall, F1, Confusion Matrix, and ROC-AUC."""),
    nbf.v4.new_code_cell("""from modules.sentiment_engine import evaluate_airline_sentiment, predict_single_text
from modules.absa_engine import compute_aspect_brand_matrix, run_single_text_absa

metrics = evaluate_airline_sentiment(df_airline)
print('RoBERTa Accuracy:', metrics['accuracy'], '| Macro F1:', metrics['macro_f1'], '| Macro ROC-AUC:', metrics['macro_roc_auc'])

matrix = compute_aspect_brand_matrix(df_airline)
print('Aspect x Brand Matrix:')
display(matrix)

sample_text = 'The flight was delayed by 3 hours in Chicago but the flight attendants were exceptionally polite.'
print('Live Prediction:', predict_single_text(sample_text))
print('Live ABSA Decomposition:', run_single_text_absa(sample_text))"""),
    nbf.v4.new_markdown_cell("""## Stage 8: Social Network Data Structures (NetworkX)
Demonstrating Edge Lists, Adjacency Lists, Adjacency Matrices, and Node Attributes."""),
    nbf.v4.new_code_cell("""from modules.network_analyzer import demonstrate_graph_data_structures
demo = demonstrate_graph_data_structures()
print('Graph Nodes:', demo['num_nodes'], '| Edges:', demo['num_edges'])
print('Adjacency List:', demo['adjacency_list'])"""),
    nbf.v4.new_markdown_cell("""## Stage 9: Centrality Leaderboard & Louvain Community Echo-Chambers
Degree, Betweenness, PageRank, Modularity, and Homophily Index."""),
    nbf.v4.new_code_cell("""from modules.network_analyzer import build_airline_interaction_graph, analyze_graph_dynamics
airline_g = build_airline_interaction_graph(df_airline, max_edges=1000)
net_summary = analyze_graph_dynamics(airline_g)
print(f"Modularity: {net_summary['modularity_score']} | Homophily Index: {net_summary['homophily_index']}")
print('Top Influencers:')
display(net_summary['top_influencers'].head(5))"""),
    nbf.v4.new_markdown_cell("""## Stage 10: Strategic Executive Decision Engine & Action Plan
Auto-generated executive strategic decision matrix and operational recommendations."""),
    nbf.v4.new_code_cell("""from modules.case_study import generate_strategic_case_study
case_study = generate_strategic_case_study(
    pareto_df=pareto,
    absa_df=matrix,
    benchmarks=benchmarks,
    network_summary=net_summary,
    geo_stats=geo_data['regional_stats']
)
print('Executive Report path:', case_study['markdown_file'])
display(pd.DataFrame(case_study['decision_matrix']))"""),
]

nb.cells = cells

with open("BrandPulse_Master.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("BrandPulse_Master.ipynb written successfully!")
