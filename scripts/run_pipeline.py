"""BrandPulse AI - Master End-to-End Analytics Pipeline.

Executes all analytics and machine learning modules in automated sequence:
- Stage 1: Data Ingestion, Schema Profiling & Statistical Diagnostics
- Stage 2: Deep Transformer Sentiment Analysis (RoBERTa on CUDA GPU) & ML Benchmarks
- Stage 3: Aspect-Based Sentiment Analysis (ABSA) & Multi-Dimension Polarity Matrix
- Stage 4: Social Graph Dynamics, Centrality Leaderboards & Echo-Chamber Homophily
- Stage 5: Geospatial Intelligence & Interactive Folium/Plotly Cartography
- Stage 6: Cross-Platform Competitor Benchmarking & Statistical Hypothesis Testing
- Stage 7: Ethical Web Scraping & REST API Ingestion Pipelines
- Stage 8: Strategic Decision Support & Executive Action Plan

Persists all artifacts to data/processed/, cache/, and reports/.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
import time

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    CACHE_DIR,
    DATA_PROCESSED_DIR,
    FIGURES_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    TABLES_DIR,
)
from modules.absa_engine import (
    compute_aspect_brand_matrix,
    evaluate_absa_proxy_ground_truth,
)
from modules.case_study import generate_strategic_case_study
from modules.competitor_benchmark import execute_all_competitor_benchmarks
from modules.data_processor import (
    classify_feature_types,
    compute_correlations_and_associations,
    compute_pareto_complaints,
    compute_sentiment_trends,
    detect_outliers_comprehensive,
    load_airline_sentiment,
    load_cyberbullying_dataset,
)
from modules.geospatial_mapper import (
    generate_folium_sentiment_map,
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
    run_scrapy_spider,
)
from modules.sentiment_engine import (
    benchmark_cyberbullying_baseline,
    evaluate_airline_sentiment,
)
from scripts.verify_gpu import get_gpu_info

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("run_pipeline")


def run_pipeline() -> int:
    """Execute master analytics pipeline."""
    start_time = time.time()
    print("=" * 70)
    print(" BRANDPULSE AI: MASTER END-TO-END ANALYTICS PIPELINE")
    print("=" * 70)

    # 0. Hardware Inspection
    logger.info("Initializing Hardware & CUDA Engine...")
    gpu_info = get_gpu_info()
    print(f"Hardware: {gpu_info.get('device_name')} (CUDA: {gpu_info.get('cuda_available')})")

    # 1. Data Processor & Schema Profiling
    logger.info("\n--- Phase 1: Data Ingestion, Schema Profiling & Statistical Diagnostics ---")
    airline_df = load_airline_sentiment()
    if airline_df.empty:
        logger.error("Airline dataset missing. Please run scripts/download_kaggle_data.py first.")
        return 1

    feature_types = classify_feature_types(airline_df)
    logger.info(f"Classified {len(feature_types)} features with visualization recommendations.")

    trends = compute_sentiment_trends(airline_df)
    correlations = compute_correlations_and_associations(airline_df)
    pareto_df = compute_pareto_complaints(airline_df)
    outliers = detect_outliers_comprehensive(airline_df)
    logger.info("Computed temporal trends, Cramér's V associations, Pareto 80/20 drivers, and outlier diagnostics.")

    # 2. GPU Sentiment Engine & ML Benchmark
    logger.info("\n--- Phase 2: Deep Transformer Sentiment & ML Benchmark ---")
    sentiment_metrics = evaluate_airline_sentiment(airline_df, sample_size=1500)
    logger.info(
        f"RoBERTa Evaluation -> Accuracy: {sentiment_metrics['accuracy']:.4f}, "
        f"Macro F1: {sentiment_metrics['macro_f1']:.4f}, Macro ROC-AUC: {sentiment_metrics['macro_roc_auc']:.4f}"
    )

    cyber_df = load_cyberbullying_dataset()
    if not cyber_df.empty:
        cyber_baseline = benchmark_cyberbullying_baseline(cyber_df, sample_size=3000)
        logger.info(f"Multi-Class NLP Baseline -> Weighted F1: {cyber_baseline['weighted_f1']:.4f}")

    # 3. ABSA Engine
    logger.info("\n--- Phase 3: Aspect-Based Sentiment Analysis (ABSA) ---")
    absa_matrix = compute_aspect_brand_matrix(airline_df)
    absa_eval = evaluate_absa_proxy_ground_truth(airline_df, sample_size=2000)
    logger.info(
        f"ABSA Matrix computed across {absa_matrix.shape[0]} operational dimensions. "
        f"Aspect Ground Truth Hit Rate: {absa_eval['aspect_recall_hit_rate']:.4f}, Macro F1: {absa_eval['macro_f1']:.4f}"
    )

    # 4. Social Network Dynamics & Echo-Chambers
    logger.info("\n--- Phase 4: Social Network Dynamics & Echo-Chamber Analysis ---")
    data_structs_demo = demonstrate_graph_data_structures()
    logger.info(f"Graph data structures initialized ({data_structs_demo['num_nodes']} demonstration nodes).")

    # Build Airline graph and SNAP graph
    airline_g = build_airline_interaction_graph(airline_df, max_edges=3000)
    network_summary = analyze_graph_dynamics(airline_g)
    logger.info(
        f"Airline Interaction Graph -> Modularity: {network_summary['modularity_score']:.4f}, "
        f"Within-Community Edge Ratio: {network_summary['within_edge_ratio']:.4f}, "
        f"Homophily Index: {network_summary['homophily_index']:.4f}"
    )
    generate_pyvis_network_html(network_summary["graph"], "pyvis_echo_chamber_network.html")

    # Also build SNAP sub-graph
    snap_g = load_snap_twitter_subgraph(max_nodes=600, k_core=10)
    snap_summary = analyze_graph_dynamics(snap_g)
    generate_pyvis_network_html(snap_summary["graph"], "pyvis_snap_network.html")
    logger.info("Exported interactive PyVis network visualizers to reports/figures/.")

    # 5. Geospatial Mapping & Regional Cartography
    logger.info("\n--- Phase 5: Geospatial Sentiment & Regional Heatmaps ---")
    geo_data = prepare_geospatial_dataset(airline_df)
    generate_folium_sentiment_map(geo_data, include_approx=False, output_filename="folium_sentiment_map.html")
    generate_folium_sentiment_map(
        geo_data, include_approx=True, output_filename="folium_sentiment_map_approx.html"
    )
    logger.info(f"Exported interactive Folium maps ({geo_data['exact_count']} exact GPS points).")

    # 6. Competitor Benchmarking & Hypothesis Testing
    logger.info("\n--- Phase 6: Competitor Benchmarks & Statistical Significance ---")
    benchmarks = execute_all_competitor_benchmarks()
    avail_count = sum(1 for b in benchmarks.values() if b.get("status") == "available")
    logger.info(f"Competitor benchmarks completed for 5 pairs ({avail_count} active with real data, data gaps audited).")

    # 7. Scrapers & REST API Ingestion Pipelines
    logger.info("\n--- Phase 7: Web Scraping & REST API Ingestion Pipelines ---")
    bs4_scraper = EthicalBeautifulSoupScraper()
    quotes = bs4_scraper.scrape_quotes_sandbox(max_pages=1)
    logger.info(f"Ethical BeautifulSoup scraper extracted {len(quotes)} records with robots.txt enforcement.")

    hn_fetcher = HackerNewsApiFetcher()
    hn_stories = hn_fetcher.fetch_data(limit=5)
    logger.info(f"Hacker News REST API ingested {len(hn_stories)} top stories.")

    scrapy_res = run_scrapy_spider(spider_name="quotes", timeout_sec=15)
    logger.info(f"Scrapy asynchronous spider executed (Success: {scrapy_res.get('success')}).")

    # 8. Strategic Executive Decision Engine
    logger.info("\n--- Phase 8: Strategic Decision Support & Executive Action Plan ---")
    case_study = generate_strategic_case_study(
        pareto_df=pareto_df,
        absa_df=absa_matrix,
        benchmarks=benchmarks,
        network_summary=network_summary,
        geo_stats=geo_data["regional_stats"],
    )
    logger.info(f"Executive strategy report published at {case_study['markdown_file']}")

    elapsed = round(time.time() - start_time, 2)
    print("\n" + "=" * 70)
    print(f" PIPELINE EXECUTION COMPLETED IN {elapsed} SECONDS")
    print(f" - Datastore (Processed): {DATA_PROCESSED_DIR}")
    print(f" - Cache:                 {CACHE_DIR}")
    print(f" - Tables & Metrics:      {TABLES_DIR}")
    print(f" - Figures & Maps:        {FIGURES_DIR}")
    print("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(run_pipeline())
