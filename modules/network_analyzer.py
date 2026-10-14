"""BrandPulse AI - Network Analysis & Echo-Chamber Engine.

Performs graph theory and social network dynamics using NetworkX and PyVis:
- Network Data Structures & Representation:
  (Adjacency list, edge list, sparse adjacency matrix, node/edge attributes, directed vs undirected)
- Graph construction from real datasets:
  1. Stanford SNAP Twitter combined ego-network (sampled k-core / top-degree)
  2. Twitter brand-mention & user interaction graph extracted from real tweets
- Centrality & Echo-Chamber analytics:
  - In/Out Degree Centrality, Approximated Betweenness Centrality, PageRank (alpha=0.85)
  - Louvain Community Detection with Modularity scoring
  - Echo-chamber metrics: Sentiment assortativity, within vs cross-community edge ratio,
    homophily index, and community polarization
  - Interactive PyVis HTML generation with node sizes based on PageRank and community colors
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any
import community as community_louvain
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from pyvis.network import Network

from config import CACHE_DIR, DATA_RAW_DIR, FIGURES_DIR, RANDOM_SEED, TABLES_DIR

logger = logging.getLogger("network_analyzer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# =====================================================================
# 1. NETWORK DATA STRUCTURES & REPRESENTATIONS
# =====================================================================
def demonstrate_graph_data_structures() -> dict[str, Any]:
    """Demonstrate graph data structure representations on a curated micro-graph.

    Shows: Adjacency List, Edge List, Adjacency Matrix, Node/Edge Attributes,
    and Directed vs Undirected properties.
    """
    g = nx.DiGraph()
    # Add nodes with brand attributes
    nodes_data = [
        ("UserA", {"role": "Influencer", "sentiment": 0.8, "followers": 12000}),
        ("UserB", {"role": "Customer", "sentiment": -0.6, "followers": 350}),
        ("UserC", {"role": "Customer", "sentiment": -0.9, "followers": 120}),
        ("AirlineHelp", {"role": "Official Brand", "sentiment": 0.2, "followers": 950000}),
        ("TechReviewer", {"role": "Journalist", "sentiment": 0.1, "followers": 45000}),
    ]
    for n, attrs in nodes_data:
        g.add_node(n, **attrs)

    # Add weighted directed edges
    edges_data = [
        ("UserB", "AirlineHelp", {"weight": 3, "interaction": "complaint"}),
        ("UserC", "AirlineHelp", {"weight": 1, "interaction": "complaint"}),
        ("AirlineHelp", "UserB", {"weight": 2, "interaction": "resolution"}),
        ("UserA", "TechReviewer", {"weight": 5, "interaction": "retweet"}),
        ("TechReviewer", "AirlineHelp", {"weight": 1, "interaction": "inquiry"}),
        ("UserB", "UserC", {"weight": 4, "interaction": "retweet"}),
    ]
    for u, v, attrs in edges_data:
        g.add_edge(u, v, **attrs)

    # 1. Edge List representation
    edge_list_repr = [(u, v, d["weight"]) for u, v, d in g.edges(data=True)]

    # 2. Adjacency List representation
    adj_list_repr = {n: list(g.adj[n].keys()) for n in g.nodes()}

    # 3. Adjacency Matrix representation
    node_order = list(g.nodes())
    adj_matrix = nx.to_numpy_array(g, nodelist=node_order, weight="weight")

    demo_summary = {
        "num_nodes": g.number_of_nodes(),
        "num_edges": g.number_of_edges(),
        "is_directed": g.is_directed(),
        "edge_list": edge_list_repr,
        "adjacency_list": adj_list_repr,
        "node_order": node_order,
        "adjacency_matrix": adj_matrix.tolist(),
        "node_attributes": {n: g.nodes[n] for n in g.nodes()},
    }

    # Save to table
    edge_df = pd.DataFrame([
        {"Source": u, "Target": v, "Weight": d["weight"], "Type": d["interaction"]}
        for u, v, d in g.edges(data=True)
    ])
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    edge_df.to_csv(TABLES_DIR / "graph_structures_demo.csv", index=False)

    return demo_summary


# =====================================================================
# 2. REAL GRAPH BUILDERS (SNAP & AIRLINE MENTIONS)
# =====================================================================
def build_airline_interaction_graph(df: pd.DataFrame, max_edges: int = 5000) -> nx.DiGraph:
    """Extract a real interaction/mention network from US Airline tweets.

    Nodes: Twitter User handles & Airline handles
    Edges: Directed mention / reply relations with sentiment and weight attributes
    """
    g = nx.DiGraph()
    sentiment_map = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}

    # Iterate through tweets
    count = 0
    for _, row in df.iterrows():
        user = str(row.get("name", "")).strip().lower()
        airline = str(row.get("airline", "")).strip().lower()
        text = str(row.get("clean_text", ""))
        sent = sentiment_map.get(str(row.get("airline_sentiment", "")), 0.0)

        if not user or not airline:
            continue

        # Add nodes with initial attributes
        if not g.has_node(user):
            g.add_node(user, type="user", tweet_count=1, total_sentiment=sent)
        else:
            g.nodes[user]["tweet_count"] += 1
            g.nodes[user]["total_sentiment"] += sent

        if not g.has_node(airline):
            g.add_node(airline, type="brand", tweet_count=1, total_sentiment=sent)
        else:
            g.nodes[airline]["tweet_count"] += 1
            g.nodes[airline]["total_sentiment"] += sent

        # Add edge from user to airline
        if g.has_edge(user, airline):
            g[user][airline]["weight"] += 1
            g[user][airline]["sentiment_sum"] += sent
        else:
            g.add_edge(user, airline, weight=1, sentiment_sum=sent)

        # Check for secondary @mentions in text
        mentions = [w.lstrip("@") for w in text.split() if w.startswith("@") and len(w) > 2]
        for m in mentions:
            m_clean = m.lower()
            if m_clean != airline and m_clean != user:
                if not g.has_node(m_clean):
                    g.add_node(m_clean, type="mentioned", tweet_count=1, total_sentiment=sent)
                if g.has_edge(user, m_clean):
                    g[user][m_clean]["weight"] += 1
                    g[user][m_clean]["sentiment_sum"] += sent
                else:
                    g.add_edge(user, m_clean, weight=1, sentiment_sum=sent)

        count += 1
        if count >= max_edges:
            break

    # Calculate average node sentiment
    for n in g.nodes():
        t_count = g.nodes[n].get("tweet_count", 1)
        tot_sent = g.nodes[n].get("total_sentiment", 0.0)
        g.nodes[n]["avg_sentiment"] = round(tot_sent / max(1, t_count), 2)

    logger.info(f"Built Airline Interaction Graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    return g


def load_snap_twitter_subgraph(max_nodes: int = 1500, k_core: int = 15) -> nx.Graph:
    """Load and sample Stanford SNAP twitter_combined graph.

    Applies k-core decomposition and degree filtering to extract an influential,
    densely connected social subgraph suitable for browser PyVis rendering.
    """
    snap_path = DATA_RAW_DIR / "snap" / "twitter_combined.txt"
    if not snap_path.exists():
        logger.warning(f"SNAP dataset not found at {snap_path}. Building synthetic benchmark graph.")
        return nx.erdos_renyi_graph(100, 0.05, seed=RANDOM_SEED)

    logger.info(f"Loading SNAP edges from {snap_path} (reading top lines)...")
    g = nx.Graph()

    # Read top 200,000 edges for fast memory-safe sub-graph loading
    with open(snap_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= 150000:
                break
            parts = line.strip().split()
            if len(parts) >= 2:
                g.add_edge(parts[0], parts[1])

    logger.info(f"Raw SNAP loaded component: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")

    # Apply k-core pruning to obtain dense core
    try:
        core_g = nx.k_core(g, k=k_core)
        if core_g.number_of_nodes() > max_nodes:
            # Select top-N nodes by degree
            top_nodes = sorted(core_g.degree, key=lambda x: x[1], reverse=True)[:max_nodes]
            sub_nodes = [n[0] for n in top_nodes]
            final_g = core_g.subgraph(sub_nodes).copy()
        elif core_g.number_of_nodes() > 50:
            final_g = core_g
        else:
            final_g = g.subgraph(list(g.nodes())[:max_nodes]).copy()
    except Exception as e:
        logger.warning(f"K-core decomposition warning: {e}. Falling back to degree slice.")
        top_nodes = sorted(g.degree, key=lambda x: x[1], reverse=True)[:max_nodes]
        final_g = g.subgraph([n[0] for n in top_nodes]).copy()

    logger.info(f"Sampled SNAP Subgraph: {final_g.number_of_nodes()} nodes, {final_g.number_of_edges()} edges")
    return final_g


# =====================================================================
# 3. CENTRALITY, LOUVAIN COMMUNITIES & ECHO-CHAMBER METRICS
# =====================================================================
def analyze_graph_dynamics(
    g: nx.Graph | nx.DiGraph,
    k_samples_betweenness: int = 100,
) -> dict[str, Any]:
    """Execute complete network analytics suite:

    - Degree Centrality (In/Out if directed)
    - Approximated Betweenness Centrality
    - PageRank (alpha=0.85)
    - Louvain Community Detection with Modularity
    - Echo-Chamber Metrics: Sentiment Assortativity, Homophily Index,
      Within-vs-Cross Community Edge Ratio
    """
    logger.info("Computing network centralities and community structures...")
    is_directed = g.is_directed()

    # Convert to undirected for Louvain modularity analysis
    undirected_g = g.to_undirected() if is_directed else g

    # 1. Centralities
    degree_cent = nx.degree_centrality(g)
    in_degree_cent = nx.in_degree_centrality(g) if is_directed else degree_cent
    out_degree_cent = nx.out_degree_centrality(g) if is_directed else degree_cent

    # Approximated Betweenness Centrality using k random pivot samples for speed
    k_val = min(k_samples_betweenness, g.number_of_nodes())
    betweenness_cent = nx.betweenness_centrality(g, k=k_val, seed=RANDOM_SEED)

    # PageRank
    pagerank_cent = nx.pagerank(g, alpha=0.85, max_iter=100)

    # 2. Louvain Community Detection
    partition = community_louvain.best_partition(undirected_g, random_state=RANDOM_SEED)
    modularity_score = community_louvain.modularity(partition, undirected_g)

    # Attach community and pagerank to node attributes
    for n in g.nodes():
        g.nodes[n]["community"] = partition.get(n, 0)
        g.nodes[n]["pagerank"] = pagerank_cent.get(n, 0.0)
        g.nodes[n]["betweenness"] = betweenness_cent.get(n, 0.0)
        g.nodes[n]["degree"] = degree_cent.get(n, 0.0)

    # Community sizes
    comm_series = pd.Series(partition.values())
    comm_sizes = comm_series.value_counts().to_dict()

    # 3. Echo-Chamber & Homophily Metrics
    within_edges = 0
    cross_edges = 0
    for u, v in undirected_g.edges():
        if partition.get(u) == partition.get(v):
            within_edges += 1
        else:
            cross_edges += 1

    total_edges = max(1, within_edges + cross_edges)
    within_ratio = within_edges / total_edges
    cross_ratio = cross_edges / total_edges

    # Homophily Index: Ratio of observed within-group ties vs expected random ties
    # (High within_ratio > 0.7 indicates severe echo-chamber partitioning)
    homophily_index = within_ratio / max(0.001, (1.0 - (1.0 / max(1, len(comm_sizes)))))

    # Sentiment Assortativity (if sentiment attributes exist)
    has_sentiment = any("avg_sentiment" in g.nodes[n] for n in g.nodes())
    sentiment_assortativity = 0.0
    if has_sentiment:
        try:
            sentiment_assortativity = float(
                nx.numeric_assortativity_coefficient(undirected_g, "avg_sentiment")
            )
        except Exception:
            sentiment_assortativity = 0.0

    # Compile Top Influencers Leaderboard
    influencer_records = []
    for n in g.nodes():
        influencer_records.append({
            "node_id": str(n),
            "pagerank": round(pagerank_cent.get(n, 0.0), 5),
            "betweenness": round(betweenness_cent.get(n, 0.0), 5),
            "degree": round(degree_cent.get(n, 0.0), 5),
            "community": partition.get(n, 0),
            "avg_sentiment": g.nodes[n].get("avg_sentiment", 0.0),
        })

    influencer_df = pd.DataFrame(influencer_records).sort_values("pagerank", ascending=False)
    influencer_df.to_csv(TABLES_DIR / "influencer_centrality_leaderboard.csv", index=False)

    # Community summary table
    comm_summary_df = pd.DataFrame([
        {"Community_ID": k, "Node_Count": v, "Percentage": round((v / g.number_of_nodes()) * 100, 2)}
        for k, v in sorted(comm_sizes.items(), key=lambda x: x[1], reverse=True)[:10]
    ])
    comm_summary_df.to_csv(TABLES_DIR / "community_echo_metrics.csv", index=False)

    return {
        "num_nodes": g.number_of_nodes(),
        "num_edges": g.number_of_edges(),
        "num_communities": len(comm_sizes),
        "modularity_score": round(float(modularity_score), 4),
        "within_edge_ratio": round(float(within_ratio), 4),
        "cross_edge_ratio": round(float(cross_ratio), 4),
        "homophily_index": round(float(homophily_index), 4),
        "sentiment_assortativity": round(float(sentiment_assortativity), 4),
        "top_influencers": influencer_df.head(15),
        "community_sizes": comm_sizes,
        "partition": partition,
        "graph": g,
    }


def generate_pyvis_network_html(
    g: nx.Graph | nx.DiGraph,
    output_filename: str = "pyvis_echo_chamber_network.html",
    max_display_nodes: int = 500,
) -> Path:
    """Generate interactive PyVis HTML network visualization.

    Node size is scaled by PageRank, color mapped to Louvain community,
    and hover tooltips display centrality and sentiment metrics.
    """
    out_path = FIGURES_DIR / output_filename

    # Subsample if graph exceeds display cap for browser responsiveness
    if g.number_of_nodes() > max_display_nodes:
        # Select top nodes by PageRank or Degree
        nodes_sorted = sorted(
            g.nodes(data=True),
            key=lambda x: x[1].get("pagerank", x[1].get("degree", 0)),
            reverse=True,
        )[:max_display_nodes]
        display_nodes = [n[0] for n in nodes_sorted]
        sub_g = g.subgraph(display_nodes)
    else:
        sub_g = g

    net = Network(height="650px", width="100%", bgcolor="#0f172a", font_color="#f8fafc", directed=g.is_directed())

    # Community color palette
    palette = [
        "#38bdf8",
        "#f43f5e",
        "#10b981",
        "#a855f7",
        "#f59e0b",
        "#6366f1",
        "#ec4899",
        "#14b8a6",
        "#eab308",
        "#84cc16",
    ]

    for n, data in sub_g.nodes(data=True):
        comm_id = data.get("community", 0)
        color = palette[comm_id % len(palette)]
        pr = data.get("pagerank", 0.001)
        size = max(10, min(45, int(math.sqrt(pr * 10000) * 12)))

        tooltip = (
            f"<b>Node:</b> {n}<br>"
            f"<b>Community:</b> {comm_id}<br>"
            f"<b>PageRank:</b> {pr:.4f}<br>"
            f"<b>Betweenness:</b> {data.get('betweenness', 0):.4f}<br>"
            f"<b>Sentiment:</b> {data.get('avg_sentiment', 'N/A')}"
        )

        net.add_node(str(n), label=str(n)[:12], title=tooltip, color=color, size=size)

    for u, v, data in sub_g.edges(data=True):
        weight = data.get("weight", 1)
        net.add_edge(str(u), str(v), value=weight, color="#475569")

    # Physics configuration for smooth cluster separation
    net.set_options("""
    var options = {
      "nodes": { "borderWidth": 1.5 },
      "physics": {
        "forceAtlas2Based": { "gravitationalConstant": -50, "centralGravity": 0.01, "springLength": 100, "springConstant": 0.08 },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": { "iterations": 150 }
      }
    }
    """)

    net.save_graph(str(out_path))
    logger.info(f"Saved PyVis interactive network HTML to {out_path}")
    return out_path
