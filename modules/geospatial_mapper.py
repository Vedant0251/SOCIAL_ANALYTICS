"""BrandPulse AI - Geospatial Sentiment & Regional Analytics.

Renders cartographic projections of social media sentiment:
- Rigorous separation between:
  1. Exact GPS Coordinates (~7% of US Airline records)
  2. Approximate Geographic Centroids derived from user_timezone / tweet_location
- Interactive Folium Cartography:
  - Fast MarkerCluster layers
  - HeatMap layer for complaint density
  - FeatureGroup overlays per negativereason with dynamic LayerControl
- Plotly Express Mapbox scatter and density visualizers
- Regional spatial aggregation:
  - Top complaint drivers by US geographic region
  - Net Sentiment distribution across geographic zones
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import folium
from folium.plugins import HeatMap, MarkerCluster
import numpy as np
import pandas as pd
import plotly.express as px

from config import CACHE_DIR, FIGURES_DIR, TABLES_DIR

logger = logging.getLogger("geospatial_mapper")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# US Timezone to Approximate Geographic Centroid Reference (Latitude, Longitude, Region)
US_TIMEZONE_CENTROIDS: dict[str, tuple[float, float, str]] = {
    "Eastern Time (US & Canada)": (40.7128, -74.0060, "Northeast / East"),
    "Central Time (US & Canada)": (41.8781, -87.6298, "Midwest / Central"),
    "Pacific Time (US & Canada)": (34.0522, -118.2437, "West Coast"),
    "Mountain Time (US & Canada)": (39.7392, -104.9903, "Mountain West"),
    "Atlantic Time (Canada)": (44.6488, -63.5752, "Atlantic"),
    "Hawaii": (21.3069, -157.8583, "Hawaii"),
    "Alaska": (61.2181, -149.9003, "Alaska"),
    "Quito": (-0.1807, -78.4678, "International"),
    "London": (51.5074, -0.1278, "International"),
    "Amsterdam": (52.3676, 4.9041, "International"),
}


def prepare_geospatial_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Prepare separate exact-coordinate and approximate-location subsets.

    Calculates exact coverage percentage and spatial statistics.
    """
    total_records = len(df)

    # 1. Exact Coordinates Subset
    exact_df = df[df["has_exact_coords"] == True].copy()
    exact_count = len(exact_df)
    exact_percentage = round((exact_count / max(1, total_records)) * 100, 2)

    # 2. Approximate Centroids Subset
    approx_records = []
    for _, row in df[df["has_exact_coords"] == False].iterrows():
        tz = str(row.get("user_timezone", "")).strip()
        if tz in US_TIMEZONE_CENTROIDS:
            lat, lon, region = US_TIMEZONE_CENTROIDS[tz]
            # Add minor deterministic jitter to prevent overlap on exact centroid
            jitter_lat = lat + np.sin(hash(row["text"]) % 100) * 0.15
            jitter_lon = lon + np.cos(hash(row["text"]) % 100) * 0.15
            r_dict = row.to_dict()
            r_dict["latitude"] = jitter_lat
            r_dict["longitude"] = jitter_lon
            r_dict["geo_source"] = "Approximate (Timezone Centroid)"
            r_dict["region"] = region
            approx_records.append(r_dict)

    approx_df = pd.DataFrame(approx_records) if approx_records else pd.DataFrame()
    exact_df["geo_source"] = "Exact GPS Coordinates"
    exact_df["region"] = "Exact Coordinates Point"

    logger.info(
        f"Geospatial Coverage: {exact_count}/{total_records} exact coordinates ({exact_percentage}%), "
        f"{len(approx_df)} approximate locations."
    )

    # 3. Regional Complaint Analytics
    combined_geo = pd.concat([exact_df, approx_df], ignore_index=True)
    if "region" in combined_geo.columns and not combined_geo.empty:
        regional_stats = (
            combined_geo.groupby("region")
            .agg(
                total_volume=("text", "count"),
                avg_sentiment=("sentiment_score", "mean"),
                negative_count=("airline_sentiment", lambda s: (s == "negative").sum()),
                top_complaint=(
                    "negativereason",
                    lambda s: s.value_counts().index[0] if len(s.dropna()) > 0 else "N/A",
                ),
            )
            .reset_index()
        )
        regional_stats["complaint_rate"] = round(
            (regional_stats["negative_count"] / regional_stats["total_volume"]) * 100, 1
        )
        regional_stats.to_csv(TABLES_DIR / "regional_sentiment_stats.csv", index=False)
    else:
        regional_stats = pd.DataFrame()

    return {
        "exact_df": exact_df,
        "approx_df": approx_df,
        "combined_df": combined_geo,
        "exact_count": exact_count,
        "total_records": total_records,
        "exact_percentage": exact_percentage,
        "regional_stats": regional_stats,
    }


def generate_folium_sentiment_map(
    geo_data: dict[str, Any],
    include_approx: bool = False,
    output_filename: str = "folium_sentiment_map.html",
) -> folium.Map:
    """Generate Folium interactive map with MarkerClusters, HeatMap, and FeatureGroups.

    Saves HTML map to reports/figures/folium_sentiment_map.html.
    """
    exact_df = geo_data["exact_df"]
    target_df = (
        pd.concat([exact_df, geo_data["approx_df"]], ignore_index=True)
        if include_approx
        else exact_df
    )

    # Center of United States
    m = folium.Map(location=[39.8283, -98.5795], zoom_start=4, tiles="OpenStreetMap")

    if target_df.empty:
        return m

    # 1. HeatMap Layer for Complaint Density
    neg_coords = target_df[target_df["airline_sentiment"] == "negative"][
        ["latitude", "longitude"]
    ].dropna().values.tolist()
    if neg_coords:
        heat_layer = HeatMap(
            neg_coords,
            name="Negative Sentiment Heatmap",
            min_opacity=0.3,
            radius=15,
            blur=12,
            max_zoom=10,
        )
        heat_layer.add_to(m)

    # 2. Clustered Markers by Sentiment
    cluster = MarkerCluster(name="Clustered Sentiment Tweets").add_to(m)

    sentiment_colors = {
        "negative": "red",
        "neutral": "gray",
        "positive": "green",
    }

    # Add sample to avoid browser memory saturation (max 1000 markers in cluster)
    sample_markers = target_df.sample(min(1000, len(target_df)), random_state=42)

    for _, row in sample_markers.iterrows():
        lat, lon = row["latitude"], row["longitude"]
        sent = str(row["airline_sentiment"]).lower()
        color = sentiment_colors.get(sent, "blue")
        airline = row.get("airline", "Airline")
        reason = row.get("negativereason", "No specific reason")
        text = str(row.get("text", "")).replace("'", "&#39;")[:120]

        popup_html = f"""
        <div style="font-family:sans-serif; width:220px;">
            <b style="color:{color};">{airline.upper()} ({sent.upper()})</b><br>
            <small><b>Reason:</b> {reason}</small><br>
            <p style="font-size:11px; margin-top:4px;">"{text}"</p>
            <small style="color:#64748b;">Source: {row.get('geo_source', 'GPS')}</small>
        </div>
        """

        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=250),
        ).add_to(cluster)

    # 3. Add FeatureGroups for Top Negative Reasons
    top_reasons = (
        target_df["negativereason"].dropna().value_counts().head(3).index.tolist()
    )
    for reason in top_reasons:
        fg = folium.FeatureGroup(name=f"Complaint: {reason}")
        reason_subset = target_df[target_df["negativereason"] == reason].head(150)
        for _, r in reason_subset.iterrows():
            folium.CircleMarker(
                location=[r["latitude"], r["longitude"]],
                radius=4,
                color="#b91c1c",
                fill=True,
                fill_color="#dc2626",
                fill_opacity=0.8,
                tooltip=f"{r.get('airline', '')}: {reason}",
            ).add_to(fg)
        fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    out_path = FIGURES_DIR / output_filename
    m.save(str(out_path))
    logger.info(f"Saved Folium interactive map to {out_path}")
    return m


def generate_plotly_geo_scatter(df: pd.DataFrame) -> Any:
    """Generate interactive Plotly scatter_mapbox of exact coordinates."""
    valid = df.dropna(subset=["latitude", "longitude"]).copy()
    if valid.empty:
        return None

    color_map = {
        "negative": "#ef4444",
        "neutral": "#64748b",
        "positive": "#10b981",
    }

    fig = px.scatter_map(
        valid,
        lat="latitude",
        lon="longitude",
        color="airline_sentiment",
        color_discrete_map=color_map,
        hover_name="airline",
        hover_data={"negativereason": True, "text": False},
        zoom=3,
        center={"lat": 38.0, "lon": -97.0},
        map_style="carto-positron",
        title="Exact GPS Coordinates by Sentiment Polarity",
        height=550,
    )
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    return fig
