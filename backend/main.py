"""BrandPulse AI - FastAPI Backend."""
from __future__ import annotations
import io, json, logging, sys, traceback
from pathlib import Path
from typing import Any
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import CACHE_DIR, FIGURES_DIR, REPORTS_DIR, TABLES_DIR, DATA_GAPS_FILE, MANIFEST_FILE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("brandpulse_api")

app = FastAPI(title="BrandPulse AI API", version="2.0.0")
app.add_middleware(CORSMiddleware,
    allow_origins=["http://localhost:5173","http://localhost:3000","http://127.0.0.1:5173"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/figures", StaticFiles(directory=str(FIGURES_DIR)), name="figures")

def _csv(path):
    if not Path(path).exists(): return []
    return pd.read_csv(path).fillna("").to_dict(orient="records")

@app.get("/api/health")
def health(): return {"status": "ok", "version": "2.0.0"}

@app.get("/api/hardware")
def hardware():
    try:
        from scripts.verify_gpu import get_gpu_info
        return get_gpu_info()
    except Exception as e: return {"cuda_available": False, "device_name": "CPU", "error": str(e)}

@app.get("/api/overview/kpis")
def overview_kpis(ds: str = "airline"):
    try:
        from modules.data_processor import (
            load_airline_sentiment, load_shopping_apps_reviews,
            load_indian_airlines_reviews, load_cell_phone_reviews,
            load_cyberbullying_dataset,
        )
        # Load the right dataset
        if ds == "airline":
            df = load_airline_sentiment()
            entity_col = "airline"; sent_col = "airline_sentiment"; date_col = "tweet_created"
        elif ds == "shopping":
            raw = load_shopping_apps_reviews()
            df = pd.concat(list(raw.values()), ignore_index=True) if raw else pd.DataFrame()
            entity_col = "entity"; sent_col = "score"; date_col = None
        elif ds == "indian_air":
            df = load_indian_airlines_reviews()
            entity_col = "airline"; sent_col = "sentiment"; date_col = None
        elif ds == "phones":
            raw = load_cell_phone_reviews()
            df = pd.concat(list(raw.values()), ignore_index=True) if raw else pd.DataFrame()
            entity_col = "entity"; sent_col = "rating"; date_col = None
        elif ds == "cyber":
            df = load_cyberbullying_dataset()
            entity_col = None; sent_col = "oh_label"; date_col = None
        else:
            raise HTTPException(400, f"Unknown dataset key: {ds}")

        if df is None or df.empty:
            raise HTTPException(404, f"Dataset '{ds}' not found or empty. Run download_kaggle_data.py first.")

        total = len(df)

        # Sentiment counts — for numeric columns map to pos/neg/neu
        if sent_col in df.columns:
            sc = df[sent_col].dropna()
            if sc.dtype == object:
                pos = int((sc.str.lower() == "positive").sum())
                neg = int((sc.str.lower() == "negative").sum())
                neu = int((sc.str.lower() == "neutral").sum())
            else:
                # Numeric ratings: >=4 positive, <=2 negative, 3 neutral
                pos = int((sc >= 4).sum()); neg = int((sc <= 2).sum()); neu = int(((sc > 2) & (sc < 4)).sum())
        else:
            pos = neg = neu = 0

        nss = round((pos - neg) / max(1, total) * 100, 1)

        # Breakdown by entity
        by_entity = []
        if entity_col and entity_col in df.columns and sent_col in df.columns:
            try:
                by_entity = df.groupby(entity_col)[sent_col].value_counts(normalize=True).mul(100).round(1).reset_index(name="pct").to_dict(orient="records")
            except Exception:
                pass

        # Trend over time
        trend = []
        if date_col and date_col in df.columns and sent_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df["_month"] = df[date_col].dt.to_period("M").astype(str)
            trend = df.groupby(["_month", sent_col]).size().reset_index(name="count").to_dict(orient="records")

        # Top driver (negative reasons or top category)
        top_driver, top_driver_pct = "N/A", 0
        if "negativereason" in df.columns:
            vc = df["negativereason"].dropna().value_counts()
            if not vc.empty:
                top_driver = vc.index[0]; top_driver_pct = round(vc.iloc[0] / max(1, total) * 100, 1)
        elif sent_col in df.columns:
            vc = df[sent_col].dropna().value_counts()
            if not vc.empty:
                top_driver = str(vc.index[0]); top_driver_pct = round(vc.iloc[0] / max(1, total) * 100, 1)

        return {
            "ds": ds, "total_records": total,
            "positive": pos, "negative": neg, "neutral": neu, "nss": nss,
            "top_driver": top_driver, "top_driver_pct": top_driver_pct,
            "entity_col": entity_col, "sent_col": sent_col,
            "by_entity": by_entity, "trend": trend,
            "pareto": _csv(TABLES_DIR / "pareto_complaint_drivers.csv") if ds == "airline" else [],
        }
    except HTTPException: raise
    except Exception as e: logger.error(traceback.format_exc()); raise HTTPException(500, str(e))


@app.get("/api/competitors")
def competitors():
    try:
        from modules.competitor_benchmark import execute_all_competitor_benchmarks
        raw = execute_all_competitor_benchmarks()
        result = {}
        for k,v in raw.items():
            row = {}
            for fk,fv in v.items():
                row[fk] = fv.item() if hasattr(fv,"item") else fv
            result[k]=row
        return result
    except Exception as e: raise HTTPException(500, str(e))

@app.get("/api/competitors/significance-table")
def comp_sig(): return _csv(TABLES_DIR/"competitor_significance_tests.csv")

@app.get("/api/features")
def features(ds: str = "airline"):
    try:
        from modules.data_processor import load_airline_sentiment, load_cyberbullying_dataset, classify_feature_types
        df = load_airline_sentiment() if ds!="cyber" else load_cyberbullying_dataset()
        return classify_feature_types(df).fillna("").to_dict(orient="records")
    except Exception as e: raise HTTPException(500, str(e))

@app.post("/api/features/upload")
async def feat_upload(file: UploadFile = File(...)):
    try:
        from modules.data_processor import classify_feature_types
        df = pd.read_csv(io.BytesIO(await file.read()), low_memory=False)
        ft = classify_feature_types(df)
        return {"shape":list(df.shape),"columns":df.columns.tolist(),"feature_types":ft.fillna("").to_dict(orient="records"),"preview":df.head(10).fillna("").to_dict(orient="records")}
    except Exception as e: raise HTTPException(500, str(e))

@app.get("/api/sentiment/absa-matrix")
def absa_matrix(): return _csv(TABLES_DIR/"absa_matrix.csv")

@app.get("/api/sentiment/metrics")
def sent_metrics(): return _csv(TABLES_DIR/"model_summary_metrics.csv")

@app.get("/api/sentiment/classification-report")
def class_report(): return _csv(TABLES_DIR/"sentiment_classification_report.csv")

@app.get("/api/sentiment/absa-eval")
def absa_eval(): return _csv(TABLES_DIR/"absa_evaluation_report.csv")

class TextInput(BaseModel):
    text: str

@app.post("/api/sentiment/predict")
def predict(body: TextInput):
    try:
        from modules.sentiment_engine import predict_single_text
        return predict_single_text(body.text)
    except Exception as e: raise HTTPException(500, str(e))

@app.post("/api/sentiment/absa")
def absa_pred(body: TextInput):
    try:
        from modules.absa_engine import run_single_text_absa
        r=run_single_text_absa(body.text)
        return {k: v.item() if hasattr(v,"item") else v for k,v in r.items()}
    except Exception as e: raise HTTPException(500, str(e))

@app.get("/api/geospatial/stats")
def geo_stats(): return _csv(TABLES_DIR/"regional_sentiment_stats.csv")

@app.get("/api/geospatial/points")
def geo_points(exact_only: bool=False, limit: int=2000):
    try:
        from modules.data_processor import load_airline_sentiment
        df=load_airline_sentiment()
        if exact_only: df=df[df["has_exact_coords"]==True]
        cols=[c for c in ["lat","lon","airline_sentiment","airline","negativereason","has_exact_coords"] if c in df.columns]
        return df[cols].dropna(subset=["lat","lon"]).head(limit).fillna("").to_dict(orient="records")
    except Exception as e: raise HTTPException(500, str(e))

@app.get("/api/geospatial/folium-map")
def folium_map(approx: bool=False):
    path=FIGURES_DIR/("folium_sentiment_map_approx.html" if approx else "folium_sentiment_map.html")
    if not path.exists(): raise HTTPException(404,"Map not found. Run pipeline.")
    return FileResponse(path, media_type="text/html")

@app.get("/api/network/summary")
def net_summary(): return _csv(TABLES_DIR/"community_echo_metrics.csv")

@app.get("/api/network/influencers")
def net_influencers(limit: int=50): return _csv(TABLES_DIR/"influencer_centrality_leaderboard.csv")[:limit]

@app.get("/api/network/graph-structures")
def graph_struct(): return _csv(TABLES_DIR/"graph_structures_demo.csv")

@app.get("/api/network/pyvis")
def net_pyvis(kind: str="echo"):
    path=FIGURES_DIR/("pyvis_echo_chamber_network.html" if kind=="echo" else "pyvis_snap_network.html")
    if not path.exists(): raise HTTPException(404,"PyVis HTML not found.")
    return FileResponse(path, media_type="text/html")

@app.get("/api/scrapers/bs4-quotes")
def scraper_bs4(): return _csv(TABLES_DIR/"scraped_bs4_quotes.csv")

@app.get("/api/scrapers/sandbox-quotes")
def scraper_sandbox(): return _csv(TABLES_DIR/"scraped_sandbox_quotes.csv")

@app.get("/api/scrapers/hackernews")
def scraper_hn():
    cached=_csv(TABLES_DIR/"api_hackernews_sample.csv")
    if cached: return cached
    try:
        from modules.scrapers_demo import HackerNewsApiFetcher
        return HackerNewsApiFetcher().fetch_data(limit=10)
    except Exception as e: raise HTTPException(500,str(e))

@app.get("/api/strategy/decision-matrix")
def strategy_matrix(): return _csv(TABLES_DIR/"recommendation_decision_matrix.csv")

@app.get("/api/strategy/report-md")
def strategy_report():
    path=REPORTS_DIR/"case_study_executive_report.md"
    if not path.exists(): raise HTTPException(404,"Report not found.")
    return {"markdown": path.read_text(encoding="utf-8")}

@app.get("/api/strategy/data-gaps")
def data_gaps_ep():
    if not DATA_GAPS_FILE.exists(): return {"markdown":"No data gaps logged."}
    return {"markdown": DATA_GAPS_FILE.read_text(encoding="utf-8")}
