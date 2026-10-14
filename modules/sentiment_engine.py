"""BrandPulse AI - GPU Sentiment Engine & ML Classification Benchmark.

Orchestrates deep transformer inference on NVIDIA CUDA GPU with:
- Model: cardiffnlp/twitter-roberta-base-sentiment-latest (3-class: negative, neutral, positive)
- FP16 precision, batched inference with dynamic OOM fallback (32 -> 16 -> 8 -> 4 -> 1)
- Ground-truth evaluation on US Airline dataset
- Comprehensive metrics: Precision, Recall, F1 (per class, macro, weighted), Accuracy
- Confusion Matrix (raw and normalized)
- Multi-class One-vs-Rest ROC curves with AUC (using full softmax probabilities)
- Precision-Recall curves per class
- Cyberbullying multi-class benchmark comparing TF-IDF + Logistic Regression baseline
- Caching predictions in cache/ for instant repeat queries
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import (
    CACHE_DIR,
    FIGURES_DIR,
    INFERENCE_BATCH_SIZES,
    RANDOM_SEED,
    SENTIMENT_MODEL_ID,
    TABLES_DIR,
    USE_FP16,
)

logger = logging.getLogger("sentiment_engine")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Global model and tokenizer singletons
_TOKENIZER = None
_MODEL = None
_DEVICE = None


def get_sentiment_model() -> tuple[Any, Any, torch.device]:
    """Initialize and cache the transformer model and tokenizer on CUDA GPU."""
    global _TOKENIZER, _MODEL, _DEVICE
    if _MODEL is not None and _TOKENIZER is not None:
        return _TOKENIZER, _MODEL, _DEVICE

    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    _DEVICE = device

    logger.info(f"Loading sentiment model '{SENTIMENT_MODEL_ID}' onto {device}...")
    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_ID)

    if cuda_available and USE_FP16:
        model = model.half()  # Convert to fp16 for Tensor Core acceleration

    model.to(device)
    model.eval()

    _TOKENIZER = tokenizer
    _MODEL = model
    logger.info(f"Sentiment model successfully loaded on {device} (FP16: {cuda_available and USE_FP16})")
    return _TOKENIZER, _MODEL, _DEVICE


def predict_sentiment_batch(
    texts: list[str],
    batch_size: int = 32,
) -> dict[str, Any]:
    """Execute batched inference with dynamic OOM fallback.

    Returns:
    - predicted_labels: list of strings ('negative', 'neutral', 'positive')
    - probabilities: np.ndarray of shape (N, 3) representing softmax probabilities
    """
    tokenizer, model, device = get_sentiment_model()
    # Model label mapping: cardiffnlp maps 0: negative, 1: neutral, 2: positive
    id2label = {0: "negative", 1: "neutral", 2: "positive"}

    all_probs = []
    all_preds = []

    # Attempt with specified batch size; on CUDA OOM, backtrack through smaller sizes
    current_batch_size = batch_size
    n_samples = len(texts)

    idx = 0
    while idx < n_samples:
        batch_texts = texts[idx : idx + current_batch_size]
        try:
            with torch.no_grad():
                encoded = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                ).to(device)

                outputs = model(**encoded)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).cpu().float().numpy()

                all_probs.append(probs)
                batch_preds = np.argmax(probs, axis=1)
                all_preds.extend([id2label[p] for p in batch_preds])

            idx += current_batch_size

        except torch.cuda.OutOfMemoryError:
            logger.warning(f"CUDA OutOfMemoryError caught at batch size {current_batch_size}!")
            torch.cuda.empty_cache()
            # Find next smaller batch size
            smaller_sizes = [b for b in INFERENCE_BATCH_SIZES if b < current_batch_size]
            if smaller_sizes:
                current_batch_size = smaller_sizes[0]
                logger.info(f"Falling back to smaller batch size: {current_batch_size}")
            else:
                logger.error("OOM even at batch size 1. Falling back to CPU.")
                model.cpu().float()
                device = torch.device("cpu")
                current_batch_size = 1

    if all_probs:
        combined_probs = np.vstack(all_probs)
    else:
        combined_probs = np.empty((0, 3))

    return {
        "labels": all_preds,
        "probabilities": combined_probs,
    }


def evaluate_airline_sentiment(
    df: pd.DataFrame,
    sample_size: int = 1500,
    force_recompute: bool = False,
) -> dict[str, Any]:
    """Evaluate cardiffnlp RoBERTa against ground truth US Airline sentiment labels.

    Computes: Precision, Recall, F1, Confusion Matrix, ROC-AUC, PR Curves.
    Caches predictions to cache/airline_predictions.parquet.
    """
    cache_path = CACHE_DIR / "airline_predictions.parquet"
    classes = ["negative", "neutral", "positive"]

    valid_df = df.dropna(subset=["clean_text", "airline_sentiment"]).copy()
    valid_df = valid_df[valid_df["airline_sentiment"].isin(classes)]

    if sample_size and len(valid_df) > sample_size:
        # Stratified sampling for evaluation speed while preserving class distribution
        eval_df = valid_df.groupby("airline_sentiment", group_keys=False).apply(
            lambda x: x.sample(
                int(np.rint(sample_size * len(x) / len(valid_df))),
                random_state=RANDOM_SEED,
            )
        )
    else:
        eval_df = valid_df

    if not force_recompute and cache_path.exists():
        logger.info(f"Loading cached sentiment predictions from {cache_path}")
        cached_res = pd.read_parquet(cache_path)
        eval_df = cached_res
        y_true = eval_df["airline_sentiment"].tolist()
        y_pred = eval_df["pred_label"].tolist()
        y_probs = eval_df[["prob_negative", "prob_neutral", "prob_positive"]].to_numpy()
    else:
        logger.info(f"Running RoBERTa inference on {len(eval_df)} evaluation tweets...")
        texts = eval_df["clean_text"].tolist()
        preds_dict = predict_sentiment_batch(texts, batch_size=32)

        eval_df = eval_df.copy()
        eval_df["pred_label"] = preds_dict["labels"]
        eval_df["prob_negative"] = preds_dict["probabilities"][:, 0]
        eval_df["prob_neutral"] = preds_dict["probabilities"][:, 1]
        eval_df["prob_positive"] = preds_dict["probabilities"][:, 2]

        eval_df.to_parquet(cache_path, index=False)
        logger.info(f"Saved evaluation predictions to {cache_path}")

        y_true = eval_df["airline_sentiment"].tolist()
        y_pred = eval_df["pred_label"].tolist()
        y_probs = preds_dict["probabilities"]

    # 1. Classification Metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    clf_rep_dict = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    clf_rep_df = pd.DataFrame(clf_rep_dict).transpose()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    clf_rep_df.to_csv(TABLES_DIR / "sentiment_classification_report.csv")

    # 2. Confusion Matrices (Raw and Normalized)
    cm_raw = confusion_matrix(y_true, y_pred, labels=classes)
    cm_norm = confusion_matrix(y_true, y_pred, labels=classes, normalize="true")

    # Plot Confusion Matrix figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm_raw, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax1)
    ax1.set_title("Confusion Matrix (Raw Counts)", fontsize=11)
    ax1.set_xlabel("Predicted Label")
    ax1.set_ylabel("True Label")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens", xticklabels=classes, yticklabels=classes, ax=ax2)
    ax2.set_title("Confusion Matrix (Normalized Recall)", fontsize=11)
    ax2.set_xlabel("Predicted Label")
    ax2.set_ylabel("True Label")
    plt.tight_layout()
    cm_fig_path = FIGURES_DIR / "confusion_matrix.png"
    fig.savefig(cm_fig_path, dpi=200)
    plt.close(fig)

    # 3. Multi-Class ROC Curves and PR Curves (One-vs-Rest)
    y_true_bin = label_binarize(y_true, classes=classes)
    roc_auc_per_class = {}
    roc_curves_data = {}
    pr_curves_data = {}

    fig_roc, ax_roc = plt.subplots(figsize=(7, 5))
    fig_pr, ax_pr = plt.subplots(figsize=(7, 5))

    colors = ["#ef4444", "#6b7280", "#10b981"]

    for i, cls_name in enumerate(classes):
        # ROC
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        cls_auc = auc(fpr, tpr)
        roc_auc_per_class[cls_name] = round(float(cls_auc), 3)
        roc_curves_data[cls_name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": cls_auc}

        ax_roc.plot(fpr, tpr, color=colors[i], lw=2, label=f"{cls_name} (AUC = {cls_auc:.2f})")

        # PR Curve
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
        pr_curves_data[cls_name] = {"precision": precision.tolist(), "recall": recall.tolist()}
        ax_pr.plot(recall, precision, color=colors[i], lw=2, label=f"{cls_name}")

    ax_roc.plot([0, 1], [0, 1], "k--", lw=1)
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("One-vs-Rest ROC Curves", fontsize=12)
    ax_roc.legend(loc="lower right")
    fig_roc.tight_layout()
    fig_roc.savefig(FIGURES_DIR / "roc_curves.png", dpi=200)
    plt.close(fig_roc)

    ax_pr.set_xlim([0.0, 1.0])
    ax_pr.set_ylim([0.0, 1.05])
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curves per Sentiment Class", fontsize=12)
    ax_pr.legend(loc="lower left")
    fig_pr.tight_layout()
    fig_pr.savefig(FIGURES_DIR / "precision_recall_curves.png", dpi=200)
    plt.close(fig_pr)

    macro_roc_auc = float(roc_auc_score(y_true_bin, y_probs, multi_class="ovr", average="macro"))

    summary = {
        "accuracy": round(float(accuracy), 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "macro_roc_auc": round(macro_roc_auc, 4),
        "class_auc": roc_auc_per_class,
        "classes": classes,
        "confusion_matrix_raw": cm_raw.tolist(),
        "confusion_matrix_norm": cm_norm.tolist(),
        "report_df": clf_rep_df,
    }

    # Save overview table
    pd.DataFrame([{
        "Metric": "Accuracy",
        "Value": summary["accuracy"],
    }, {
        "Metric": "Macro Precision",
        "Value": summary["macro_precision"],
    }, {
        "Metric": "Macro Recall",
        "Value": summary["macro_recall"],
    }, {
        "Metric": "Macro F1",
        "Value": summary["macro_f1"],
    }, {
        "Metric": "Weighted F1",
        "Value": summary["weighted_f1"],
    }, {
        "Metric": "Macro ROC-AUC (OvR)",
        "Value": summary["macro_roc_auc"],
    }]).to_csv(TABLES_DIR / "model_summary_metrics.csv", index=False)

    return summary


def benchmark_cyberbullying_baseline(
    df: pd.DataFrame,
    sample_size: int = 5000,
) -> dict[str, Any]:
    """Benchmark multi-class cyberbullying classification.

    Compares TF-IDF + Logistic Regression baseline against stratified train/test split.
    """
    valid_df = df.dropna(subset=["clean_text", "oh_label"]).copy()
    if sample_size and len(valid_df) > sample_size:
        valid_df = valid_df.sample(sample_size, random_state=RANDOM_SEED)

    X = valid_df["clean_text"]
    y = valid_df["oh_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y
    )

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)
    y_prob = clf.predict_proba(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    logger.info(f"Cyberbullying Baseline (TF-IDF + LogReg): Accuracy={acc:.4f}, Weighted F1={f1:.4f}")
    return {
        "model_name": "TF-IDF + Logistic Regression Baseline",
        "test_size": len(y_test),
        "accuracy": round(float(acc), 4),
        "weighted_f1": round(float(f1), 4),
        "macro_f1": round(float(macro_f1), 4),
    }


def predict_single_text(text: str) -> dict[str, Any]:
    """Run real-time GPU inference for a single user-supplied text string."""
    cleaned = text.strip()
    if not cleaned:
        return {"sentiment": "neutral", "confidence": 0.33, "probabilities": {}}

    res = predict_sentiment_batch([cleaned], batch_size=1)
    label = res["labels"][0]
    probs = res["probabilities"][0]
    return {
        "sentiment": label,
        "confidence": round(float(np.max(probs)), 3),
        "probabilities": {
            "negative": round(float(probs[0]), 3),
            "neutral": round(float(probs[1]), 3),
            "positive": round(float(probs[2]), 3),
        },
    }
