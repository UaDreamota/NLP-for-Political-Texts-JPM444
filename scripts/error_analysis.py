"""Error analysis helpers for the strongest baseline models.

This fits the top baseline models (from outputs/baselines_metrics_*.csv)
on the same 80/20 split used during baseline scoring, then exports plots
and tables that highlight where the models go wrong.
"""

from pathlib import Path
from typing import Dict, Tuple
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data_processing import load_processing

plt.style.use("seaborn-v0_8-whitegrid")

# Hand-picked best baselines from outputs/baselines_metrics_{target}.csv
BEST_BASELINES: Dict[str, Pipeline] = {
    "political": Pipeline(
        [
            (
                "vec",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=0.002,
                    max_df=0.9,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    solver="saga",
                    max_iter=10000,
                    C=1.0,
                    class_weight="balanced",
                ),
            ),
        ]
    ),
    "domestic": Pipeline(
        [
            (
                "vec",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=0.001,
                    max_df=0.9,
                    max_features=100000,
                    stop_words=None,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LinearSVC(
                    C=1.0,
                    class_weight=None,
                    max_iter=10000,
                ),
            ),
        ]
    ),
}


def _prepare_split(df: pd.DataFrame, target: str, test_size: float = 0.2, seed: int = 42):
    """Mirror the baseline split: 80/20 without stratification, description-only features."""
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not in dataframe.")

    df = df.copy()
    df["description"] = df["description"].fillna("")
    idx = df.index.to_numpy()
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=seed)

    X_train = df.loc[train_idx, "description"]
    X_test = df.loc[test_idx, "description"]
    y_train = df.loc[train_idx, target].astype(int)
    y_test = df.loc[test_idx, target].astype(int)

    meta = df.loc[test_idx, ["id", "headline", "description"]].copy()
    meta["y_true"] = y_test.values
    return X_train, X_test, y_train, y_test, meta


def _safe_text(text: str, max_len: int = 180) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def _scores_from_model(model: Pipeline, X_test):
    """Return a score usable for ranking errors (probability or decision value)."""
    clf = model.named_steps.get("clf")
    if hasattr(clf, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    if hasattr(clf, "decision_function"):
        return model.decision_function(X_test)
    return None


def analyze_target(
    target: str,
    data_path: Path = Path("data") / "belgium_newspaper_new_filter.csv",
    outputs_dir: Path = Path("outputs"),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the best baseline for a target and write plots/tables under outputs/error_analysis."""
    if target not in BEST_BASELINES:
        raise ValueError(f"Unsupported target '{target}'. Expected one of {list(BEST_BASELINES)}")

    outputs_dir = Path(outputs_dir)
    ea_dir = outputs_dir / "error_analysis"
    ea_dir.mkdir(parents=True, exist_ok=True)

    df = load_processing(data_path)
    X_train, X_test, y_train, y_test, meta = _prepare_split(df, target)

    model = BEST_BASELINES[target]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    scores = _scores_from_model(model, X_test)

    # Classification report
    report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T
    report_path = ea_dir / f"{target}_classification_report.csv"
    report.to_csv(report_path, index=True)

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    disp = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"{target} confusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    fig.colorbar(disp, ax=ax, shrink=0.75)
    fig.tight_layout()
    fig.savefig(ea_dir / f"{target}_confusion.png", dpi=150)
    plt.close(fig)

    # ROC and PR curves when we have scores
    if scores is not None:
        fig_roc, ax_roc = plt.subplots(figsize=(4.5, 4))
        RocCurveDisplay.from_predictions(y_test, scores, ax=ax_roc, name="baseline")
        ax_roc.set_title(f"{target} ROC")
        fig_roc.tight_layout()
        fig_roc.savefig(ea_dir / f"{target}_roc.png", dpi=150)
        plt.close(fig_roc)

        fig_pr, ax_pr = plt.subplots(figsize=(4.5, 4))
        PrecisionRecallDisplay.from_predictions(y_test, scores, ax=ax_pr, name="baseline")
        ax_pr.set_title(f"{target} precision-recall")
        fig_pr.tight_layout()
        fig_pr.savefig(ea_dir / f"{target}_pr.png", dpi=150)
        plt.close(fig_pr)

    # Error tables
    meta = meta.copy()
    meta["y_pred"] = y_pred
    meta["score"] = scores if scores is not None else np.nan
    meta["is_fp"] = (meta["y_true"] == 0) & (meta["y_pred"] == 1)
    meta["is_fn"] = (meta["y_true"] == 1) & (meta["y_pred"] == 0)
    meta["headline_short"] = meta["headline"].apply(_safe_text)
    meta["description_short"] = meta["description"].apply(_safe_text)

    top_fp = meta[meta["is_fp"]].sort_values("score", ascending=False).head(20)
    top_fn = meta[meta["is_fn"]].sort_values("score", ascending=True).head(20)

    fp_path = ea_dir / f"{target}_top_false_positives.csv"
    fn_path = ea_dir / f"{target}_top_false_negatives.csv"
    keep_cols = ["id", "y_true", "y_pred", "score", "headline_short", "description_short"]
    top_fp[keep_cols].to_csv(fp_path, index=False)
    top_fn[keep_cols].to_csv(fn_path, index=False)

    return report, meta


if __name__ == "__main__":
    for tgt in ["political", "domestic"]:
        analyze_target(tgt)
