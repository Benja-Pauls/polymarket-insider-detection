"""Model training and evaluation for the insider-detection classifier.

Models considered:

  * Logistic Regression (L2)    — interpretable baseline
  * Random Forest               — nonlinear tabular baseline
  * XGBoost                     — gradient-boosted ensemble, typically SOTA on
                                  small-medium tabular
  * LightGBM                    — faster GB alternative, histogram-based
  * Isolation Forest            — unsupervised anomaly detector, useful for
                                  markets where we lack a confident label

All splits are BY MARKET (``condition_id``) — never by feature-row — so features
from the same market never straddle train/test boundaries. This matters even
though each market is one row in the current pipeline, because (a) multi-token
and multi-outcome markets may produce multiple rows in a future version, and
(b) it keeps the split definition reproducible by condition_id.

Evaluation metrics:

  * ROC-AUC          — threshold-agnostic ranking
  * PR-AUC           — more informative than ROC when positives are sparse
  * Precision@k      — what fraction of our top-k flagged markets were positives
                       (practitioner-relevant)
  * Calibration      — reliability diagram / Brier score
  * Per-tier metrics — broken out historical vs live to check that the model
                       generalizes past the resync snapshot boundary
"""
from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

POSITIVE_LABELS = {"weak_positive", "strong_positive"}
NEGATIVE_LABELS = {"negative", "strong_negative"}


# ----------------------------------------------------------------------
# Data prep
# ----------------------------------------------------------------------

def select_feature_columns(feats: pd.DataFrame) -> list[str]:
    """Columns to feed into the model — everything numeric except IDs and labels."""
    drop = {
        "condition_id",
        "source_tier",
        "label",
        "label_source",
        "label_confidence",
        "split",
    }
    # meta_winning_outcome_index leaks the outcome — drop it
    drop.add("meta_winning_outcome_index")
    drop.add("dir_resolution_outcome_index")
    drop |= {c for c in feats.columns if c.startswith("_pred_")}
    numeric = feats.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric if c not in drop]


def make_xy(
    feats: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Return (X, y, split, condition_id) after merging features with labels.

    Labels not in POSITIVE_LABELS ∪ NEGATIVE_LABELS (e.g. "ambiguous",
    "excluded_low_activity") are dropped.
    """
    merged = feats.merge(
        labels[["condition_id", "label", "split"]],
        on="condition_id",
        how="inner",
    )
    # Filter to labeled
    keep = merged["label"].isin(POSITIVE_LABELS | NEGATIVE_LABELS)
    merged = merged[keep].reset_index(drop=True)

    cols = select_feature_columns(merged)
    X = merged[cols].copy()
    y = merged["label"].isin(POSITIVE_LABELS).astype(int)
    split = merged["split"]
    cid = merged["condition_id"]
    return X, y, split, cid


# ----------------------------------------------------------------------
# Model factory
# ----------------------------------------------------------------------

@dataclass
class ModelSpec:
    name: str
    build: Any  # callable returning sklearn-compatible estimator


def _logreg():
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="l2",
                C=1.0,
                max_iter=2000,
                class_weight="balanced",
                random_state=17,
            )),
        ]
    )


def _rf():
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=400,
                max_depth=None,
                min_samples_leaf=3,
                class_weight="balanced",
                n_jobs=-1,
                random_state=17,
            )),
        ]
    )


def _xgb():
    import xgboost as xgb
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("clf", xgb.XGBClassifier(
                n_estimators=500,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=17,
                eval_metric="aucpr",
                tree_method="hist",
            )),
        ]
    )


def _lgb():
    import lightgbm as lgb
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("clf", lgb.LGBMClassifier(
                n_estimators=600,
                max_depth=-1,
                num_leaves=63,
                learning_rate=0.05,
                min_child_samples=5,
                class_weight="balanced",
                random_state=17,
                verbose=-1,
            )),
        ]
    )


def default_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec("logreg",    _logreg),
        ModelSpec("randforest", _rf),
        ModelSpec("xgboost",   _xgb),
        ModelSpec("lightgbm",  _lgb),
    ]


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

@dataclass
class ModelResult:
    name: str
    train_roc_auc: float
    val_roc_auc: float
    test_roc_auc: float
    train_pr_auc: float
    val_pr_auc: float
    test_pr_auc: float
    val_brier: float
    test_brier: float
    test_precision_at_k: dict[int, float]
    feature_importance: pd.Series | None = None
    estimator: Any = None

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d.pop("estimator", None)
        d.pop("feature_importance", None)
        return d


def _precision_at_k(y_true: np.ndarray, y_score: np.ndarray, ks: list[int]) -> dict[int, float]:
    order = np.argsort(-y_score)
    y_ranked = y_true[order]
    out = {}
    for k in ks:
        k = min(k, len(y_ranked))
        if k == 0:
            out[k] = float("nan")
        else:
            out[k] = float(y_ranked[:k].mean())
    return out


def _safe_roc_auc(y, s):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def _safe_pr_auc(y, s):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, s))


def _feature_importance(estimator, feature_names: list[str]) -> pd.Series | None:
    """Best-effort extract feature importance from the trained pipeline."""
    clf = estimator.named_steps.get("clf") if hasattr(estimator, "named_steps") else estimator
    if hasattr(clf, "feature_importances_"):
        return pd.Series(clf.feature_importances_, index=feature_names).sort_values(ascending=False)
    if hasattr(clf, "coef_"):
        coef = clf.coef_.ravel() if clf.coef_.ndim > 1 else clf.coef_
        return pd.Series(np.abs(coef), index=feature_names).sort_values(ascending=False)
    return None


def train_and_evaluate(
    spec: ModelSpec,
    X: pd.DataFrame,
    y: pd.Series,
    split: pd.Series,
    *,
    ks: list[int] | None = None,
) -> ModelResult:
    ks = ks or [5, 10, 20, 50, 100]
    Xtr, ytr = X[split == "train"], y[split == "train"]
    Xva, yva = X[split == "val"], y[split == "val"]
    Xte, yte = X[split == "test"], y[split == "test"]

    est = spec.build()
    est.fit(Xtr, ytr)

    def _proba(model, features):
        if hasattr(model, "predict_proba"):
            return model.predict_proba(features)[:, 1]
        if hasattr(model, "decision_function"):
            s = model.decision_function(features)
            # min-max to [0,1] for comparability
            s_min, s_max = s.min(), s.max()
            rng = s_max - s_min
            return (s - s_min) / rng if rng > 0 else np.zeros_like(s)
        return model.predict(features).astype(float)

    str_scores = _proba(est, Xtr)
    sva = _proba(est, Xva)
    ste = _proba(est, Xte)

    result = ModelResult(
        name=spec.name,
        train_roc_auc=_safe_roc_auc(ytr, str_scores),
        val_roc_auc=_safe_roc_auc(yva, sva),
        test_roc_auc=_safe_roc_auc(yte, ste),
        train_pr_auc=_safe_pr_auc(ytr, str_scores),
        val_pr_auc=_safe_pr_auc(yva, sva),
        test_pr_auc=_safe_pr_auc(yte, ste),
        val_brier=float(brier_score_loss(yva, sva)) if len(yva) else float("nan"),
        test_brier=float(brier_score_loss(yte, ste)) if len(yte) else float("nan"),
        test_precision_at_k=_precision_at_k(yte.values, ste, ks),
        feature_importance=_feature_importance(est, X.columns.tolist()),
        estimator=est,
    )
    return result


def evaluate_isolation_forest(
    X: pd.DataFrame,
    y: pd.Series,
    split: pd.Series,
    *,
    contamination: float = 0.1,
) -> ModelResult:
    """Unsupervised anomaly detector — trains on ONLY the negatives from the training split."""
    Xtr = X[(split == "train") & (y == 0)]
    Xva, yva = X[split == "val"], y[split == "val"]
    Xte, yte = X[split == "test"], y[split == "test"]

    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", IsolationForest(
            n_estimators=400,
            contamination=contamination,
            random_state=17,
            n_jobs=-1,
        )),
    ])
    model.fit(Xtr)

    # Higher anomaly score = higher "insider" likelihood → invert
    def _score(model, features):
        raw = model.decision_function(features)  # high = inlier
        return -raw

    sva = _score(model, Xva)
    ste = _score(model, Xte)
    return ModelResult(
        name="isolation_forest",
        train_roc_auc=float("nan"),
        val_roc_auc=_safe_roc_auc(yva, sva),
        test_roc_auc=_safe_roc_auc(yte, ste),
        train_pr_auc=float("nan"),
        val_pr_auc=_safe_pr_auc(yva, sva),
        test_pr_auc=_safe_pr_auc(yte, ste),
        val_brier=float("nan"),
        test_brier=float("nan"),
        test_precision_at_k=_precision_at_k(yte.values, ste, [5, 10, 20, 50, 100]),
        feature_importance=None,
        estimator=model,
    )


# ----------------------------------------------------------------------
# Full training run
# ----------------------------------------------------------------------

def run_all_models(
    features_path: Path,
    labels_path: Path,
    *,
    models_dir: Path,
    results_path: Path,
) -> pd.DataFrame:
    feats = pd.read_parquet(features_path)
    labels = pd.read_parquet(labels_path)
    X, y, split, cid = make_xy(feats, labels)
    log.info(
        "dataset: %d markets  [train=%d val=%d test=%d]  positives=%d (%.1f%%)",
        len(X), (split == "train").sum(), (split == "val").sum(), (split == "test").sum(),
        int(y.sum()), 100 * y.mean(),
    )

    results: list[ModelResult] = []
    for spec in default_model_specs():
        log.info("training %s…", spec.name)
        res = train_and_evaluate(spec, X, y, split)
        results.append(res)

    # Unsupervised
    log.info("training isolation_forest…")
    results.append(evaluate_isolation_forest(X, y, split))

    # Persist artifacts
    models_dir.mkdir(parents=True, exist_ok=True)
    for res in results:
        if res.estimator is not None:
            joblib.dump(res.estimator, models_dir / f"{res.name}.joblib")
        if res.feature_importance is not None:
            res.feature_importance.to_csv(
                models_dir / f"{res.name}__feature_importance.csv"
            )

    results_path.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([r.to_dict() for r in results])
    summary.to_csv(results_path, index=False)
    return summary
