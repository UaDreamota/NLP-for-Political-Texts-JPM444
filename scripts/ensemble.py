from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import random
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, ParameterGrid, train_test_split

from scripts import baselines_bulk
from scripts.data_processing import load_processing


@dataclass
class TrainedModel:
    name: str
    estimator: BaseEstimator
    f1: float
    best_params: Dict[str, Any]
    fit_time_s: float
    y_pred: np.ndarray
    is_linear: bool


def _is_linear_name(name: str) -> bool:
    """Treat logistic regression and linear SVM variants as linear models."""
    return "log_reg" in name or "linear_svm" in name


def _select_candidates(
    candidate_names: Optional[Sequence[str]],
    verbose: bool,
) -> List[baselines_bulk.Candidate]:
    candidates = baselines_bulk.models
    if candidate_names:
        requested = set(candidate_names)
        candidates = [m for m in baselines_bulk.models if m.name in requested]
        missing = sorted(requested - {m.name for m in baselines_bulk.models})
        if missing and verbose:
            print(f"[ensemble] Unknown model names ignored: {missing}")
        if not candidates:
            raise ValueError("No valid model names provided for ensemble training")
    return candidates


def _prepare_split(
    data: pd.DataFrame,
    target_var: str,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    features = baselines_bulk._prepare_features(data)
    y = data[target_var]
    train_idx, test_idx = train_test_split(
        data.index.to_numpy(),
        test_size=test_size,
        random_state=random_state,
    )
    X_train = features.loc[train_idx]
    X_test = features.loc[test_idx]
    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]
    return X_train, X_test, y_train, y_test


def _train_candidate(
    candidate: baselines_bulk.Candidate,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    scoring: str,
    cv: int,
    n_jobs: int,
    grid_verbose: int,
    random_state: int,
    verbose: bool,
) -> TrainedModel:
    pipe = clone(candidate.pipe)
    pipe_params = pipe.get_params()
    if "clf__random_state" in pipe_params:
        pipe.set_params(clf__random_state=random_state)
    if "poly__random_state" in pipe_params:
        pipe.set_params(poly__random_state=random_state)

    uses_dataframe_features = False
    if hasattr(pipe, "steps") and pipe.steps:
        uses_dataframe_features = isinstance(pipe.steps[0][1], ColumnTransformer)
    X_train_used = X_train if uses_dataframe_features else X_train["description"]
    X_test_used = X_test if uses_dataframe_features else X_test["description"]

    if candidate.params:
        param_count = len(ParameterGrid(candidate.params))
        if verbose:
            print(f"[ensemble] {candidate.name} param combos: {param_count}")
        search = GridSearchCV(
            pipe,
            candidate.params,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            verbose=grid_verbose if verbose else 0,
        )
        start = time.perf_counter()
        search.fit(X_train_used, y_train)
        elapsed = time.perf_counter() - start
        estimator = search.best_estimator_
        best_params = search.best_params_
        if verbose:
            print(f"[ensemble] {candidate.name} grid done in {elapsed:.1f}s")
    else:
        start = time.perf_counter()
        pipe.fit(X_train_used, y_train)
        elapsed = time.perf_counter() - start
        estimator = pipe
        best_params = {}
        if verbose:
            print(f"[ensemble] {candidate.name} fit done in {elapsed:.1f}s")

    y_pred = estimator.predict(X_test_used)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    print(f"[ensemble] {candidate.name}: f1={f1:.4f}")

    return TrainedModel(
        name=candidate.name,
        estimator=estimator,
        f1=f1,
        best_params=best_params,
        fit_time_s=elapsed,
        y_pred=np.asarray(y_pred),
        is_linear=_is_linear_name(candidate.name),
    )


def _ensemble_predictions(models_to_combine: List[TrainedModel]) -> np.ndarray:
    if not models_to_combine:
        raise ValueError("No models provided for ensembling")
    preds = np.vstack([m.y_pred for m in models_to_combine])
    votes = preds.sum(axis=0)
    half = len(models_to_combine) / 2
    primary = models_to_combine[0].y_pred
    ensemble = []
    for i, vote in enumerate(votes):
        if vote > half:
            ensemble.append(1)
        elif vote < half:
            ensemble.append(0)
        else:
            ensemble.append(int(primary[i]))
    return np.asarray(ensemble)


def train_linear_ensemble(
    data_path: str,
    target_var: str,
    candidate_names: Optional[Sequence[str]] = None,
    top_k: int = 2,
    test_size: float = 0.2,
    scoring: str = "f1",
    cv: int = 5,
    n_jobs: int = -1,
    random_state: int = 42,
    verbose: bool = True,
    grid_verbose: int = 2,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Train baseline models and build a simple majority-vote ensemble
    from the best-performing linear models (max two).
    Returns an ensemble summary row and a per-model metrics dataframe.
    """
    random.seed(random_state)
    np.random.seed(random_state)
    data = load_processing(data_path)
    if target_var not in data.columns:
        raise ValueError(f"Target column '{target_var}' not found in data")

    X_train, X_test, y_train, y_test = _prepare_split(data, target_var, test_size, random_state)
    candidates = _select_candidates(candidate_names, verbose)

    trained_models: List[TrainedModel] = []
    for idx, candidate in enumerate(candidates, start=1):
        if verbose:
            print(f"[ensemble] Starting {idx}/{len(candidates)}: {candidate.name}")
        trained_models.append(
            _train_candidate(
                candidate,
                X_train,
                X_test,
                y_train,
                y_test,
                scoring,
                cv,
                n_jobs,
                grid_verbose,
                random_state,
                verbose,
            )
        )

    linear_models = [m for m in trained_models if m.is_linear]
    if not linear_models:
        raise ValueError("No linear models were trained; cannot build ensemble")
    linear_models = sorted(linear_models, key=lambda m: m.f1, reverse=True)

    max_members = max(1, min(top_k, 2))
    if top_k > 2 and verbose:
        print(f"[ensemble] Requested top_k={top_k}, capping to 2 for linear ensemble")
    selected = linear_models[:max_members]
    ensemble_pred = _ensemble_predictions(selected)
    ensemble_f1 = f1_score(y_test, ensemble_pred, pos_label=1)

    model_rows = []
    selected_names = {m.name for m in selected}
    for model in trained_models:
        model_rows.append(
            {
                "model": model.name,
                "f1": model.f1,
                "best_params": model.best_params,
                "fit_time_s": model.fit_time_s,
                "is_linear": model.is_linear,
                "selected_for_ensemble": model.name in selected_names,
                "target_var": target_var,
                "data_path": data_path,
                "cv": cv,
                "n_jobs": n_jobs,
            }
        )
    model_df = pd.DataFrame(model_rows).sort_values("f1", ascending=False)

    ensemble_row = {
        "model": "linear_vote",
        "ensemble_f1": ensemble_f1,
        "members": [m.name for m in selected],
        "member_f1": {m.name: m.f1 for m in selected},
        "target_var": target_var,
        "data_path": data_path,
        "top_k_requested": top_k,
        "top_k_used": len(selected),
        "n_candidates_trained": len(trained_models),
        "n_linear_candidates": len(linear_models),
        "cv": cv,
        "n_jobs": n_jobs,
        "test_size": test_size,
        "random_state": random_state,
    }
    print(f"[ensemble] Linear vote ({' + '.join(ensemble_row['members'])}): f1={ensemble_f1:.4f}")
    return ensemble_row, model_df
