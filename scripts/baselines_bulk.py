from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import time
import random

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.kernel_approximation import PolynomialCountSketch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, ParameterGrid, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

from scripts.data_processing import load_processing


@dataclass
class Candidate:
    name: str
    pipe: BaseEstimator
    params: Union[Dict[str, Any], List[Dict[str, Any]]]


def _prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=data.index)
    features["description"] = data["description"].fillna("")
    if "headline" in data.columns:
        features["headline"] = data["headline"].fillna("")
    else:
        features["headline"] = ""

    if "date" in data.columns:
        year = pd.to_datetime(data["date"], dayfirst=True, errors="coerce").dt.year
        features["year"] = year.fillna(-1).astype(int).astype(str)
    else:
        features["year"] = "-1"

    return features


models = [
    Candidate(
        name="tfidf_log_reg",
        pipe=Pipeline(
            [
                ("vec", TfidfVectorizer()),
                ("clf", LogisticRegression(solver="saga", max_iter=2000)),
            ]
        ),
        params=[
            {
                "vec__analyzer": ["word"],
                "vec__ngram_range": [(1, 2)],
                "vec__min_df": [0.001, 0.005],
                "vec__max_df": [0.9],
                "vec__sublinear_tf": [True],
                "vec__stop_words": [None, "english"],
                "clf__C": [0.1, 1.0, 10.0],
                "clf__class_weight": [None, "balanced"],
            },
            {
                "vec__analyzer": ["char_wb"],
                "vec__ngram_range": [(3, 5)],
                "vec__min_df": [0.002],
                "vec__max_df": [0.9],
                "vec__sublinear_tf": [True],
                "clf__C": [1.0],
                "clf__class_weight": [None, "balanced"],
            },
        ],
    ),
    Candidate(
        name="tfidf_log_reg_desc_headline_year",
        pipe=Pipeline(
            [
                (
                    "features",
                    ColumnTransformer(
                        transformers=[
                            (
                                "desc",
                                TfidfVectorizer(
                                    analyzer="char_wb",
                                    ngram_range=(3, 5),
                                    min_df=0.002,
                                    max_df=0.9,
                                    sublinear_tf=True,
                                ),
                                "description",
                            ),
                            (
                                "headline",
                                TfidfVectorizer(
                                    analyzer="char_wb",
                                    ngram_range=(3, 5),
                                    min_df=0.002,
                                    max_df=0.9,
                                    sublinear_tf=True,
                                ),
                                "headline",
                            ),
                            ("year", OneHotEncoder(handle_unknown="ignore"), ["year"]),
                        ],
                        remainder="drop",
                    ),
                ),
                ("clf", LogisticRegression(solver="saga", max_iter=4000, C=1.0, class_weight="balanced")),
            ]
        ),
        params={},
    ),
    Candidate(
        name="tfidf_linear_svm",
        pipe=Pipeline([("vec", TfidfVectorizer()), ("clf", LinearSVC(max_iter=5000))]),
        params=[
            {
                "vec__analyzer": ["word"],
                "vec__ngram_range": [(1, 2)],
                "vec__min_df": [0.001, 0.005],
                "vec__max_df": [0.9],
                "vec__max_features": [100000],
                "vec__sublinear_tf": [True],
                "vec__stop_words": [None, "english"],
                "clf__C": [0.1, 1.0, 3],
                "clf__class_weight": [None, "balanced"],
            },
            {
                "vec__analyzer": ["char_wb"],
                "vec__ngram_range": [(3, 5)],
                "vec__min_df": [0.002],
                "vec__max_df": [0.9],
                "vec__sublinear_tf": [True],
                "clf__C": [1.0],
                "clf__class_weight": [None, "balanced"],
            },
        ],
    ),
    Candidate(
        name="tfidf_naive_bayes",
        pipe=Pipeline([("vec", TfidfVectorizer()), ("clf", MultinomialNB())]),
        params={
            "vec__ngram_range": [(1, 1), (1, 2)],
            "vec__min_df": [0.001, 0.005],
            "vec__max_df": [0.9],
            "vec__sublinear_tf": [True],
            "clf__alpha": [0.1, 1.0],
        },
    ),
    Candidate(
        name="tfidf_log_reg_poly2",
        pipe=Pipeline(
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
                ("poly", PolynomialCountSketch(degree=2, coef0=1, n_components=8192)),
                ("clf", LogisticRegression(solver="saga", max_iter=4000, C=1.0, class_weight="balanced")),
            ]
        ),
        params={},
    ),
    Candidate(
        name="tfidf_log_reg_poly4",
        pipe=Pipeline(
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
                ("poly", PolynomialCountSketch(degree=4, coef0=1, n_components=8192)),
                ("clf", LogisticRegression(solver="saga", max_iter=4000, C=1.0, class_weight="balanced")),
            ]
        ),
        params={},
    ),
    Candidate(
        name="tfidf_log_reg_poly5",
        pipe=Pipeline(
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
                ("poly", PolynomialCountSketch(degree=5, coef0=1, n_components=8192)),
                ("clf", LogisticRegression(solver="saga", max_iter=4000, C=1.0, class_weight="balanced")),
            ]
        ),
        params={},
    ),
    Candidate(
        name="count_log_reg",
        pipe=Pipeline(
            [
                ("vec", CountVectorizer()),
                ("clf", LogisticRegression(solver="saga", max_iter=2000)),
            ]
        ),
        params={
            "vec__ngram_range": [(1, 1), (1, 2)],
            "vec__min_df": [0.001, 0.005],
            "vec__max_df": [0.9],
            "clf__C": [0.1, 1.0, 10.0],
            "clf__class_weight": [None, "balanced"],
        },
    ),
    Candidate(
        name="count_linear_svm",
        pipe=Pipeline([("vec", CountVectorizer()), ("clf", LinearSVC(max_iter=5000))]),
        params={
            "vec__ngram_range": [(1, 1), (1, 2)],
            "vec__min_df": [0.001, 0.005],
            "vec__max_df": [0.9],
            "clf__C": [0.1, 1.0, 10.0],
            "clf__class_weight": [None, "balanced"],
        },
    ),
]


def run_baselines(
    data_path: str,
    target_var: str,
    test_size: float = 0.2,
    random_state: int = 42,
    scoring: str = "f1",
    cv: int = 5,
    n_jobs: int = -1,
    verbose: bool = True,
    grid_verbose: int = 2,
    model_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    random.seed(random_state)
    np.random.seed(random_state)
    bg = load_processing(data_path)
    if target_var not in bg.columns:
        raise ValueError(f"Target column '{target_var}' not found in data")

    features = _prepare_features(bg)
    y = bg[target_var]
    train_idx, test_idx = train_test_split(
        bg.index.to_numpy(),
        test_size=test_size,
        random_state=random_state,
    )
    X_train_text = features.loc[train_idx, "description"]
    X_test_text = features.loc[test_idx, "description"]
    X_train_full = features.loc[train_idx]
    X_test_full = features.loc[test_idx]
    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]

    if verbose:
        total = len(bg)
        train_size = len(X_train_text)
        test_size_count = len(X_test_text)
        label_dist = bg[target_var].value_counts(normalize=True).to_dict()
        print(f"[baseline] Data: {data_path}")
        print(f"[baseline] Target: {target_var}")
        print(f"[baseline] Samples: {total} (train={train_size}, test={test_size_count})")
        print(f"[baseline] Label distribution: {label_dist}")
        print(f"[baseline] CV: {cv}, scoring={scoring}, n_jobs={n_jobs}")

    candidate_models = models
    if model_names:
        requested = set(model_names)
        candidate_models = [m for m in models if m.name in requested]
        missing = sorted(requested - {m.name for m in models})
        if missing and verbose:
            print(f"[baseline] Unknown model names ignored: {missing}")
        if not candidate_models:
            raise ValueError("No valid model names provided for baseline training")

    results = []
    for idx, candidate in enumerate(candidate_models, start=1):
        pipe_params = candidate.pipe.get_params()
        if "clf__random_state" in pipe_params:
            candidate.pipe.set_params(clf__random_state=random_state)
        if "poly__random_state" in pipe_params:
            candidate.pipe.set_params(poly__random_state=random_state)
        if verbose:
            print(f"[baseline] Starting {idx}/{len(candidate_models)}: {candidate.name}")

        uses_dataframe_features = isinstance(getattr(candidate.pipe, "steps", [[None, None]])[0][1], ColumnTransformer)
        X_train = X_train_full if uses_dataframe_features else X_train_text
        X_test = X_test_full if uses_dataframe_features else X_test_text

        if candidate.params:
            param_count = len(ParameterGrid(candidate.params))
            if verbose:
                print(f"[baseline] {candidate.name} param combos: {param_count}")
            search = GridSearchCV(
                candidate.pipe,
                candidate.params,
                scoring=scoring,
                cv=cv,
                n_jobs=n_jobs,
                verbose=grid_verbose if verbose else 0,
            )
            start = time.perf_counter()
            search.fit(X_train, y_train)
            elapsed = time.perf_counter() - start
            best_estimator = search.best_estimator_
            best_params = search.best_params_
            if verbose:
                print(f"[baseline] {candidate.name} grid done in {elapsed:.1f}s")
        else:
            start = time.perf_counter()
            candidate.pipe.fit(X_train, y_train)
            elapsed = time.perf_counter() - start
            best_estimator = candidate.pipe
            best_params = {}
            if verbose:
                print(f"[baseline] {candidate.name} fit done in {elapsed:.1f}s")

        y_pred = best_estimator.predict(X_test)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        results.append(
            {
                "model": candidate.name,
                "f1": f1,
                "best_params": best_params,
                "fit_time_s": elapsed,
            }
        )
        print(f"[baseline] {candidate.name}: f1={f1:.4f}")
        if best_params:
            print(f"[baseline] {candidate.name} best_params: {best_params}")

    return pd.DataFrame(results).sort_values("f1", ascending=False)


def main():
    data_path = "data/belgium_newspaper_new_filter.csv"
    bg = load_processing(data_path)
    return bg


if __name__ == "__main__":
    main()
