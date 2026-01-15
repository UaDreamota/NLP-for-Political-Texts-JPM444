from dataclasses import dataclass
from typing import Any, Dict

import argparse

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from scripts.data_processing import load_processing


@dataclass
class Candidate:
    name: str
    pipe: BaseEstimator
    params: Dict[str, Any]


models = [
    Candidate(
        name="tfidf_log_reg",
        pipe=Pipeline([("vec", TfidfVectorizer()), ("clf", LogisticRegression())]),
        params={
            "vec__ngram_range": [(1, 1), (1, 2)],
            "vec__min_df": [1, 2],
            "vec__max_df": [0.9, 1.0],
            "vec__sublinear_tf": [False, True],
            "clf__C": [0.1, 1.0, 10.0],
            "clf__class_weight": [None, "balanced"],
        },
    ),
    Candidate(
        name="tfidf_linear_svm",
        pipe=Pipeline([("vec", TfidfVectorizer()), ("clf", LinearSVC())]),
        params={
            "vec__ngram_range": [(1, 1), (1, 2)],
            "vec__min_df": [1, 2],
            "vec__max_df": [0.9, 1.0],
            "vec__sublinear_tf": [False, True],
            "clf__C": [0.1, 1.0, 10.0],
            "clf__class_weight": [None, "balanced"],
        },
    ),
    Candidate(
        name="tfidf_naive_bayes",
        pipe=Pipeline([("vec", TfidfVectorizer()), ("clf", MultinomialNB())]),
        params={
            "vec__ngram_range": [(1, 1), (1, 2)],
            "vec__min_df": [1, 2],
            "vec__max_df": [0.9, 1.0],
            "vec__sublinear_tf": [False, True],
            "clf__alpha": [0.1, 1.0],
        },
    ),
    Candidate(
        name="count_log_reg",
        pipe=Pipeline([("vec", CountVectorizer()), ("clf", LogisticRegression())]),
        params={
            "vec__ngram_range": [(1, 1), (1, 2)],
            "vec__min_df": [1, 2],
            "vec__max_df": [0.9, 1.0],
            "clf__C": [0.1, 1.0, 10.0],
            "clf__class_weight": [None, "balanced"],
        },
    ),
    Candidate(
        name="count_linear_svm",
        pipe=Pipeline([("vec", CountVectorizer()), ("clf", LinearSVC())]),
        params={
            "vec__ngram_range": [(1, 1), (1, 2)],
            "vec__min_df": [1, 2],
            "vec__max_df": [0.9, 1.0],
            "clf__C": [0.1, 1.0, 10.0],
            "clf__class_weight": [None, "balanced"],
        },
    ),
]


def main():
    bg = load_processing("")
    pass


if __name__ == "__main__":
    main()
