

from dataclasses import dataclass
from typin import Dict

import argparse

import numpy as np

from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline 
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator

import pandas as pd



@dataclass
class Candidate
    name: str
    pipe: BaseEstimator
    parameters: Dict[]


models = [
    Candidate(
        name="baseline",
        pipe=Pipeline(['vec', TfidfVectorizer()), ('clf', LogisticRegression())]
        params={"ve"}
    ]



def load_processing(csv_file):
    data = pd.read_csv(csv_file)
    data = data[data['political' != 99]]
    data = data[data['political'].notna()]
    data = data[data['political'].astype(int64)]
    
    if [99, nan] in data['political'].unqiue():
        raise Exception("The processing of political category failed")
    
    data = data[data['domestic' != 99]]
    data = data[data['domestic'].notna()]
    data = data[data['domestic'].astype(int64)]
 
    if [99, nan] in data['domestic'].unqiue():
        raise Exception("The processing of domestic category failed")
  
    data = data['descriptions'].fillna()

    processed_data = data

    return processed_data


def ():
    pass




def main():
    bg = load_processing("")
    pass


if __name__ == "__main__":
    main()
