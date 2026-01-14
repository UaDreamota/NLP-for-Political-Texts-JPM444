import numpy as np
import numpy.typing as npt

import argparse

from sklearn.pipeline import make_pipeline, make_union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


parser = argparse.ArgumentParser()

parser.add_argument("--data", default="belgium_newspaper_new_filter.csv", type=str, help="Path to the dataset")
#I plan to add: (1) Which bulk of models to use: scikit-learn, berts or apis with potential DeepLearning(?)
#I plan to add: (2) Maybe some general hyperparameters for each model? 
#I plan to add: (3) Path for inference? 
#I plan to add: (4) Maybe some flags on how expressive to be with different vizualizations
#I plan to add: (5) Deep Learning Approach with LSTMs? 




def main():
    pass

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)



