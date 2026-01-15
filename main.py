import numpy as np
import numpy.typing as npt

import argparse

from sklearn.pipeline import make_pipeline, make_union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


from scripts.data_processing import load_processing
from scripts.api_models import send_requests


parser = argparse.ArgumentParser()

parser.add_argument("--data", default="belgium_newspaper_new_filter.csv", type=str, help="Path to the dataset")
#I plan to add: (1) Which bulk of models to use: scikit-learn, berts or apis with potential DeepLearning(?)
#I plan to add: (2) Maybe some general hyperparameters for each model? 
#I plan to add: (3) Path for inference? 
#I plan to add: (4) Maybe some flags on how expressive to be with different vizualizations
#I plan to add: (5) Deep Learning Approach with LSTMs? w1
parser.add_argument("--vizuals", default=False, type=bool, help="Regenerate the graphs for vizualization")
parser.add_argument("--baseline_training", default=True, type=bool, help="Rerun the scikit baselines?")
parser.add_argument("--api", default=False, type=bool, help="Set to use rerun the API inference (API KEYS ARE NEEDED)")
parser.add_argument("--target_var", default="political", type=str, help="Target variable to predict: 'political' or 'domestic'")


def main():
    
    return None

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)



