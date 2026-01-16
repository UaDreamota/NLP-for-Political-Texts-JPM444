import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.baselines_bulk import run_baselines


parser = argparse.ArgumentParser()

#I plan to add: (1) Which bulk of models to use: scikit-learn, berts or apis with potential DeepLearning(?)
#I plan to add: (2) Maybe some general hyperparameters for each model? 
#I plan to add: (3) Path for inference? 
#I plan to add: (4) Maybe some flags on how expressive to be with different vizualizations
#I plan to add: (5) Deep Learning Approach with LSTMs? w1

# General parameters flags
parser.add_argument("--seed", default=42, type=int, help="Random seed for replication")
parser.add_argument("--data", default="data/belgium_newspaper_new_filter.csv", type=str, help="Path to the dataset")
parser.add_argument("--vizuals", default=True, type=bool, help="Regenerate the graphs for vizualization")
parser.add_argument("--save_metrics", default=True, type=bool, help="Save model params/metrics to CSV files")
parser.add_argument("--outputs_dir", default="outputs", type=str, help="Directory to write metrics CSVs")



# Scikit-learn model (general supervised ones) flags
parser.add_argument("--baseline_training", default=False, type=bool, help="Rerun the scikit baselines?")
parser.add_argument("--target_var", default="political", type=str, help="Target variable to predict: 'political' or 'domestic'")
parser.add_argument("--verbose", default=True, type=bool, help="Verbose baseline training output")
parser.add_argument("--grid_verbose", default=2, type=int, help="GridSearchCV verbosity level")
parser.add_argument("--cv", default=5, type=int, help="Cross-validation folds for grid search")
parser.add_argument("--n_jobs", default=1, type=int, help="Parallel jobs for grid search (-1 uses all cores)")
parser.add_argument("--baseline_models", default="", type=str, help="Comma-separated candidate names to train (e.g. tfidf_log_reg,count_linear_svm)")

#BERT transformers flags
parser.add_argument("--transformers_zero", default=False, type=bool, help="Zero-shot predict, evaluate BERT/ROBERTA type models")
parser.add_argument("--transformers_tune", default=False, type=bool, help="Fine-tune, predict, evaluate BERT/ROBERTA type models")
parser.add_argument("--epochs_b", default=3, type=int, help="Transformer fine-tune epochs")
parser.add_argument("--lr_b", default=2e-5, type=float, help="Transformer fine-tune learning rate")
parser.add_argument("--weight_decay_b", default=0.01, type=float, help="Transformer fine-tune weight decay")
parser.add_argument("--dropout_b", default=None, type=float, help="Transformer dropout override")
parser.add_argument("--bert_model", default=None, type=str, help="Transformer model name for fine-tuning")
parser.add_argument("--bert_max_length", default=256, type=int, help="Max token length for transformers")
parser.add_argument("--bert_batch_size", default=8, type=int, help="Per-device batch size for transformers")
parser.add_argument("--bert_zero_model", default=None, type=str, help="Zero-shot model name")
parser.add_argument("--bert_zero_max_samples", default=None, type=int, help="Limit zero-shot samples for speed")
parser.add_argument("--bert_zero_batch_size", default=8, type=int, help="Zero-shot pipeline batch size")

# ChatGPT flags
parser.add_argument("--api", default=False, type=bool, help="Set to use rerun the API inference (API KEYS ARE NEEDED)")

def _metrics_path(outputs_dir, base_name, target_var):
    filename = f"{base_name}_{target_var}.csv".replace("/", "_")
    return Path(outputs_dir) / filename


def _serialize_cell(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, default=str)
    return value


def _append_csv(df, path):
    path = Path(path)
    df = df.copy()
    for col in df.columns:
        df[col] = df[col].apply(_serialize_cell)

    if path.exists() and path.stat().st_size > 0:
        existing_cols = pd.read_csv(path, nrows=0).columns.tolist()
        for col in existing_cols:
            if col not in df.columns:
                df[col] = None
        for col in df.columns:
            if col not in existing_cols:
                existing_cols.append(col)
        df = df[existing_cols]
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


def main(main_args):
    random.seed(main_args.seed)
    np.random.seed(main_args.seed)
    if main_args.save_metrics:
        Path(main_args.outputs_dir).mkdir(parents=True, exist_ok=True)

    if main_args.baseline_training:
        model_names = [m.strip() for m in main_args.baseline_models.split(",") if m.strip()]
        results = run_baselines(
            main_args.data,
            main_args.target_var,
            random_state=main_args.seed,
            cv=main_args.cv,
            n_jobs=main_args.n_jobs,
            verbose=main_args.verbose,
            grid_verbose=main_args.grid_verbose,
            model_names=model_names or None,
        )
        if not results.empty:
            best = results.iloc[0]
            print(f"[baseline] Best: {best['model']} (f1={best['f1']:.4f})")
            if main_args.save_metrics:
                baseline_df = results.copy()
                baseline_df["target_var"] = main_args.target_var
                baseline_df["cv"] = main_args.cv
                baseline_df["n_jobs"] = main_args.n_jobs
                baseline_df["grid_verbose"] = main_args.grid_verbose
                baseline_df["data_path"] = main_args.data
                baseline_df["baseline_models"] = main_args.baseline_models or "all"
                baseline_path = _metrics_path(main_args.outputs_dir, "baselines_metrics", main_args.target_var)
                _append_csv(baseline_df, baseline_path)
    if main_args.api:
        from scripts import api_models

        f1 = api_models.send_requests(main_args.target_var, data_path=main_args.data, seed=main_args.seed)
        if main_args.save_metrics:
            pred_path = Path(f"predictions_{main_args.target_var}_{api_models.model}.csv".replace("/", "_"))
            n_samples = None
            positive_rate = None
            if pred_path.exists():
                pred_df = pd.read_csv(pred_path)
                if "y_true" in pred_df.columns:
                    n_samples = len(pred_df)
                    positive_rate = pred_df["y_true"].mean()

            row = {
                "model": api_models.model,
                "target_var": main_args.target_var,
                "data_path": main_args.data,
                "f1": f1,
                "n_samples": n_samples,
                "positive_rate": positive_rate,
                "predictions_path": str(pred_path),
            }
            api_path = _metrics_path(main_args.outputs_dir, "api_metrics", main_args.target_var)
            _append_csv(pd.DataFrame([row]), api_path)
    if main_args.transformers_tune:
        from scripts.bt_transformers import berts_orchestrated_tune, DEFAULT_MODEL_TUNE

        log_history_path = None
        if main_args.save_metrics or main_args.vizuals:
            log_history_path = _metrics_path(main_args.outputs_dir, "transformers_tune_log", main_args.target_var)

        metrics = berts_orchestrated_tune(
            main_args.target_var, 
            data_path=main_args.data,
            epochs=main_args.epochs_b,
            lr=main_args.lr_b,
            weight_decay=main_args.weight_decay_b,
            dropout=main_args.dropout_b,
            model_name=main_args.bert_model or None,
            max_length=main_args.bert_max_length,
            batch_size=main_args.bert_batch_size,
            random_state=main_args.seed,
            log_history_path=str(log_history_path) if log_history_path else None,
        )
        if main_args.save_metrics:
            model_name = main_args.bert_model or DEFAULT_MODEL_TUNE
            row = {
                "model": model_name,
                "target_var": main_args.target_var,
                "data_path": main_args.data,
                "epochs": main_args.epochs_b,
                "lr": main_args.lr_b,
                "weight_decay": main_args.weight_decay_b,
                "dropout": main_args.dropout_b,
                "max_length": main_args.bert_max_length,
                "batch_size": main_args.bert_batch_size,
            }
            if isinstance(metrics, dict):
                row.update(metrics)
            transformers_path = _metrics_path(main_args.outputs_dir, "transformers_tune_metrics", main_args.target_var)
            _append_csv(pd.DataFrame([row]), transformers_path)
            
    if main_args.transformers_zero:
        from scripts.bt_transformers import berts_orchestrated_zero, DEFAULT_MODEL_ZERO

        f1 = berts_orchestrated_zero(
            main_args.target_var,
            data_path=main_args.data,
            model_name=main_args.bert_zero_model or None,
            max_samples=main_args.bert_zero_max_samples,
            batch_size=main_args.bert_zero_batch_size,
            random_state=main_args.seed,
        )
        if main_args.save_metrics:
            model_name = main_args.bert_zero_model or DEFAULT_MODEL_ZERO
            row = {
                "model": model_name,
                "target_var": main_args.target_var,
                "data_path": main_args.data,
                "max_samples": main_args.bert_zero_max_samples,
                "batch_size": main_args.bert_zero_batch_size,
                "f1": f1,
            }
            transformers_zero_path = _metrics_path(main_args.outputs_dir, "transformers_zero_metrics", main_args.target_var)
            _append_csv(pd.DataFrame([row]), transformers_zero_path)

    if main_args.vizuals:
        from scripts.vizuals import generate_all_plots

        generate_all_plots(main_args.outputs_dir, main_args.target_var)

    return None

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
