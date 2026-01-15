import argparse

from scripts.baselines_bulk import run_baselines


parser = argparse.ArgumentParser()

parser.add_argument("--data", default="data/belgium_newspaper_new_filter.csv", type=str, help="Path to the dataset")
#I plan to add: (1) Which bulk of models to use: scikit-learn, berts or apis with potential DeepLearning(?)
#I plan to add: (2) Maybe some general hyperparameters for each model? 
#I plan to add: (3) Path for inference? 
#I plan to add: (4) Maybe some flags on how expressive to be with different vizualizations
#I plan to add: (5) Deep Learning Approach with LSTMs? w1
parser.add_argument("--vizuals", default=False, type=bool, help="Regenerate the graphs for vizualization")
parser.add_argument("--baseline_training", default=False, type=bool, help="Rerun the scikit baselines?")
parser.add_argument("--api", default=False, type=bool, help="Set to use rerun the API inference (API KEYS ARE NEEDED)")
parser.add_argument("--target_var", default="political", type=str, help="Target variable to predict: 'political' or 'domestic'")
parser.add_argument("--verbose", default=True, type=bool, help="Verbose baseline training output")
parser.add_argument("--grid_verbose", default=2, type=int, help="GridSearchCV verbosity level")
parser.add_argument("--cv", default=5, type=int, help="Cross-validation folds for grid search")
parser.add_argument("--n_jobs", default=-1, type=int, help="Parallel jobs for grid search (-1 uses all cores)")
parser.add_argument("--baseline_models", default="", type=str, help="Comma-separated candidate names to train (e.g. tfidf_log_reg,count_linear_svm)")
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


def main(main_args):
    if main_args.baseline_training:
        model_names = [m.strip() for m in main_args.baseline_models.split(",") if m.strip()]
        results = run_baselines(
            main_args.data,
            main_args.target_var,
            cv=main_args.cv,
            n_jobs=main_args.n_jobs,
            verbose=main_args.verbose,
            grid_verbose=main_args.grid_verbose,
            model_names=model_names or None,
        )
        if not results.empty:
            best = results.iloc[0]
            print(f"[baseline] Best: {best['model']} (f1={best['f1']:.4f})")
    if main_args.api:
        from scripts.api_models import send_requests

        send_requests(main_args.target_var, data_path=main_args.data)
    if main_args.transformers_tune:
        from scripts.bt_transformers import berts_orchestrated_tune

        berts_orchestrated_tune(
            main_args.target_var, 
            data_path=main_args.data,
            epochs=main_args.epochs_b,
            lr=main_args.lr_b,
            weight_decay=main_args.weight_decay_b,
            dropout=main_args.dropout_b,
            model_name=main_args.bert_model or None,
            max_length=main_args.bert_max_length,
            batch_size=main_args.bert_batch_size,
        )
            
    if main_args.transformers_zero:
        from scripts.bt_transformers import berts_orchestrated_zero

        berts_orchestrated_zero(
            main_args.target_var,
            data_path=main_args.data,
            model_name=main_args.bert_zero_model or None,
            max_samples=main_args.bert_zero_max_samples,
            batch_size=main_args.bert_zero_batch_size,
        )


    return None

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
