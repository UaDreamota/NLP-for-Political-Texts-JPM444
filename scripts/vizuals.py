from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _safe_name(value):
    return str(value).replace("/", "_")


def _load_metrics(path):
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _summarize_scores(df, score_col="f1"):
    if "model" not in df.columns or score_col not in df.columns:
        return None
    summary = (
        df.groupby("model")[score_col]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": score_col})
    )
    summary["std"] = summary["std"].fillna(0)
    return summary.sort_values(score_col, ascending=False)


def _bar_plot(df, title, ylabel, out_path):
    if df is None or df.empty:
        return
    plt.figure(figsize=(10, 6))
    yerr = df["std"] if "std" in df.columns else None
    plt.bar(df["model"], df["f1"], yerr=yerr, color="#4C78A8")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _scatter_plot(df, x_col, y_col, title, xlabel, ylabel, out_path):
    if df is None or df.empty or x_col not in df.columns or y_col not in df.columns:
        return
    plt.figure(figsize=(7, 6))
    plt.scatter(df[x_col], df[y_col], color="#54A24B")
    for _, row in df.iterrows():
        if "model" in df.columns:
            plt.annotate(str(row["model"]), (row[x_col], row[y_col]))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_training_curve(log_path, out_path):
    df = _load_metrics(log_path)
    if df is None or df.empty:
        return

    x_key = "step" if "step" in df.columns else "epoch"
    fig, axes = plt.subplots(2, 1, figsize=(9, 8))

    if "loss" in df.columns:
        axes[0].plot(df[x_key], df["loss"], color="#4C78A8")
        axes[0].set_title("Training Loss")
        axes[0].set_xlabel(x_key)
        axes[0].set_ylabel("loss")
    else:
        axes[0].axis("off")

    eval_df = df.copy()
    has_eval = False
    if "eval_f1" in eval_df.columns:
        eval_df = eval_df[eval_df["eval_f1"].notna()]
        axes[1].plot(eval_df[x_key], eval_df["eval_f1"], color="#F58518", label="eval_f1")
        has_eval = True
    if "eval_loss" in df.columns:
        eval_loss_df = df[df["eval_loss"].notna()]
        axes[1].plot(eval_loss_df[x_key], eval_loss_df["eval_loss"], color="#E45756", label="eval_loss")
        has_eval = True

    if has_eval:
        axes[1].set_title("Evaluation Metrics")
        axes[1].set_xlabel(x_key)
        axes[1].legend()
    else:
        axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def generate_all_plots(outputs_dir, target_var):
    outputs_dir = Path(outputs_dir)
    plots_dir = outputs_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    safe_target = _safe_name(target_var)

    baseline_path = outputs_dir / f"baselines_metrics_{safe_target}.csv"
    baseline_df = _load_metrics(baseline_path)
    baseline_summary = _summarize_scores(baseline_df) if baseline_df is not None else None
    _bar_plot(
        baseline_summary,
        title=f"Baselines F1 ({target_var})",
        ylabel="f1",
        out_path=plots_dir / f"baselines_f1_{safe_target}.png",
    )
    if baseline_df is not None and "fit_time_s" in baseline_df.columns:
        _scatter_plot(
            baseline_df,
            x_col="fit_time_s",
            y_col="f1",
            title=f"Baseline F1 vs Fit Time ({target_var})",
            xlabel="fit_time_s",
            ylabel="f1",
            out_path=plots_dir / f"baselines_f1_vs_time_{safe_target}.png",
        )

    tune_path = outputs_dir / f"transformers_tune_metrics_{safe_target}.csv"
    tune_df = _load_metrics(tune_path)
    tune_summary = _summarize_scores(tune_df) if tune_df is not None else None
    _bar_plot(
        tune_summary,
        title=f"Transformer Tune F1 ({target_var})",
        ylabel="f1",
        out_path=plots_dir / f"transformers_tune_f1_{safe_target}.png",
    )

    zero_path = outputs_dir / f"transformers_zero_metrics_{safe_target}.csv"
    zero_df = _load_metrics(zero_path)
    zero_summary = _summarize_scores(zero_df) if zero_df is not None else None
    _bar_plot(
        zero_summary,
        title=f"Transformer Zero-shot F1 ({target_var})",
        ylabel="f1",
        out_path=plots_dir / f"transformers_zero_f1_{safe_target}.png",
    )

    api_path = outputs_dir / f"api_metrics_{safe_target}.csv"
    api_df = _load_metrics(api_path)
    api_summary = _summarize_scores(api_df) if api_df is not None else None
    _bar_plot(
        api_summary,
        title=f"API Model F1 ({target_var})",
        ylabel="f1",
        out_path=plots_dir / f"api_f1_{safe_target}.png",
    )

    log_path = outputs_dir / f"transformers_tune_log_{safe_target}.csv"
    _plot_training_curve(
        log_path,
        out_path=plots_dir / f"transformers_tune_training_{safe_target}.png",
    )
