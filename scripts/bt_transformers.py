# This one is for BERT and other transformer models.

from typing import Optional

import pandas as pd

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    pipeline,
    set_seed,
)

from scripts.data_processing import load_processing

DEFAULT_MODEL_TUNE = "xlm-roberta-base"
DEFAULT_MODEL_ZERO = "joeddav/xlm-roberta-large-xnli"


class TorchDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, max_length=self.max_length)
        enc["labels"] = int(self.labels[i])
        return enc


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "f1": f1_score(labels, preds, pos_label=1),
        "accuracy": accuracy_score(labels, preds),
    }


def train_transformer_model(
    X_train,
    y_train,
    X_test,
    y_test,
    model_name: str,
    num_labels: int = 2,
    max_length: int = 256,
    lr: float = 2e-5,
    epochs: int = 3,
    batch_size: int = 8,
    weight_decay: float = 0.01,
    dropout: Optional[float] = None,
    output_dir: str = "outputs/bt_transformers",
    log_history_path: Optional[str] = None,
    seed: Optional[int] = None,
):
    if seed is not None:
        set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    if dropout is not None:
        config.hidden_dropout_prob = dropout
        config.attention_probs_dropout_prob = dropout

    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

    train_ds = TorchDataset(X_train, y_train, tokenizer, max_length=max_length)
    eval_ds = TorchDataset(X_test, y_test, tokenizer, max_length=max_length)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    base_args = dict(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=weight_decay,
        logging_steps=50,
        report_to="none",
    )
    if seed is not None:
        base_args["seed"] = seed
        base_args["data_seed"] = seed
    training_args = _build_training_args(base_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=_compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    if log_history_path:
        pd.DataFrame(trainer.state.log_history).to_csv(log_history_path, index=False)
    print(f"[transformers] Eval metrics: {metrics}")
    return metrics


def _build_training_args(base_args):
    try:
        return TrainingArguments(
            **base_args,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            save_total_limit=1,
        )
    except TypeError:
        pass

    try:
        return TrainingArguments(
            **base_args,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            save_total_limit=1,
        )
    except TypeError:
        pass

    legacy_args = dict(base_args)
    legacy_args.pop("report_to", None)
    try:
        return TrainingArguments(
            **legacy_args,
            evaluate_during_training=True,
        )
    except TypeError:
        return TrainingArguments(**legacy_args)


def berts_orchestrated_tune(
    target_var,
    data_path,
    epochs,
    lr,
    weight_decay,
    dropout,
    model_name: Optional[str] = None,
    max_length: int = 256,
    batch_size: int = 8,
    test_size: float = 0.2,
    random_state: int = 42,
    log_history_path: Optional[str] = None,
):
    if random_state is not None:
        set_seed(random_state)
    data = load_processing(data_path)
    if target_var not in data.columns:
        raise ValueError(f"Target column '{target_var}' not found in data")

    X_train, X_test, y_train, y_test = train_test_split(
        data["description"],
        data[target_var],
        test_size=test_size,
        random_state=random_state,
    )
    model_name = model_name or DEFAULT_MODEL_TUNE
    print(f"[transformers] Fine-tuning model: {model_name}")
    return train_transformer_model(
        X_train,
        y_train,
        X_test,
        y_test,
        model_name=model_name,
        num_labels=2,
        max_length=max_length,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        weight_decay=weight_decay,
        dropout=dropout,
        log_history_path=log_history_path,
        seed=random_state,
    )


def berts_orchestrated_zero(
    target_var,
    data_path,
    model_name: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    max_samples: Optional[int] = None,
    batch_size: int = 8,
):
    if random_state is not None:
        set_seed(random_state)
    data = load_processing(data_path)
    if target_var not in data.columns:
        raise ValueError(f"Target column '{target_var}' not found in data")

    X_train, X_test, y_train, y_test = train_test_split(
        data["description"],
        data[target_var],
        test_size=test_size,
        random_state=random_state,
    )

    model_name = model_name or DEFAULT_MODEL_ZERO
    if target_var == "political":
        labels = ["political", "not political"]
        positive_label = "political"
    elif target_var == "domestic":
        labels = ["domestic", "international"]
        positive_label = "domestic"
    else:
        raise ValueError("Incorrect target variable. Only 'domestic' or 'political' are allowed")

    texts = list(X_test)
    y_true = list(y_test)
    if max_samples is not None:
        texts = texts[:max_samples]
        y_true = y_true[:max_samples]

    clf = pipeline("zero-shot-classification", model=model_name)
    outputs = clf(texts, candidate_labels=labels, batch_size=batch_size)
    preds = [1 if out["labels"][0] == positive_label else 0 for out in outputs]

    f1 = f1_score(y_true, preds, pos_label=1)
    print(f"[transformers] Zero-shot model: {model_name}")
    print(f"[transformers] Samples: {len(y_true)}")
    print(f"[transformers] F1 (pos_label=1): {f1:.4f}")
    return f1
