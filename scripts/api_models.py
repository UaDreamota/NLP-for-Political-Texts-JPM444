# for api models
# Planned: ChatGPT, Anthropic, Grok/Gemini
# I need to add the function that would take the proper 


import os
from dotenv import load_dotenv
from pathlib import Path
import json
import time
from datetime import datetime

import random
import pandas as pd
from openai import APIConnectionError, APITimeoutError, BadRequestError, InternalServerError, OpenAI, RateLimitError

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from scripts.data_processing import load_processing

target_var = None #political or domestic
bg = None

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

MODEL = "gpt-5-mini"
model = MODEL

SYSTEM = "Return only valid JSON. No extra text."

PROMPT = ""

DEFAULT_MAX_COMPLETION_TOKENS = 512
MAX_COMPLETION_TOKENS_CAP = 2048

_token_param_name = None  # "max_completion_tokens" or "max_tokens"
_supports_temperature_zero = None
_supports_response_format = None
_client = None


def predictions_path(target, seed=42, max_samples=None, scope="test"):
    filename = f"predictions_{target}_{model}_seed{seed}.csv".replace("/", "_")
    if scope == "all":
        filename = filename.replace(".csv", "_all.csv")
    if max_samples is not None:
        filename = filename.replace(".csv", f"_n{int(max_samples)}.csv")
    return Path(filename)


def split_data(target, seed=42):
    if target not in ['domestic','political']:
        raise ValueError("Incorrect target variable. Only 'domestic' or 'political' are allowed")
    global target_var
    target_var = target
    X_train, X_test, y_train, y_test = train_test_split(
        bg['description'],
        bg[target],
        test_size=0.2,
        random_state=seed,
    )
    return  X_train, X_test, y_train, y_test

def _get_client():
    global _client
    if _client is None:
        token = os.getenv("OPENAI_API_KEY")
        if not token:
            raise EnvironmentError("Missing OPENAI_API_KEY; set it in .env or your shell")
        _client = OpenAI(api_key=token)
    return _client


def predict_one(txt, json_key):
        client = _get_client()

        def _extract_json_object(text: str) -> str | None:
            start = text.find("{")
            end = text.rfind("}")
            if 0 <= start < end:
                return text[start : end + 1]
            return None

        request_kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": PROMPT.format(txt=str(txt))},
            ],
        )
        global _token_param_name, _supports_temperature_zero, _supports_response_format
        max_tokens = DEFAULT_MAX_COMPLETION_TOKENS
        last_finish_reason = None
        last_content = None
        last_error = None
        for _attempt in range(6):
            token_param_name = _token_param_name or "max_completion_tokens"
            token_kwargs = {token_param_name: max_tokens}
            use_temperature_zero = _supports_temperature_zero is not False
            temperature_kwargs = {"temperature": 0} if use_temperature_zero else {}
            use_response_format = _supports_response_format is not False
            response_format_kwargs = {"response_format": {"type": "json_object"}} if use_response_format else {}
            try:
                resp = client.chat.completions.create(
                    **request_kwargs,
                    **response_format_kwargs,
                    **temperature_kwargs,
                    **token_kwargs,
                )
            except BadRequestError as exc:
                msg = str(exc)
                last_error = msg
                if token_param_name == "max_completion_tokens" and "Unsupported parameter: 'max_completion_tokens'" in msg:
                    _token_param_name = "max_tokens"
                    continue
                if use_temperature_zero and "Unsupported value: 'temperature' does not support 0" in msg:
                    _supports_temperature_zero = False
                    continue
                if use_response_format and "Unsupported parameter: 'response_format'" in msg:
                    _supports_response_format = False
                    continue
                if "Please try again with higher max_tokens" in msg:
                    max_tokens = min(MAX_COMPLETION_TOKENS_CAP, max_tokens * 2)
                    continue
                raise
            except (json.JSONDecodeError, APIConnectionError, APITimeoutError, RateLimitError, InternalServerError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(min(10, 2 ** _attempt))
                continue

            choice = resp.choices[0]
            content = choice.message.content or ""
            last_finish_reason = choice.finish_reason
            last_content = content
            if not content.strip() and choice.finish_reason == "length":
                max_tokens = min(MAX_COMPLETION_TOKENS_CAP, max_tokens * 2)
                continue
            if not content.strip():
                raise ValueError(f"Empty response content (finish_reason={choice.finish_reason!r})")

            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                extracted = _extract_json_object(content)
                if extracted:
                    try:
                        data = json.loads(extracted)
                    except json.JSONDecodeError:
                        max_tokens = min(MAX_COMPLETION_TOKENS_CAP, max_tokens * 2)
                        continue
                else:
                    max_tokens = min(MAX_COMPLETION_TOKENS_CAP, max_tokens * 2)
                    continue

            if json_key not in data:
                if len(data) == 1:
                    pred_raw = next(iter(data.values()))
                else:
                    raise KeyError(f"Missing JSON key {json_key!r} in response: {data}")
            else:
                pred_raw = data[json_key]

            pred = int(pred_raw)
            if pred not in (0, 1):
                raise ValueError(f"Bad prediction value for {json_key}: {pred}")
            return pred

        last_content_preview = None
        if last_content is not None:
            last_content_preview = last_content[:200] + ("..." if len(last_content) > 200 else "")
        raise RuntimeError(
            "Failed to get a valid JSON prediction after multiple attempts "
            f"(last_finish_reason={last_finish_reason!r}, last_error={last_error!r}, last_content={last_content_preview!r})"
        )


def send_requests(target_var, data_path=None, seed=42, max_samples=None, scope="test", log_every=50, flush_every=50):
    global bg, PROMPT
    random.seed(seed)
    if data_path is None:
        data_path = REPO_ROOT / "data" / "belgium_newspaper_new_filter.csv"
    bg = load_processing(data_path)

    out_path = predictions_path(target_var, seed=seed, max_samples=max_samples, scope=scope)

    dataset_n = len(bg)
    if scope == "all":
        X_run = bg["description"]
        y_run = bg[target_var]
        split_summary = f"all_rows={dataset_n}"
    elif scope == "test":
        X_train, X_run, y_train, y_run = split_data(target_var, seed=seed)
        split_summary = f"dataset={dataset_n} train={len(X_train)} test={len(X_run)} (test_size=0.2)"
    else:
        raise ValueError("scope must be 'test' or 'all'")

    if max_samples is not None:
        X_run = X_run.iloc[:max_samples]
        y_run = y_run.iloc[:max_samples]
    if target_var == "political":
        json_key = "political"
        PROMPT = """You are performing a binary topic classification task.

Goal:
Determine whether the main topic of the following newspaper article is POLITICS.

Definition:
Code the article as political (1) if its primary focus is on:
- Government, public policy, legislation, or regulation
- Political parties, elections, campaigns, or voting
- Politicians or public officials acting in an official or political role
- International relations, diplomacy, war, or geopolitical conflict
- Public administration, state institutions, or governance
- Political ideology, political movements, or political protests

Code the article as non-political (0) if:
- Politics is only mentioned incidentally or as background
- The focus is primarily cultural, economic, sports-related, criminal, scientific, or personal
- Politicians appear only in a non-political or personal context
- The article reports events without political decision-making, conflict, or policy relevance

Decision rule:
If politics is the central subject of the article, return 1.
If politics is secondary or absent, return 0.

Output format:
Return ONLY valid JSON, with no additional text:
{{"political": 0 or 1}}

Article text:
{txt}"""
    elif target_var == "domestic":
        json_key = "domestic"
        PROMPT = """You are performing a binary classification task.

Goal:
Determine whether the following newspaper article is primarily about DOMESTIC politics (code 1) or INTERNATIONAL politics (code 0).

Code the article as domestic (1) if its main focus is on:
- National or local government, policy, legislation, regulation, or public administration inside the country
- Domestic elections, parties, campaigns, coalitions, or politicians acting in a national/local role
- National security, public safety, justice, or law enforcement within the country
- Domestic social, economic, cultural, or environmental issues framed around national/local policy

Code the article as international (0) if:
- The primary focus is foreign governments, diplomacy, treaties, conflicts, or geopolitics
- Events abroad with no substantive domestic policy/governance angle
- International organizations, foreign elections, or foreign policy that does not center domestic implications

Decision rule:
If domestic politics/governance is the central subject, return 1; if the article is mainly about foreign/international politics, return 0.

Output format:
Return ONLY valid JSON, with no additional text:
{{"domestic": 0 or 1}}

Article text:
{txt}"""
    else:
        raise ValueError("Incorrect target variable. Only 'domestic' or 'political' are allowed")

    if PROMPT.strip() == "":
        raise ValueError("No prompt was selected, but the variable is set correctly. For some reason...")

    n_total = len(y_run)
    if n_total == 0:
        raise ValueError("No test samples were selected (n_total == 0).")

    print(f"[api] model={model} target={target_var} scope={scope} seed={seed} samples={n_total} out={out_path}")
    print(f"[api] {split_summary}")

    existing_rows = None
    if out_path.exists():
        try:
            existing_rows = pd.read_csv(out_path, on_bad_lines="skip")
        except Exception as exc:
            backup_path = out_path.with_suffix(out_path.suffix + f".corrupt_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            out_path.rename(backup_path)
            print(f"[api] Existing predictions file could not be read; moved to {backup_path} ({type(exc).__name__}: {exc})")
            existing_rows = None

    start_at = 0
    y_pred = []
    if existing_rows is not None and not existing_rows.empty:
        if "y_pred" in existing_rows.columns and "y_true" in existing_rows.columns:
            existing_rows = existing_rows.iloc[:n_total].copy()
            already_done = len(existing_rows)

            expected_y_true = list(map(int, y_run.iloc[:already_done].tolist()))
            existing_y_true = list(map(int, existing_rows["y_true"].tolist()))
            y_true_matches = expected_y_true == existing_y_true

            idx_matches = True
            if "row_index" in existing_rows.columns:
                expected_idx = list(map(int, X_run.index[:already_done].tolist()))
                existing_idx = list(map(int, existing_rows["row_index"].tolist()))
                idx_matches = expected_idx == existing_idx

            if not y_true_matches or not idx_matches:
                backup_path = out_path.with_suffix(out_path.suffix + f".mismatch_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                out_path.rename(backup_path)
                print(f"[api] Existing predictions did not match current split; moved to {backup_path} and restarting.")
            else:
                y_pred = list(map(int, existing_rows["y_pred"].tolist()))
                start_at = already_done
                if start_at >= n_total:
                    f1 = f1_score(y_run, y_pred[:n_total], pos_label=1)
                    print(f"[cache] Loaded predictions from {out_path}")
                    print(f"F1 (pos_label=1): {f1:.4f}")
                    return f1
                print(f"[api] Resuming: {start_at}/{n_total} done, {n_total - start_at} left")

    run_start = time.perf_counter()
    rows_to_write = []
    wrote_header = out_path.exists() and out_path.stat().st_size > 0

    for i in range(start_at, n_total):
        if log_every and (i == start_at or (i % log_every == 0)):
            elapsed = time.perf_counter() - run_start
            processed = i - start_at
            rate = processed / elapsed if elapsed > 0 and processed > 0 else None
            remaining = n_total - i
            eta = (remaining / rate) if rate else None
            rate_str = f"{rate:.3f}/s" if rate else "n/a"
            eta_str = f"{eta/60:.1f} min" if eta else "n/a"
            print(f"[api] progress {i}/{n_total} (left {remaining}) | elapsed {elapsed/60:.1f} min | rate {rate_str} | eta {eta_str}")

        row_index = int(X_run.index[i])
        txt = X_run.iloc[i]
        y_true = int(y_run.iloc[i])
        try:
            pred = predict_one(txt, json_key=json_key)
        except Exception as exc:
            print(f"[api] ERROR at {i}/{n_total} row_index={row_index}: {type(exc).__name__}: {exc}")
            raise

        y_pred.append(int(pred))
        rows_to_write.append({"row_index": row_index, "y_true": y_true, "y_pred": int(pred)})

        should_flush = flush_every and len(rows_to_write) >= flush_every
        is_last = i == n_total - 1
        if should_flush or is_last:
            df_out = pd.DataFrame(rows_to_write, columns=["row_index", "y_true", "y_pred"])
            df_out.to_csv(out_path, mode="a", header=not wrote_header, index=False)
            wrote_header = True
            rows_to_write.clear()

    f1 = f1_score(y_run, y_pred, pos_label=1)
    print(f"Target: {target_var}")
    print(f"Test size: {len(y_run)}")
    print(f"Positive rate: {sum(y_run)/len(y_run):.4f}")
    print(f"F1 (pos_label=1): {f1:.4f}")

    return f1 
