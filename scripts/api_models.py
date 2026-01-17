# for api models
# Planned: ChatGPT, Anthropic, Grok/Gemini
# I need to add the function that would take the proper 


import os
from dotenv import load_dotenv
from pathlib import Path
import json
import time

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


def send_requests(target_var, data_path=None, seed=42, max_samples=None):
    global bg, PROMPT
    random.seed(seed)
    if data_path is None:
        data_path = REPO_ROOT / "data" / "belgium_newspaper_new_filter.csv"
    bg = load_processing(data_path)
    out_path = f"predictions_{target_var}_{model}.csv".replace("/", "_")
    if max_samples is not None:
        out_path = out_path.replace(".csv", f"_n{int(max_samples)}.csv")
    if os.path.exists(out_path):
        pred_df = pd.read_csv(out_path)
        f1 = f1_score(pred_df["y_true"], pred_df["y_pred"], pos_label=1)
        print(f"[cache] Loaded predictions from {out_path}")
        print(f"F1 (pos_label=1): {f1:.4f}")
        return f1

    X_train, X_test, y_train, y_test = split_data(target_var, seed=seed)
    if max_samples is not None:
        X_test = X_test.iloc[:max_samples]
        y_test = y_test.iloc[:max_samples]
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
        PROMPT = """You are a political expert that knows all languages in the world. you are given articles \
        from dutch newspaper with ranging from 1999 to 2008. you need to critically assess whether this\
        article's topic is about domestic politics issue or international. \
        if it is about domestic issue, code it as 1, otherwise code it as 0.
        Return JSON exactly: {{"domestic": 0 or 1}}

        Text:
        {txt}"""
    else:
        raise ValueError("Incorrect target variable. Only 'domestic' or 'political' are allowed")

    if PROMPT.strip() == "":
        raise ValueError("No prompt was selected, but the variable is set correctly. For some reason...")

    y_pred = [predict_one(txt, json_key=json_key) for txt in X_test]

    f1 = f1_score(y_test, y_pred, pos_label=1)
    print(f"Target: {target_var}")
    print(f"Test size: {len(y_test)}")
    print(f"Positive rate: {sum(y_test)/len(y_test):.4f}")
    print(f"F1 (pos_label=1): {f1:.4f}")
   
    pred_df = pd.DataFrame({"y_true": list(y_test), "y_pred": y_pred})
    pred_df.to_csv(out_path, index=False)

    return f1 
