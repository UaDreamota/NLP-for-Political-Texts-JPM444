# for api models
# Planned: ChatGPT, Anthropic, Grok/Gemini
# I need to add the function that would take the proper 


import os
from dotenv import load_dotenv
from pathlib import Path
import json

import random
import pandas as pd
from openai import OpenAI

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from scripts.data_processing import load_processing

target_var = None #political or domestic
bg = None

random.seed(42)

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

MODEL = "gpt-5-mini"
model = MODEL

SYSTEM = "Return only valid JSON. No extra text."

PROMPT = ""



def split_data(target):
    if target not in ['domestic','political']:
        raise ValueError("Incorrect target variable. Only 'domestic' or 'political' are allowed")
    global target_var
    target_var = target
    X_train, X_test, y_train, y_test = train_test_split(bg['description'], bg[target], test_size=0.2, random_state=42)
    return  X_train, X_test, y_train, y_test

def _get_client():
    token = os.getenv("OPENAI_API_KEY")
    if not token:
        raise EnvironmentError("Missing OPENAI_API_KEY; set it in .env or your shell")
    return OpenAI(api_key=token)


def predict_one(txt, json_key):
        client = _get_client()
        resp = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            max_tokens = 20,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": PROMPT.format(txt=str(txt))},
            ],
        )
        data = json.loads(resp.choices[0].message.content)
        pred = int(data[json_key])
        if pred not in (0, 1):
            raise ValueError(f"Bad prediction value for {json_key}: {pred}")
        return pred


def send_requests(target_var, data_path=None):
    global bg, PROMPT
    if data_path is None:
        data_path = REPO_ROOT / "data" / "belgium_newspaper_new_filter.csv"
    bg = load_processing(data_path)
    out_path = f"predictions_{target_var}_{model}.csv".replace("/", "_")
    if os.path.exists(out_path):
        pred_df = pd.read_csv(out_path)
        f1 = f1_score(pred_df["y_true"], pred_df["y_pred"], pos_label=1)
        print(f"[cache] Loaded predictions from {out_path}")
        print(f"F1 (pos_label=1): {f1:.4f}")
        return f1

    X_train, X_test, y_train, y_test = split_data(target_var)
    if target_var == "political":
        json_key = "political"
        PROMPT = """You are a political expert that knows all languages in the world. you are given articles \
        from dutch newspaper with ranging from 1999 to 2008. you need to critically assess whether this\
        article's topic is politics. if so, code it as 1, otherwise code it as 0.
        Return JSON exactly: {"political": 0 or 1}

        Text:
        {txt}"""
    elif target_var == "domestic":
        json_key = "domestic"
        PROMPT = """You are a political expert that knows all languages in the world. you are given articles \
        from dutch newspaper with ranging from 1999 to 2008. you need to critically assess whether this\
        article's topic is about domestic politics issue or international. \
        if it is about domestic issue, code it as 1, otherwise code it as 0.
        Return JSON exactly: {"domestic": 0 or 1}

        Text:
        {txt}"""
    else:
        raise ValueError("Incorrect target variable. Only 'domestic' or 'political' are allowed")

    if PROMPT.strip() == "":
        raise ValueError("No prompt was selected, but the variable is set correctly")

    y_pred = [predict_one(txt, json_key=json_key) for txt in X_test]

    f1 = f1_score(y_test, y_pred, pos_label=1)
    print(f"Target: {target_var}")
    print(f"Test size: {len(y_test)}")
    print(f"Positive rate: {sum(y_test)/len(y_test):.4f}")
    print(f"F1 (pos_label=1): {f1:.4f}")
   
    pred_df = pd.DataFrame({"y_true": list(y_test), "y_pred": y_pred})
    pred_df.to_csv(out_path, index=False)

    return f1 
