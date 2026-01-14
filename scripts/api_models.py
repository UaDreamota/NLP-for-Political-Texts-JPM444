# A place to look for api models
# Planned: ChatGPT, Anthropic, Grok/Gemini
# I need to add the function that would take the proper 


import os
import dotenv
from pathlib import Path

import random
import pandas as pd
from openai import OpenAI

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from utils.data_processing import load_processing

target_var = None #political or domestic


random.seed(42)

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

TOKEN = os.environ["OPENAI_API_KEY"]

def split_data(target):
    if target not in ['domestic','political']:
        raise ValueError("Incorrect target variable. Only 'domestic' or 'political' are allowed")
    global target_var
    target_var = target
    return X_train, X_test, y_train, y_test = train_test_split(bg['description'], bg[target], test_size=0.2, random_state=42)


def predict_one(txt)
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
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

client = OpenAI(api_key=TOKEN)
model = MODEL

SYSTEM = "Return only valid JSON. No extra text."

PROMPT = ""


def send_requests(target_var):
    bg = load_processing(../belgium_newspaper_new_filter.csv)
    X_train, X_test, y_train, y_test = split_data(target_var)
    if target_var == "political":
        PROMPT =
    elif target_var == "domestic":
        PROMPT = 
    else:
        raise ValueError("Incorrect target variable. Only 'domestic' or 'political' are allowed")
    return None


