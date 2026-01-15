import pandas as pd

# For data processing


def _clean_label_column(data: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in data.columns:
        raise KeyError(f"Missing expected column: {column}")
    data = data.copy()
    data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data[data[column].notna()]
    data = data[data[column] != 99]
    data[column] = data[column].astype(int)
    return data


def load_processing(csv_file):
    data = pd.read_csv(csv_file)
    data = _clean_label_column(data, "political")
    data = _clean_label_column(data, "domestic")

    if "description" not in data.columns:
        raise KeyError("Missing expected column: description")
    data["description"] = data["description"].fillna("")

    return data
