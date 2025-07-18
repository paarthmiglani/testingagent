# src/nlp/search.py

import pandas as pd


def keyword_search(query, dataset_path):
    # Try UTF-8 first, fall back to latin1 (ISO-8859-1) if error
    try:
        df = pd.read_csv(dataset_path, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(dataset_path, encoding="latin1")
        except Exception as e:
            print(f"Failed to read {dataset_path}: {e}")
            return pd.DataFrame()  # Return empty if all fail
    # Now do search
    results = df[df.apply(lambda row: query.lower() in str(row).lower(), axis=1)]
    return results
