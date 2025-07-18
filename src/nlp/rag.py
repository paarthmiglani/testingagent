# src/nlp/rag.py
import pandas as pd

def retrieve_passages(query, knowledge_files, top_k=3):
    matches = []
    for name, path in knowledge_files.items():
        if path.endswith(".csv"):
            df = pd.read_csv(path)
        elif path.endswith(".jsonl"):
            df = pd.read_json(path, lines=True)
        else:
            continue
        # Simple keyword match; can replace with semantic search!
        hits = df[df.apply(lambda row: query.lower() in str(row).lower(), axis=1)]
        if not hits.empty:
            for _, row in hits.iterrows():
                matches.append((name, dict(row)))
    return matches[:top_k]
