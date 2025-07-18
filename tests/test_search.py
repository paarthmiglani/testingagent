import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
from src.nlp.search import keyword_search


def test_keyword_search_returns_dataframe(capsys):
    dataset = "data/nlpdata/captions.csv"
    result = keyword_search("temple", dataset)
    captured = capsys.readouterr()
    assert isinstance(result, pd.DataFrame)
    assert captured.out == ""
