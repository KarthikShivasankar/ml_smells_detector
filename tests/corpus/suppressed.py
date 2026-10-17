"""Findings silenced with mlscent comments."""
# mlscent: ignore-file[ML.PANDAS.COLUMN_SELECTION_CHECKER]
import pandas as pd

frame = pd.read_csv("input.csv")
frame.fillna(0, inplace=True)  # mlscent: ignore[ML.PANDAS.INPLACE_CHECKER]
frame.dropna(inplace=True)  # mlscent: ignore
frame["a"][0] = 1  # mlscent: ignore[ML.PANDAS.INPLACE_CHECKER]  # expect: ML.PANDAS.CHAIN_INDEXING
matrix = frame.values
