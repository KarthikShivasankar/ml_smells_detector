"""Sales report built with pandas: several classic pandas smells."""
import heapq

import pandas as pd

sales = pd.read_csv("sales.csv")  # expect: ML.PANDAS.COLUMN_SELECTION_CHECKER, ML.PANDAS.DATATYPE_CHECKER
regions = pd.read_csv("regions.csv", dtype={"region": "category"}, usecols=["region", "manager"])

sales.dropna(inplace=True)  # expect: ML.PANDAS.INPLACE_CHECKER
cleaned = sales.drop_duplicates(inplace=True)  # expect: ML.PANDAS.INPLACE_CHECKER

joined = sales.merge(regions)  # expect: ML.PANDAS.MERGE_PARAMETER_CHECKER
explicit = pd.merge(sales, regions, on="region", how="left", validate="many_to_one")
merged_streams = list(heapq.merge([1, 3], [2, 4]))

sales["amount"][0] = 0  # expect: ML.PANDAS.CHAIN_INDEXING
sales[sales["amount"] < 0]["amount"] = 0  # expect: ML.PANDAS.CHAIN_INDEXING
grid = [[0] * 3 for _ in range(3)]  # expect: ML.MAGIC_NUMBER
grid[1][2] = 5
first = sales["amount"][0]  # expect: ML.PANDAS.CHAIN_INDEXING

for idx, row in sales.iterrows():
    sales.loc[idx, "net"] = row["amount"] * 0.8  # expect: ML.MAGIC_NUMBER, ML.PANDAS.DATAFRAME_ITERATION_MODIFICATION
    row["amount"] = 0  # expect: ML.PANDAS.DATAFRAME_ITERATION_MODIFICATION

for col in sales.columns:
    sales[col] = pd.to_numeric(sales[col], errors="coerce")

matrix = sales.values
totals = sales.groupby("region").sum()

# File-level findings (reported at line 0):
# expect-file: ML.NO_ERROR_HANDLING
