"""
data_lookup.py
Direct pandas queries for numeric ranking questions.
Used alongside RAG for questions like "lowest scores", "highest rates" etc.
"""

import pandas as pd

df = pd.read_csv("research_data_clean.csv")

def get_lowest_reading(n=10):
    result = (
        df[["School Name", "Board Name", "Municipality",
            "Grade6Reading", "LowIncome_pct", "NoDegree_pct",
            "Grade6Reading_3yr_change"]]
        .dropna(subset=["Grade6Reading"])
        .drop_duplicates(subset=["School Name", "Board Name"])
        .sort_values("Grade6Reading")
        .head(n)
    )
    return result.to_string(index=False)

def get_highest_lowincome(n=10):
    result = (
        df[["School Name", "Board Name", "Municipality",
            "Grade6Reading", "LowIncome_pct"]]
        .dropna(subset=["LowIncome_pct"])
        .drop_duplicates(subset=["School Name", "Board Name"])
        .sort_values("LowIncome_pct", ascending=False)
        .head(n)
    )
    return result.to_string(index=False)

def get_equity_risk_schools(n=15):
    result = (
        df[df["EquityRisk"] == True][
            ["School Name", "Board Name", "Municipality",
             "Grade6Reading", "LowIncome_pct", "NoDegree_pct"]
        ]
        .drop_duplicates(subset=["School Name", "Board Name"])
        .sort_values("Grade6Reading")
        .head(n)
    )
    return result.to_string(index=False)

if __name__ == "__main__":
    print("=== LOWEST READING SCORES ===")
    print(get_lowest_reading())
    print()
    print("=== EQUITY RISK SCHOOLS ===")
    print(get_equity_risk_schools())