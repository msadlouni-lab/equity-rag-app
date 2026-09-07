import pandas as pd

# Load the raw EQAO file — update the filename to match yours exactly
df = pd.read_csv("G3_2025_1.csv")

# Keep only the columns that matter for equity analysis
columns_to_keep = [
    "BoardName",
    "SchoolName",
    "Grade",
    "Language",
    "FundingType",
    "pctFullyParticipating_Read",
    "pctFullyParticipating_Write",
    "pctFullyParticipating_Math",
    "pctOverallR_L34",   # % at or above standard in Reading
    "pctOverallW_L34",   # % at or above standard in Writing
    "pctOverallM_L34",   # % at or above standard in Math
    "pctExempted_Read",
    "pctExempted_Write",
    "pctExempted_Math",
]

df_clean = df[columns_to_keep]

# Drop rows where school name is missing
df_clean = df_clean.dropna(subset=["SchoolName"])

# Save the clean version
df_clean.to_csv("eqao_clean.csv", index=False)

print(f"Done. Clean file has {len(df_clean)} rows and {len(df_clean.columns)} columns.")
print(df_clean.head())