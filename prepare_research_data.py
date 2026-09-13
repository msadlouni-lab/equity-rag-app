"""
prepare_research_data.py
Prepares Manal's actual research dataset (CW1_processed_analysis_data.csv)
for loading into the RAG vector store.
Run this ONCE to generate research_data_clean.csv
"""

import pandas as pd

df = pd.read_csv("CW1_processed_analysis_data.csv")

# Keep only the columns meaningful for conversational RAG queries
cols = [
    'School Name',
    'Board Name',
    'Municipality_clean',
    'School Level',
    'School Language',
    'Grade6Reading',
    'LowIncome_imputed',
    'NoDegree_imputed',
    'Density_2021',
    'Percentage of Grade 6 Students Achieving the Provincial Standard in Reading',
    'Percentage of Grade 6 Students Achieving the Provincial Standard in Writing',
    'Percentage of Grade 6 Students Achieving the Provincial Standard in Mathematics',
    'Percentage of Grade 3 Students Achieving the Provincial Standard in Reading',
    'Percentage of Grade 3 Students Achieving the Provincial Standard in Mathematics',
    'Change in Grade 6 Reading Achievement Over Three Years',
    'Change in Grade 6 Mathematics Achievement Over Three Years',
    'Percentage of Students Receiving Special Education Services',
    'Enrolment',
    'City',
    'Latitude',
    'Longitude'
]

df_clean = df[cols].copy()

# Rename for clarity
df_clean = df_clean.rename(columns={
    'Municipality_clean': 'Municipality',
    'LowIncome_imputed': 'LowIncome_pct',
    'NoDegree_imputed': 'NoDegree_pct',
    'Percentage of Grade 6 Students Achieving the Provincial Standard in Reading': 'Grade6Reading_pct',
    'Percentage of Grade 6 Students Achieving the Provincial Standard in Writing': 'Grade6Writing_pct',
    'Percentage of Grade 6 Students Achieving the Provincial Standard in Mathematics': 'Grade6Math_pct',
    'Percentage of Grade 3 Students Achieving the Provincial Standard in Reading': 'Grade3Reading_pct',
    'Percentage of Grade 3 Students Achieving the Provincial Standard in Mathematics': 'Grade3Math_pct',
    'Change in Grade 6 Reading Achievement Over Three Years': 'Grade6Reading_3yr_change',
    'Change in Grade 6 Mathematics Achievement Over Three Years': 'Grade6Math_3yr_change',
    'Percentage of Students Receiving Special Education Services': 'SpecialEd_pct',
})

# Only keep schools with Grade 6 data
df_clean = df_clean.dropna(subset=['Grade6Reading'])

# Add equity risk flag — reading below 60% = at risk (from your research findings)
df_clean['EquityRisk'] = df_clean['Grade6Reading'] < 60

df_clean.to_csv("research_data_clean.csv", index=False)

print(f"✅ Done. {len(df_clean)} schools with Grade 6 data.")
print(f"   Equity risk schools (reading < 60%): {df_clean['EquityRisk'].sum()}")
print(f"   Saved to research_data_clean.csv")
