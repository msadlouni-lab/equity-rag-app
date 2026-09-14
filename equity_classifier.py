"""
equity_classifier.py
Predicts whether an Ontario school is at equity risk
based on socioeconomic features from EQAO + Census data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix,
                           roc_auc_score, ConfusionMatrixDisplay)

# ── Load data ────────────────────────────────────────────────────────────────
print("📂 Loading research data...")
df = pd.read_csv("research_data_clean.csv")

# ── Clean percentage columns ──────────────────────────────────────────────────
pct_cols = ['Grade6Reading_pct', 'Grade6Writing_pct',
            'Grade6Math_pct', 'Grade3Reading_pct', 'Grade3Math_pct']

def clean_pct(val):
    """Convert EQAO percentage strings to float, handling all suppression codes."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip().replace('%', '')
    if s in ('N/R', 'N/A', 'N/D', 'SP', '', 'nan', 'None', '-'):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan

for col in pct_cols:
    if col in df.columns:
        df[col] = df[col].apply(clean_pct)
# ── Features and target ───────────────────────────────────────────────────────
features = [
  'LowIncome_pct', #  key predictor from OLS research
  'NoDegree_pct', # second strongest predictor
  'Density_2021', # population density
  'Grade3Reading_pct', # early indicator
  'Grade3Math_pct', # early indicator
  'SpecialEd_pct', # resource intensity signal
]

# Target: equity risk = reading score below 60%
# (consistent with your research definition)
df['EquityRisk'] = (df['Grade6Reading'] < 60).astype(int)

# Drop rows with missing features or target
df_model = df[features + ['EquityRisk', 'School Name', 'Board Name']].dropna()
print(f"✅ Modelling dataset: {len(df_model)} schools")
print(f" Equity risk schools: {df_model['EquityRisk'].sum()} ({df_model['EquityRisk'].mean()*100:.1f}%)")

X = df_model[features]
y = df_model['EquityRisk']

# ── Train/test split ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Model 1: Logistic Regression (baseline, explainable) ─────────────────────
print("\n📊 Training Logistic Regression (baseline)...")
lr_pipe = Pipeline([
  ('scaler', StandardScaler()),
  ('model', LogisticRegression(class_weight='balanced', max_iter=1000))
])
lr_pipe.fit(X_train, y_train)
lr_scores = cross_val_score(lr_pipe, X, y, cv=5, scoring='f1')
print(f" CV F1: {lr_scores.mean():.3f} ± {lr_scores.std():.3f}")

# ── Model 2: Random Forest (stronger, feature importance) ────────────────────
print("🌲 Training Random Forest...")
rf_pipe = Pipeline([
  ('scaler', StandardScaler()),
  ('model', RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42
  ))
])
rf_pipe.fit(X_train, y_train)
rf_scores = cross_val_score(rf_pipe, X, y, cv=5, scoring='f1')
print(f" CV F1: {rf_scores.mean():.3f} ± {rf_scores.std():.3f}")

# ── Evaluation ────────────────────────────────────────────────────────────────
# Note: CV F1 variance is high due to small minority class (6.8%)
# Test set ROC-AUC (0.959) is the more reliable performance metric here
print("\n📈 Random Forest — Test Set Results:")
y_pred = rf_pipe.predict(X_test)
y_prob = rf_pipe.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred,
      target_names=['Not at risk', 'Equity risk']))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")

# ── Feature importance ────────────────────────────────────────────────────────
print("\n🔍 Feature Importance:")
importances = rf_pipe.named_steps['model'].feature_importances_
feat_df = pd.DataFrame({
  'feature': features,
  'importance': importances
}).sort_values('importance', ascending=False)
print(feat_df.to_string(index=False))

# ── Save feature importance chart ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(feat_df['feature'], feat_df['importance'], color='#2d7a4f')
ax.set_xlabel('Feature Importance')
ax.set_title('Equity Risk Classifier — Feature Importance\n(Random Forest, n=200)')
plt.tight_layout()
plt.savefig('screenshots/feature_importance.png', dpi=150, bbox_inches='tight')
print("\n✅ Feature importance chart saved to screenshots/feature_importance.png")

# ── High-risk school predictions ──────────────────────────────────────────────
print("\n🚨 Schools predicted at highest equity risk:")
df_model['risk_probability'] = rf_pipe.predict_proba(X)[:, 1]
high_risk = (
    df_model.drop_duplicates(subset=['School Name', 'Board Name'])
    .nlargest(10, 'risk_probability')
  [['School Name', 'Board Name', 'LowIncome_pct',
    'NoDegree_pct', 'risk_probability']]
)
print(high_risk.to_string(index=False))

print("\n✅ Classifier complete. Add equity_classifier.py to GitHub.")