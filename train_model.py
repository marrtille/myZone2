# train_model.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Features you collect in Flask form
FEATURES = [
    'age', 'sex', 'bmi', 'family_history', 'smoking', 'alcohol',
    'early_periods', 'late_menopause', 'ovarian_cancer_history',
    'low_activity', 'hormone_therapy', 'brca_mutation', 'no_pregnancy_over_40'
]

# 1. Generate synthetic dataset
def generate_data(n=1000):
    np.random.seed(42)
    data = {
        'age': np.random.randint(25, 80, n),
        'sex': np.random.randint(0, 2, n),
        'bmi': np.round(np.random.uniform(18, 40, n), 1),
        'family_history': np.random.randint(0, 2, n),
        'smoking': np.random.randint(0, 2, n),
        'alcohol': np.random.randint(0, 2, n),
        'early_periods': np.random.randint(0, 2, n),
        'late_menopause': np.random.randint(0, 2, n),
        'ovarian_cancer_history': np.random.randint(0, 2, n),
        'low_activity': np.random.randint(0, 2, n),
        'hormone_therapy': np.random.randint(0, 2, n),
        'brca_mutation': np.random.randint(0, 2, n),
        'no_pregnancy_over_40': np.random.randint(0, 2, n),
    }
    df = pd.DataFrame(data)

    # Label: risk=1 if 3+ risk factors active
    df['risk'] = (
        df['family_history'] + df['brca_mutation'] +
        df['smoking'] + df['hormone_therapy'] +
        df['late_menopause'] + df['no_pregnancy_over_40']
    ) > 2
    df['risk'] = df['risk'].astype(int)
    return df

# 2. Train & save model
df = generate_data()
X = df[FEATURES]
y = df['risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("✅ Classification Report:")
print(classification_report(y_test, model.predict(X_test)))

joblib.dump(model, 'breast_risk_model.pkl')
print("🎯 Model saved to breast_risk_model.pkl")
