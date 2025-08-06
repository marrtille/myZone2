# shap_analysis.py

import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Загружаем модель
model = joblib.load("breast_risk_model.pkl")

# Пример — заполни своими значениями
sample = pd.DataFrame([{
    "age": 50,
    "sex": 0,
    "bmi": 27.3,
    "family_history": 1,
    "smoking": 1,
    "alcohol": 1,
    "early_periods": 1,
    "late_menopause": 0,
    "ovarian_cancer_history": 0,
    "low_activity": 1,
    "hormone_therapy": 0,
    "brca_mutation": 0,
    "no_pregnancy_over_40": 0
}])

# SHAP анализ
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample)

# Визуализация
shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
plt.savefig("static/shap_feature_importance.png", bbox_inches="tight")
print("✅ SHAP график сохранён в static/shap_feature_importance.png")
