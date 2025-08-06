from flask import Flask, render_template, request
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

app = Flask(__name__)
model = joblib.load("breast_risk_model.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/risk_form", methods=["GET", "POST"])
def risk_form():
    if request.method == "POST":
        def get_int(name): return int(request.form.get(name, 0))
        def get_float(name): return float(request.form.get(name, 0.0))

        age = get_int("age")
        sex = get_int("sex")
        bmi = get_float("bmi")
        family_history = get_int("family_history")
        smoking = get_int("smoking")
        alcohol = get_int("alcohol")
        early_periods = get_int("early_periods")
        late_menopause = get_int("late_menopause")
        ovarian_cancer_history = get_int("ovarian_history")
        low_activity = get_int("low_activity")
        hormone_therapy = get_int("hormone_therapy")
        brca_mutation = get_int("brca_mutation")
        no_pregnancy_over_40 = get_int("never_pregnant_40")

        input_data = [[
            age, sex, bmi, family_history, smoking, alcohol,
            early_periods, late_menopause, ovarian_cancer_history,
            low_activity, hormone_therapy, brca_mutation, no_pregnancy_over_40
        ]]
        cols = [
            "age", "sex", "bmi", "family_history", "smoking", "alcohol",
            "early_periods", "late_menopause", "ovarian_cancer_history",
            "low_activity", "hormone_therapy", "brca_mutation", "no_pregnancy_over_40"
        ]
        df = pd.DataFrame(input_data, columns=cols)

        prediction = model.predict(df)[0]

        # SHAP plot
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df)
        shap.summary_plot(shap_values, df, plot_type="bar", show=False)
        plt.savefig("static/shap_feature_importance.png", bbox_inches="tight")
        plt.close()

        # Output
        title, advice = {
            0: ("Низкий риск", "🟢 Риск низкий. Продолжайте профилактику."),
            1: ("Средний риск", "🟡 Рекомендуем проконсультироваться."),
            2: ("Высокий риск", "🔴 Срочно обратитесь к врачу.")
        }.get(prediction, ("Неопределено", "Проверьте данные."))

        return render_template("result.html", title=title, advice=advice, shap_img=True)

    return render_template("risk_form.html")

@app.route("/result")
def result():
    # fallback route — used only if risk passed in URL (old version)
    risk = request.args.get("risk")
    if risk is None:
        return render_template("result.html", title="Результат недоступен", advice="Пожалуйста, пройдите тест сначала.")
    risk = int(risk)
    title, advice = {
        0: ("Низкий риск", "🟢 Риск низкий. Продолжайте профилактику."),
        1: ("Средний риск", "🟡 Рекомендуем проконсультироваться."),
        2: ("Высокий риск", "🔴 Срочно обратитесь к врачу.")
    }.get(risk, ("Неопределено", "Проверьте данные."))
    return render_template("result.html", title=title, advice=advice, shap_img=True)


@app.route("/self_exam")
def self_exam():
    return render_template("self_exam.html")

@app.route("/usg", methods=["GET", "POST"])
def usg():
    status = None
    advice = None
    if request.method == "POST":
        result = request.form.get("usg_result")
        if result == "tumour":
            status = "Опухоль обнаружена"
            advice = "🔴 Срочно обратитесь к врачу для дальнейшего обследования и лечения."
        elif result == "no_tumour":
            status = "Опухоль не обнаружена"
            advice = "🟢 Всё хорошо! Продолжайте регулярные обследования и заботьтесь о здоровье."
    return render_template("usg.html", status=status, advice=advice)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
