feature_namesfrom flask import Flask, render_template, request, redirect, url_for, make_response, send_file, session
import os, uuid, sqlite3
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

APP_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(APP_DIR, "static")
TEMPLATES_DIR = os.path.join(APP_DIR, "templates")
DB_PATH = os.path.join(APP_DIR, "diary.db")
os.makedirs(STATIC_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-me")

MODEL_PATH = os.path.join(APP_DIR, "breast_risk_model.pkl")
model = joblib.load(MODEL_PATH)

def get_or_create_user_id():
    if "user_id" not in session:
        session["user_id"] = str(uuid.uuid4())
    return session["user_id"]

def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS entries(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            created_at TEXT,
            breast_side TEXT,
            area TEXT,
            pain INTEGER,
            lump INTEGER,
            discharge INTEGER,
            notes TEXT
        )
    """)
    con.commit(); con.close()
init_db()

MODIFIABLE = {"smoking","alcohol","low_activity","bmi","hormone_therapy"}
HUMAN_NAMES = {
    "smoking":"курение","alcohol":"регулярный алкоголь","low_activity":"низкая физическая активность",
    "bmi":"повышенный индекс массы тела (BMI)","hormone_therapy":"гормональная терапия",
    "family_history":"семейная история рака груди","early_periods":"раннее начало менструаций",
    "late_menopause":"поздняя менопауза","ovarian_cancer_history":"личная/семейная история рака яичников",
    "brca_mutation":"мутация BRCA","no_pregnancy_over_40":"отсутствие беременности до 40",
    "age":"возраст","sex":"пол",
}

@app.route("/")
def index():
    return render_template("index.html") if os.path.exists(os.path.join(TEMPLATES_DIR,"index.html")) else """
    <h1>myZone</h1>
    <p><a href="/risk_form">Оценка риска</a> • <a href="/diary">Дневник</a> • <a href="/self_exam">Памятка</a></p>
    """

@app.route("/risk_form", methods=["GET","POST"])
def risk_form():
    if request.method == "GET":
        return render_template("risk_form.html") if os.path.exists(os.path.join(TEMPLATES_DIR,"risk_form.html")) else """
        <form method="post">
          age <input name="age" type="number" value="40"><br>
          sex (0=female,1=male) <input name="sex" type="number" value="0"><br>
          bmi <input name="bmi" type="number" value="25" step="0.1"><br>
          family_history (0/1) <input name="family_history" type="number" value="0"><br>
          smoking (0/1) <input name="smoking" type="number" value="0"><br>
          alcohol (0/1) <input name="alcohol" type="number" value="0"><br>
          early_periods (0/1) <input name="early_periods" type="number" value="0"><br>
          late_menopause (0/1) <input name="late_menopause" type="number" value="0"><br>
          ovarian_history (0/1) <input name="ovarian_history" type="number" value="0"><br>
          low_activity (0/1) <input name="low_activity" type="number" value="0"><br>
          hormone_therapy (0/1) <input name="hormone_therapy" type="number" value="0"><br>
          brca_mutation (0/1) <input name="brca_mutation" type="number" value="0"><br>
          never_pregnant_40 (0/1) <input name="never_pregnant_40" type="number" value="0"><br>
          <button type="submit">Predict</button>
        </form>
        """

    f = request.form
    def geti(n, default=0):
        try: return int(f.get(n, default))
        except: return int(float(f.get(n, default)))
    def getf(n, default=0.0):
        try: return float(f.get(n, default))
        except: return 0.0

    X = pd.DataFrame([{
        "age": geti("age"), "sex": geti("sex"), "bmi": getf("bmi"),
        "family_history": geti("family_history"), "smoking": geti("smoking"),
        "alcohol": geti("alcohol"), "early_periods": geti("early_periods"),
        "late_menopause": geti("late_menopause"), "ovarian_cancer_history": geti("ovarian_history"),
        "low_activity": geti("low_activity"), "hormone_therapy": geti("hormone_therapy"),
        "brca_mutation": geti("brca_mutation"), "no_pregnancy_over_40": geti("never_pregnant_40"),
    }])

    # robust risk mapping
    risk_label = "Средний риск"
    try:
        proba = model.predict_proba(X)
        if proba.shape[1] == 3:
            cls = int(np.argmax(proba[0]))
            risk_label = ["Низкий риск","Средний риск","Высокий риск"][cls]
        else:
            p1 = float(proba[0,1])
            if p1 < 0.33: risk_label = "Низкий риск"
            elif p1 < 0.66: risk_label = "Средний риск"
            else: risk_label = "Высокий риск"
    except Exception:
        pred = int(model.predict(X)[0])
        risk_label = {0:"Низкий риск",1:"Средний риск",2:"Высокий риск"}.get(pred,"Средний риск")

  # --- explainability (safe & 1-D)
feature_names = list(X.columns)
contrib = None

if SHAP_AVAILABLE:
    try:
        explainer = shap.Explainer(model, X)
        sv = explainer(X)  # sv.values could be (1, n_features) or (1, n_features, n_classes)
        raw = getattr(sv, "values", None)
        if raw is not None:
            arr = np.asarray(raw)
            # take instance 0 if present
            if arr.ndim >= 2 and arr.shape[0] == 1:
                arr = arr[0]
            # reduce to 1-D (n_features)
            if arr.ndim >= 2:
                # common: (n_features, n_classes) -> mean across classes
                arr = arr.mean(axis=-1)
            contrib = arr
    except Exception:
        contrib = None

if contrib is None:
    fi = getattr(model, "feature_importances_", None)
    if fi is not None:
        contrib = np.asarray(fi, dtype=float)
    else:
        contrib = np.zeros(len(feature_names), dtype=float)

# Final safety: ensure 1-D length == n_features
contrib = np.asarray(contrib, dtype=float)
if contrib.ndim >= 2:
    contrib = contrib.mean(axis=-1)
contrib = contrib.reshape(-1)
if contrib.shape[0] != len(feature_names):
    contrib = np.resize(contrib, (len(feature_names),))

order = np.argsort(np.abs(contrib))[::-1]

MODIFIABLE = {"smoking","alcohol","low_activity","bmi","hormone_therapy"}
HUMAN_NAMES = {
    "smoking":"курение","alcohol":"регулярный алкоголь","low_activity":"низкая физическая активность",
    "bmi":"повышенный индекс массы тела (BMI)","hormone_therapy":"гормональная терапия",
    "family_history":"семейная история рака груди","early_periods":"раннее начало менструаций",
    "late_menopause":"поздняя менопауза","ovarian_cancer_history":"личная/семейная история рака яичников",
    "brca_mutation":"мутация BRCA","no_pregnancy_over_40":"отсутствие беременности до 40",
    "age":"возраст","sex":"пол",
}

top3 = []
for idx in order[:3]:
    idx = int(idx)  # ensure scalar
    name = feature_names[idx]
    human = HUMAN_NAMES.get(name, name)
    mod = " (можно изменить)" if name in MODIFIABLE else " (неизменяемый фактор)"
    top3.append(f"• {human}{mod}")
shap_reason = "Этот балл повышается из-за:\n" + "\n".join(top3) if top3 else None



    try:
        top_k = min(8, len(feature_names))
        sel = order[:top_k]
        plt.figure(figsize=(5, 3.5), dpi=160)
        plt.barh([feature_names[i] for i in sel][::-1], np.abs(contrib[sel])[::-1])
        plt.xlabel("вклад признака (чем больше — тем важнее)")
        plt.tight_layout()
        feat_img = os.path.join(STATIC_DIR, "feature_importance.png")
        plt.savefig(feat_img, bbox_inches="tight"); plt.close()
        shap_img_url = url_for("static", filename="feature_importance.png")
    except Exception:
        shap_img_url = None

    session["last_input"] = X.to_dict(orient="records")[0]
    session["last_risk_label"] = risk_label
    session["last_shap_top"] = top3

    return render_template(
        "result.html",
        title=risk_label,
        advice=("🔴 Срочно обратитесь к врачу." if "Высокий" in risk_label
                else "🟡 Рекомендуем проконсультироваться." if "Средний" in risk_label
                else "🟢 Риск низкий. Продолжайте профилактику."),
        shap_reason=shap_reason,
        shap_img_url=shap_img_url,
        export_pdf_url=url_for("export_pdf"),
        reminder_ics_url=url_for("reminder_ics"),
        diary_url=url_for("diary"),
        self_exam_url=url_for("self_exam"),
    )

@app.route("/self_exam")
def self_exam():
    return render_template("self_exam.html") if os.path.exists(os.path.join(TEMPLATES_DIR,"self_exam.html")) else """
    <h2>Самообследование</h2>
    <img src="/static/laying.png" alt="Положение лёжа">
    <p>Лёжа, рука за головой. Круговыми движениями прощупайте грудь по квадрантам.</p>
    """

@app.route("/diary", methods=["GET","POST"])
def diary():
    uid = get_or_create_user_id()
    if request.method == "POST":
        data = {
            "breast_side": request.form.get("breast_side","left"),
            "area": request.form.get("area","outer-upper"),
            "pain": int(request.form.get("pain",0)),
            "lump": int(request.form.get("lump",0)),
            "discharge": int(request.form.get("discharge",0)),
            "notes": request.form.get("notes","").strip()
        }
        con = sqlite3.connect(DB_PATH); cur = con.cursor()
        cur.execute("""INSERT INTO entries(user_id, created_at, breast_side, area, pain, lump, discharge, notes)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (uid, datetime.utcnow().isoformat(), data["breast_side"], data["area"],
             data["pain"], data["lump"], data["discharge"], data["notes"]))
        con.commit(); con.close()
        return redirect(url_for("diary"))

    con = sqlite3.connect(DB_PATH); con.row_factory = sqlite3.Row
    cur = con.cursor(); cur.execute("SELECT * FROM entries WHERE user_id=? ORDER BY created_at DESC", (uid,))
    rows = cur.fetchall(); con.close()
    return render_template("diary.html", entries=[dict(r) for r in rows])

@app.route("/diary/export_json")
def diary_export_json():
    uid = get_or_create_user_id()
    con = sqlite3.connect(DB_PATH); con.row_factory = sqlite3.Row
    cur = con.cursor(); cur.execute("SELECT * FROM entries WHERE user_id=? ORDER BY created_at DESC", (uid,))
    rows = cur.fetchall(); con.close()
    df = pd.DataFrame([dict(r) for r in rows])
    resp = make_response(df.to_json(orient="records", force_ascii=False))
    resp.headers["Content-Type"] = "application/json; charset=utf-8"
    resp.headers["Content-Disposition"] = "attachment; filename=diary_entries.json"
    return resp

@app.route("/export_pdf")
def export_pdf():
    inputs = session.get("last_input")
    risk_label = session.get("last_risk_label")
    shap_top = session.get("last_shap_top", [])
    if not inputs or not risk_label:
        return redirect(url_for("risk_form"))

    pdf_path = os.path.join(STATIC_DIR, "myzone_report.pdf")
    c = canvas.Canvas(pdf_path, pagesize=A4)
    W,H = A4
    y = H-2*cm
    c.setFont("Helvetica-Bold", 16); c.drawString(2*cm, y, "myZone — Отчёт по оценке риска")
    y -= 1.0*cm
    c.setFont("Helvetica", 12); c.drawString(2*cm, y, f"Итоговый риск: {risk_label}")
    y -= 0.8*cm

    c.setFont("Helvetica-Bold", 12); c.drawString(2*cm, y, "Входные данные:"); y -= 0.6*cm
    c.setFont("Helvetica", 10)
    for k,v in inputs.items():
        c.drawString(2.2*cm, y, f"{k}: {v}")
        y -= 0.5*cm
        if y < 3*cm:
            c.showPage(); y = H-2*cm; c.setFont("Helvetica",10)

    feat_img = os.path.join(STATIC_DIR, "feature_importance.png")
    if os.path.exists(feat_img):
        c.showPage()
        c.drawImage(feat_img, 2*cm, H/2-2*cm, width=W-4*cm, preserveAspectRatio=True, mask='auto')

    c.showPage()
    c.setFont("Helvetica-Bold", 12); c.drawString(2*cm, H-2*cm, "Почему такой балл:")
    y = H-3*cm; c.setFont("Helvetica", 10)
    for line in shap_top:
        c.drawString(2.2*cm, y, line); y -= 0.5*cm
    c.save()

    return send_file(pdf_path, as_attachment=True, download_name="myzone_report.pdf")

@app.route("/reminder_ics")
def reminder_ics():
    risk_label = session.get("last_risk_label", "Низкий риск")
    if "Высокий" in risk_label:
        title, days = "myZone: запишитесь к врачу", 3
    elif "Средний" in risk_label:
        title, days = "myZone: консультация у врача", 7
    else:
        title, days = "myZone: повторная самооценка", 60
    start = (datetime.utcnow() + timedelta(days=days)).strftime("%Y%m%dT%H%M%SZ")
    ics = f"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//myZone//RU
BEGIN:VEVENT
DTSTART:{start}
DURATION:PT30M
SUMMARY:{title}
DESCRIPTION:Создано в myZone.
END:VEVENT
END:VCALENDAR
"""
    resp = make_response(ics)
    resp.headers["Content-Type"] = "text/calendar; charset=utf-8"
    resp.headers["Content-Disposition"] = "attachment; filename=myzone_reminder.ics"
    return resp

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
PY



