from flask import Flask, render_template, request, redirect, url_for, make_response, send_file, session
import os, uuid, sqlite3
import numpy as np
import pandas as pd
import shap
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

APP_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(APP_DIR, "templates")
DB_PATH = os.path.join(APP_DIR, "diary.db")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
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

@app.route("/risk_form", methods=["GET", "POST"])
def risk_form():
    if request.method == "GET":
        return render_template("risk_form.html")

    try:
        f = request.form

        def geti(name, default=0):
            v = f.get(name, default)
            try:
                return int(v)
            except Exception:
                try:
                    return int(float(v))
                except Exception:
                    return int(default)

        def getf(name, default=0.0):
            v = f.get(name, default)
            try:
                return float(v)
            except Exception:
                return float(default)

        X = pd.DataFrame([{
            "age": geti("age"),
            "sex": geti("sex"),
            "bmi": getf("bmi"),
            "family_history": geti("family_history"),
            "smoking": geti("smoking"),
            "alcohol": geti("alcohol"),
            "early_periods": geti("early_periods"),
            "late_menopause": geti("late_menopause"),
            "ovarian_cancer_history": geti("ovarian_history"),
            "low_activity": geti("low_activity"),
            "hormone_therapy": geti("hormone_therapy"),
            "brca_mutation": geti("brca_mutation"),
            "no_pregnancy_over_40": geti("never_pregnant_40"),
        }])

        # ---------- Risk label
        try:
            proba = model.predict_proba(X)
            if proba.shape[1] >= 3:
                cls = int(np.argmax(proba[0]))
                risk_label = ["Низкий риск","Средний риск","Высокий риск"][cls]
            else:
                p1 = float(proba[0,1])
                risk_label = "Низкий риск" if p1 < 0.33 else ("Средний риск" if p1 < 0.66 else "Высокий риск")
        except Exception:
            pred = int(model.predict(X)[0])
            risk_label = {0:"Низкий риск",1:"Средний риск",2:"Высокий риск"}.get(pred,"Средний риск")

        # ---------- Explainability (force 1-D)
        feature_names = list(X.columns)
        contrib = None
        try:
            if "shap" in globals():
                explainer = shap.Explainer(model, X)
                sv = explainer(X)
                raw = getattr(sv, "values", None)
                if raw is not None:
                    import numpy as _np
                    arr = _np.asarray(raw)
                    arr = _np.squeeze(arr)            # drop singleton dims
                    if arr.ndim == 2:                 # (n_features, n_classes)
                        arr = arr.mean(axis=1)
                    elif arr.ndim == 0:
                        arr = _np.array([float(arr)])
                    contrib = arr
        except Exception:
            contrib = None

        if contrib is None:
            fi = getattr(model, "feature_importances_", None)
            contrib = np.array(fi, dtype=float) if fi is not None else np.zeros(len(feature_names), dtype=float)

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

        reasons = []
        for idx in order[:3]:
            i = int(idx)
            name = feature_names[i]
            human = HUMAN_NAMES.get(name, name)
            tag = " (можно изменить)" if name in MODIFIABLE else " (неизменяемый фактор)"
            reasons.append(f"• {human}{tag}")
        shap_reason = "Этот балл повышается из-за:\n" + "\n".join(reasons)

        # ---------- Bar chart
        shap_img_url = None
        try:
            top_k = min(8, len(feature_names))
            sel = np.array(order[:top_k], dtype=int)
            labels = [feature_names[i] for i in sel][::-1]
            vals = contrib[sel][::-1]  # signed

            plt.figure(figsize=(7, 4), dpi=160)
            # color by sign: positive => increase risk, negative => decrease
            colors = ["tab:red" if v > 0 else "tab:blue" for v in vals]
            plt.barh(labels, vals, color=colors)
            plt.axvline(0, linewidth=1, color="#444")
            plt.xlabel("вклад признака (− снижает, + повышает)")
            plt.tight_layout()

            out = os.path.join(STATIC_DIR, "feature_importance.png")
            plt.savefig(out, bbox_inches="tight")
            plt.close()
            shap_img_url = url_for("static", filename="feature_importance.png")
        except Exception:
            shap_img_url = None

        # ---------- Save for PDF/ICS
        session["last_input"] = X.to_dict(orient="records")[0]
        session["last_risk_label"] = risk_label
        session["last_shap_top"] = reasons

        advice = ("🔴 Срочно обратитесь к врачу." if "Высокий" in risk_label
                  else "🟡 Рекомендуем консультацию у врача." if "Средний" in risk_label
                  else "🟢 Риск низкий. Продолжайте профилактику.")

        return render_template(
            "result.html",
            title=risk_label,
            advice=advice,
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

    # Register Unicode font if available
    font_path = os.path.join(STATIC_DIR, "fonts", "DejaVuSans.ttf")
    use_font = os.path.exists(font_path)
    if use_font:
        try:
            pdfmetrics.registerFont(TTFont("DZV", font_path))
            base_font = "DZV"
        except Exception:
            base_font = "Helvetica"
    else:
        base_font = "Helvetica"

    c = canvas.Canvas(pdf_path, pagesize=A4)
    W, H = A4
    y = H - 2 * cm

    c.setFont(base_font, 16)
    c.drawString(2 * cm, y, "myZone — Отчёт по оценке риска")
    y -= 1.0 * cm

    c.setFont(base_font, 12)
    c.drawString(2 * cm, y, f"Итоговый риск: {risk_label}")
    y -= 0.8 * cm

    c.setFont(base_font, 12)
    c.drawString(2 * cm, y, "Входные данные:")
    y -= 0.6 * cm

    c.setFont(base_font, 10)
    for k, v in inputs.items():
        line = f"{k}: {v}"
        c.drawString(2.2 * cm, y, line)
        y -= 0.48 * cm
        if y < 3 * cm:
            c.showPage()
            c.setFont(base_font, 10)
            y = H - 2 * cm

    feat_img = os.path.join(STATIC_DIR, "feature_importance.png")
    if os.path.exists(feat_img):
        c.showPage()
        # keep aspect ratio
        img_w = W - 4 * cm
        c.drawImage(feat_img, 2 * cm, H/2 - 2 * cm, width=img_w, preserveAspectRatio=True, mask='auto')

    c.showPage()
    c.setFont(base_font, 12)
    c.drawString(2 * cm, H - 2 * cm, "Почему такой балл:")
    y = H - 3 * cm
    c.setFont(base_font, 10)
    for line in shap_top:
        c.drawString(2.2 * cm, y, line)
        y -= 0.48 * cm
        if y < 2 * cm:
            c.showPage()
            c.setFont(base_font, 10)
            y = H - 2 * cm

    c.save()
    return send_file(pdf_path, as_attachment=True, download_name="myzone_report.pdf")

@app.route("/reminder_ics")
def reminder_ics():
    # be resilient even if no session values yet
    risk_label = session.get("last_risk_label") or request.args.get("risk", "Низкий риск")

    if "Высокий" in risk_label:
        title, days = "myZone: запишитесь к врачу", 3
    elif "Средний" in risk_label:
        title, days = "myZone: консультация у врача", 7
    else:
        title, days = "myZone: повторная самооценка", 60

    from datetime import datetime, timedelta
    start = (datetime.utcnow() + timedelta(days=days)).strftime("%Y%m%dT%H%M%SZ")

    # ICS requires CRLF line endings; keep text very ASCII-friendly
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//myZone//RU",
        "BEGIN:VEVENT",
        f"DTSTART:{start}",
        "DURATION:PT30M",
        "SUMMARY:myZone reminder",
        "DESCRIPTION:Follow up based on your myZone risk result.",
        "END:VEVENT",
        "END:VCALENDAR",
        ""
    ]
    ics = "\r\n".join(lines)

    resp = make_response(ics)
    resp.headers["Content-Type"] = "text/calendar; charset=utf-8"
    resp.headers["Content-Disposition"] = "attachment; filename=myzone_reminder.ics"
    return resp

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
PY






