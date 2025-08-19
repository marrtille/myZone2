from flask import Flask, render_template, request, redirect, url_for, make_response, send_file, session, Response, jsonify
import os, uuid, sqlite3
from datetime import datetime, timedelta
import json, threading, time
from collections import deque

from flask_cors import CORS

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================
# Flask app (must exist before CORS)
# =========================
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-me")
CORS(app)  # ok to keep on for local dev

# =========================
# SHAP (optional but preferred)
# =========================
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

# =========================
# Paths & Model
# =========================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(APP_DIR, "templates")
DB_PATH = os.path.join(APP_DIR, "diary.db")
STATIC_DIR = os.path.join(APP_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

MODEL_PATH = os.path.join(APP_DIR, "breast_risk_model.pkl")
model = joblib.load(MODEL_PATH)

# =========================
# Serial bridge (optional; works locally, degrades on Render)
# =========================
SERIAL_PORT = os.environ.get("SERIAL_PORT")  # set locally (e.g., COM5 or /dev/ttyACM0). Leave unset on Render.
BAUD = int(os.environ.get("SERIAL_BAUD", "115200"))

latest = {}               # last reading (dict)
ring = deque(maxlen=5000) # recent readings
stop_flag = False

# Try importing pyserial; it's present locally, may be absent in cloud
try:
    import serial  # type: ignore
    HAVE_SERIAL = True
except Exception:
    HAVE_SERIAL = False

def _serial_reader():
    """Read JSON lines from Arduino and update latest/ring."""
    global latest
    while not stop_flag:
        try:
            with serial.Serial(SERIAL_PORT, BAUD, timeout=1) as ser:
                buf = ""
                while not stop_flag:
                    chunk = ser.read(ser.in_waiting or 1).decode('utf-8', errors='ignore')
                    if not chunk:
                        continue
                    buf += chunk
                    while '\n' in buf:
                        line, buf = buf.split('\n', 1)
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            j = json.loads(line)
                            latest = j
                            ring.append(j)
                        except Exception:
                            # ignore non-JSON line
                            pass
        except Exception:
            # Port unavailable: retry after a short delay (e.g., not plugged in)
            time.sleep(1)

# Start reader only if we have both pyserial and a configured port
if HAVE_SERIAL and SERIAL_PORT:
    threading.Thread(target=_serial_reader, daemon=True).start()

@app.route("/api/latest")
def api_latest():
    return jsonify(latest or {})

@app.route("/api/history")
def api_history():
    return jsonify(list(ring))

@app.route("/api/stream")
def api_stream():
    def gen():
        last_ts = None
        while True:
            # If serial is unavailable (e.g., Render), stream a heartbeat so the page still works
            if not (HAVE_SERIAL and SERIAL_PORT):
                yield f"data: {json.dumps({'piezo_mV':0,'drop_pct':0,'objC':0,'ambC':0,'ts':int(time.time()*1000)})}\n\n"
                time.sleep(0.5)
                continue

            if latest and latest.get("ts") != last_ts:
                last_ts = latest.get("ts")
                yield f"data: {json.dumps(latest)}\n\n"
            time.sleep(0.05)
    return Response(gen(), mimetype="text/event-stream")

# =========================
# DB init
# =========================
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
    con.commit()
    con.close()

init_db()

# =========================
# Risk form helpers
# =========================
MODIFIABLE = {"smoking","alcohol","low_activity","bmi","hormone_therapy"}
HUMAN_NAMES = {
    "age":"возраст",
    "sex":"пол",
    "bmi":"повышенный индекс массы тела (BMI)",
    "family_history":"семейная история рака груди",
    "early_periods":"раннее начало менструаций",
    "late_menopause":"поздняя менопауза",
    "ovarian_cancer_history":"личная/семейная история рака яичников",
    "low_activity":"низкая физическая активность",
    "hormone_therapy":"гормональная терапия",
    "brca_mutation":"мутация BRCA",
    "no_pregnancy_over_40":"отсутствие беременности до 40",
    "smoking":"курение",
    "alcohol":"регулярный алкоголь",
}

def _geti(form, name, default=0):
    v = form.get(name, default)
    try:
        return int(v)
    except Exception:
        try:
            return int(float(v))
        except Exception:
            return int(default)

def _getf(form, name, default=0.0):
    v = form.get(name, default)
    try:
        return float(v)
    except Exception:
        return float(default)

def collect_input(f):
    """Parse incoming form into a single-row DataFrame with the model's features."""
    X = pd.DataFrame([{
        "age": _geti(f, "age"),
        "sex": _geti(f, "sex"),
        "bmi": _getf(f, "bmi"),
        "family_history": _geti(f, "family_history"),
        "smoking": _geti(f, "smoking"),
        "alcohol": _geti(f, "alcohol"),
        "early_periods": _geti(f, "early_periods"),
        "late_menopause": _geti(f, "late_menopause"),
        # form field is ovarian_history, model feature is ovarian_cancer_history
        "ovarian_cancer_history": _geti(f, "ovarian_history"),
        "low_activity": _geti(f, "low_activity"),
        "hormone_therapy": _geti(f, "hormone_therapy"),
        "brca_mutation": _geti(f, "brca_mutation"),
        # form field is never_pregnant_40, model feature is no_pregnancy_over_40
        "no_pregnancy_over_40": _geti(f, "never_pregnant_40"),
    }])
    return X

def _risk_label_from_proba(proba):
    """Turn predict_proba output into RU label."""
    if proba.shape[1] >= 3:
        cls = int(np.argmax(proba[0]))
        return ["Низкий риск","Средний риск","Высокий риск"][cls]
    else:
        p1 = float(proba[0,1])
        return "Низкий риск" if p1 < 0.33 else ("Средний риск" if p1 < 0.66 else "Высокий риск")

def _bg_for_kernel(feature_names: list[str]) -> pd.DataFrame:
    """
    Small, reasonable background for KernelExplainer when we don't have training data.
    Uses typical medians and toggles of binary features to give contrast.
    """
    base = {
        "age": 45, "sex": 1, "bmi": 25.0,
        "family_history": 0, "smoking": 0, "alcohol": 0,
        "early_periods": 0, "late_menopause": 0,
        "ovarian_cancer_history": 0, "low_activity": 0,
        "hormone_therapy": 0, "brca_mutation": 0,
        "no_pregnancy_over_40": 0,
    }
    base = {k: v for k, v in base.items() if k in feature_names}
    rows = [base.copy()]

    if "age" in feature_names:   rows += [{**base, "age": v} for v in (30, 55, 65)]
    if "bmi" in feature_names:   rows += [{**base, "bmi": v} for v in (22.0, 30.0)]
    for b in ("family_history","smoking","alcohol","low_activity",
              "hormone_therapy","brca_mutation","no_pregnancy_over_40",
              "early_periods","late_menopause","ovarian_cancer_history"):
        if b in feature_names:
            rows.append({**base, b: 1})

    bg = pd.DataFrame(rows)[feature_names]
    return bg

def _safe_shap_contrib(model, X: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    """
    Return 1-D signed contributions for the *positive* class.
    Uses TreeExplainer for tree models; KernelExplainer otherwise with a small background.
    """
    contrib = None
    if SHAP_AVAILABLE:
        try:
            is_tree = hasattr(model, "estimators_") or hasattr(model, "tree_")
            if is_tree:
                explainer = shap.TreeExplainer(model)
                sv = explainer.shap_values(X)
                if isinstance(sv, list):  # legacy API
                    arr = np.asarray(sv[1][0]) if len(sv) > 1 else np.asarray(sv[0][0])
                else:                      # modern API
                    vals = getattr(sv, "values", None)
                    arr = np.asarray(vals[0]) if vals is not None else np.asarray(sv)[0]
            else:
                bg = _bg_for_kernel(feature_names)
                explainer = shap.KernelExplainer(model.predict_proba, bg, link="logit")
                sv = explainer.shap_values(X, nsamples="auto")
                arr = np.asarray(sv[1][0]) if isinstance(sv, list) else np.asarray(sv)[0]

            arr = np.asarray(arr, dtype=float).squeeze()
            if arr.ndim == 2:
                arr = arr[:, 1] if arr.shape[1] > 1 else arr[:, 0]
            contrib = arr
        except Exception:
            contrib = None

    if contrib is None:
        fi = getattr(model, "feature_importances_", None)
        contrib = np.asarray(fi, dtype=float) if fi is not None else np.zeros(len(feature_names), dtype=float)

    contrib = np.asarray(contrib, dtype=float).reshape(-1)
    if contrib.shape[0] != len(feature_names):
        contrib = np.resize(contrib, (len(feature_names),))
    return contrib

def model_predict(X: pd.DataFrame):
    """Predict risk and compute signed contributions."""
    try:
        proba = model.predict_proba(X)
        risk_label = _risk_label_from_proba(proba)
    except Exception:
        pred = int(model.predict(X)[0])
        risk_label = {0:"Низкий риск",1:"Средний риск",2:"Высокий риск"}.get(pred, "Средний риск")

    feature_names = list(X.columns)
    contrib = _safe_shap_contrib(model, X, feature_names)
    order = np.argsort(np.abs(contrib))[::-1]
    return contrib, risk_label, order, feature_names

# =========================
# Routes
# =========================
@app.route("/")
def index():
    if os.path.exists(os.path.join(TEMPLATES_DIR, "index.html")):
        return render_template("index.html")
    return """
    <h1>myZone</h1>
    <p><a href="/risk_form">Оценка риска</a> • <a href="/diary">Дневник</a> • <a href="/self_exam">Памятка</a> • <a href="/ultrasound">Ультразвуковое исследование</a></p>
    """

@app.route("/ultrasound")
def ultrasound():
    # Page that subscribes to /api/stream
    if os.path.exists(os.path.join(TEMPLATES_DIR, "ultrasound.html")):
        return render_template("ultrasound.html", title="Ультразвуковое исследование")
    # Fallback simple page if template missing
    return """
    <h1>Ультразвуковое исследование</h1>
    <p>Подключаюсь к потоку…</p>
    <script>
    const es = new EventSource('/api/stream');
    es.onmessage = (ev) => { try { const j = JSON.parse(ev.data); console.log(j); } catch(e){} };
    </script>
    """

@app.route("/risk_form", methods=["GET", "POST"])
def risk_form():
    if request.method == "GET":
        return render_template("risk_form.html")

    try:
        X = collect_input(request.form)
        contrib, risk_label, order, feature_names = model_predict(X)

        reasons = []
        for idx in order[:3]:
            i = int(idx)
            name = feature_names[i]
            human = HUMAN_NAMES.get(name, name)
            tag = " (можно изменить)" if name in MODIFIABLE else " (неизменяемый фактор)"
            reasons.append(f"• {human}{tag}")
        shap_reason = "Этот балл повышается из-за:\n" + "\n".join(reasons)

        shap_img_url = None
        try:
            top_k = min(8, len(feature_names))
            sel = np.array(order[:top_k], dtype=int)
            labels = [feature_names[i] for i in sel][::-1]
            vals = contrib[sel][::-1]

            plt.figure(figsize=(7, 4), dpi=160)
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
    except Exception as e:
        import traceback
        err = traceback.format_exc()
        print("risk_form ERROR:\n", err)
        return f"<h2>Ошибка обработки формы</h2><pre>{err}</pre>", 500

@app.route("/self_exam")
def self_exam():
    if os.path.exists(os.path.join(TEMPLATES_DIR, "self_exam.html")):
        return render_template("self_exam.html")
    return """
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
    base_font = "Helvetica"
    if os.path.exists(font_path):
        try:
            pdfmetrics.registerFont(TTFont("MYZONE_UNI", font_path))
            base_font = "MYZONE_UNI"
        except Exception:
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
        c.drawString(2.2 * cm, y, f"{k}: {v}")
        y -= 0.48 * cm
        if y < 3 * cm:
            c.showPage()
            c.setFont(base_font, 10)
            y = H - 2 * cm

    feat_img = os.path.join(STATIC_DIR, "feature_importance.png")
    if os.path.exists(feat_img):
        c.showPage()
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
    risk_label = session.get("last_risk_label") or request.args.get("risk", "Низкий риск")

    if "Высокий" in risk_label:
        days = 3
    elif "Средний" in risk_label:
        days = 7
    else:
        days = 60

    start = (datetime.utcnow() + timedelta(days=days)).strftime("%Y%m%dT%H%M%SZ")

    # ICS requires CRLF line endings
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

# =========================
# Run
# =========================
if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
    finally:
        stop_flag = True
