"""
app.py - Certified Adaptive Screening (CAS) web app.

Asks PHQ-9 (depression) and GAD-7 (anxiety) items one at a time, choosing the
most informative next question and stopping early only when the conformal
safety rule allows it. The self-harm item is always asked.

Run:  python app.py   ->  http://localhost:5000
"""
import csv
import json
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from cas.core import GAD7, PHQ9
from cas.engine import Engine

BASE = Path(__file__).resolve().parent
app = Flask(__name__, template_folder="templates", static_folder="static")
app.json.sort_keys = False

ENGINE = None
LOAD_ERROR = None
try:
    ENGINE = Engine(BASE / "models" / "cas_artifacts.joblib")
except Exception as e:  # noqa: BLE001
    LOAD_ERROR = f"{type(e).__name__}: {e}"
    print("[WARN] Screener not loaded. Run: python study/run_all.py  -", LOAD_ERROR)

CRISIS = json.loads((BASE / "crisis_resources.json").read_text(encoding="utf-8"))
INSTRUMENTS = {"phq9": PHQ9, "gad7": GAD7}
SEXES = {"female", "male"}
AGE_BANDS = {"18-39", "40-59", "60+"}

RISK_INFO = {
    "low": {
        "title": "Low risk", "color": "green",
        "description": ("Your answers suggest few or mild symptoms at the moment. Keep looking after "
                        "your wellbeing and repeat this check in about 3 months, or sooner if things change."),
        "recommendations": [
            "Keep a regular sleep pattern: the same bedtime and wake-up time most days.",
            "Stay in touch with people you trust; regular contact protects mental health.",
            "Aim for about 30 minutes of physical activity on most days.",
        ],
    },
    "moderate": {
        "title": "Moderate risk", "color": "amber",
        "description": ("Your answers show symptoms that are worth taking seriously. Many people benefit "
                        "from support at this stage, before symptoms become more severe."),
        "recommendations": [
            "Talk to a doctor or a primary health care worker; you can show them this result.",
            "Repeat this check in two weeks to see whether things are improving.",
            "Short daily walks or other physical activity help mild to moderate low mood and worry.",
            "If worry is strong, try slow breathing: breathe in for 4, hold for 7, out for 8.",
        ],
    },
    "high": {
        "title": "Elevated risk", "color": "red",
        "description": ("Your answers suggest significant symptoms. Please speak with a mental health "
                        "professional soon. This is a screening tool; a professional can make a proper "
                        "assessment and discuss treatment with you."),
        "recommendations": [
            "Arrange to see a psychiatrist, clinical psychologist or doctor as soon as you can.",
            "In Lagos, the Federal Neuropsychiatric Hospital (Yaba) and LUTH (Idi-Araba) run psychiatric clinics.",
            "Tell someone you trust how you are feeling.",
        ],
    },
}


# ----------------------------------------------------------------- validation
def _bad(msg):
    return jsonify({"error": msg}), 400


def _validate(answers, module):
    if module not in INSTRUMENTS:
        return None, "module must be 'phq9' or 'gad7'"
    inst = INSTRUMENTS[module]
    if not isinstance(answers, dict):
        return None, "answers must be an object"
    clean = {}
    for k, v in answers.items():
        if k not in inst.items:
            return None, f"unknown item '{k}' for {module}"
        if isinstance(v, bool) or not isinstance(v, int) or v not in (0, 1, 2, 3):
            return None, f"{k} must be an integer 0-3"
        clean[k] = v
    return clean, None


def _demo(payload):
    sex = payload.get("sex")
    age = payload.get("age_band")
    if sex is not None and sex not in SEXES | {"prefer_not"}:
        return None, None, "sex must be 'female', 'male' or 'prefer_not'"
    if age is not None and age not in AGE_BANDS | {"prefer_not"}:
        return None, None, "age_band must be '18-39', '40-59', '60+' or 'prefer_not'"
    return (sex if sex in SEXES else None), (age if age in AGE_BANDS else None), None


# ----------------------------------------------------------------- routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/next", methods=["POST"])
def next_question():
    if ENGINE is None:
        return jsonify({"error": "Screener not loaded. Run: python study/run_all.py"}), 503
    p = request.get_json(silent=True) or {}
    module = p.get("module")
    answers, err = _validate(p.get("answers"), module)
    if err:
        return _bad(err)
    sex, age, err = _demo(p)
    if err:
        return _bad(err)
    return jsonify(ENGINE.step(module, answers, sex, age))


@app.route("/api/summary", methods=["POST"])
def summary():
    if ENGINE is None:
        return jsonify({"error": "Screener not loaded. Run: python study/run_all.py"}), 503
    p = request.get_json(silent=True) or {}
    sex, age, err = _demo(p)
    if err:
        return _bad(err)
    results = {}
    for module in ("phq9", "gad7"):
        answers, err = _validate(p.get(module), module)
        if err:
            return _bad(err)
        r = ENGINE.step(module, answers, sex, age)
        if not r["done"]:
            return _bad(f"{module} screening is not finished")
        results[module] = r

    dep, anx = results["phq9"], results["gad7"]
    safety = dep["safety_flag"]
    severe = dep.get("severity") in ("moderately severe", "severe") or anx.get("severity") == "severe"
    both = dep["decision"] == "positive" and anx["decision"] == "positive"
    if safety or severe or both:
        level = "high"
    elif dep["decision"] == "positive" or anx["decision"] == "positive":
        level = "moderate"
    else:
        level = "low"
    info = RISK_INFO[level]
    asked = dep["items_asked"] + anx["items_asked"]
    return jsonify({
        "risk_level": level, "risk_title": info["title"], "risk_color": info["color"],
        "description": info["description"], "recommendations": info["recommendations"],
        "safety_alert": safety,
        "crisis_resources": CRISIS["resources"] if (safety or level == "high") else [],
        "crisis_note": CRISIS["warning"],
        "depression": dep, "anxiety": anx,
        "questions_asked": asked, "questions_full": PHQ9.J + GAD7.J,
    })


@app.route("/api/model-info")
def model_info():
    if ENGINE is None:
        return jsonify({"error": "Screener not loaded", "detail": LOAD_ERROR}), 503
    perf = []
    table = BASE / "study" / "results" / "table2_methods.csv"
    if table.exists():
        with open(table, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["method"].startswith("CAS-G") and "internal" in row["dataset"]:
                    perf.append({k: row[k] for k in ("dataset", "instrument", "mean_items", "sensitivity")})
    art = ENGINE.art
    return jsonify({
        "name": "Certified Adaptive Screening (CAS)",
        "alpha": art["alpha"], "created": art["created"],
        "depression_calibration": art["depression"]["calibrated_on"],
        "anxiety_calibration": art["anxiety"]["calibrated_on"],
        "internal_test_performance": perf,
        "note": "Not yet calibrated on Nigerian data: recalibrate locally before clinical use.",
    })


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model_loaded": ENGINE is not None})


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", port=int(os.environ.get("PORT", 5000)))
