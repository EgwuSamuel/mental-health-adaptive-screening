"""
Runtime engine for the web app: decides the next question or the final result
from the answers given so far. Uses the same CASModel and stopping rule as the
study (cas/core.py), so the app behaves exactly as evaluated.

Two app-only choices, both conservative (they can only add questions):
  * after a POSITIVE screen the remaining items are asked so the full-scale
    severity band can be shown;
  * the safety item's answer is included when re-checking the stop rule.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from .core import GAD7, PHQ9, Instrument, deterministic_status

ROOT = Path(__file__).resolve().parents[1]

QUESTIONS = {
    "phq1": ("Little interest or pleasure in doing things", "Interest"),
    "phq2": ("Feeling down, depressed, or hopeless", "Mood"),
    "phq3": ("Trouble falling or staying asleep, or sleeping too much", "Sleep"),
    "phq4": ("Feeling tired or having little energy", "Energy"),
    "phq5": ("Poor appetite or overeating", "Appetite"),
    "phq6": ("Feeling bad about yourself, or that you are a failure or have let yourself or your family down",
             "Self-worth"),
    "phq7": ("Trouble concentrating on things, such as reading the newspaper or watching television",
             "Concentration"),
    "phq8": ("Moving or speaking so slowly that other people could have noticed, or the opposite: being so "
             "fidgety or restless that you have been moving around a lot more than usual", "Movement"),
    "phq9": ("Thoughts that you would be better off dead, or of hurting yourself in some way", "Safety"),
    "gad1": ("Feeling nervous, anxious, or on edge", "Anxiety"),
    "gad2": ("Not being able to stop or control worrying", "Worry"),
    "gad3": ("Worrying too much about different things", "Worry"),
    "gad4": ("Trouble relaxing", "Tension"),
    "gad5": ("Being so restless that it is hard to sit still", "Restlessness"),
    "gad6": ("Becoming easily annoyed or irritable", "Irritability"),
    "gad7": ("Feeling afraid, as if something awful might happen", "Fear"),
}

SEVERITY = {
    "PHQ-9": [(20, "severe"), (15, "moderately severe"), (10, "moderate"), (5, "mild"), (0, "minimal")],
    "GAD-7": [(15, "severe"), (10, "moderate"), (5, "mild"), (0, "minimal")],
}


class Engine:
    def __init__(self, path: Path = ROOT / "models" / "cas_artifacts.joblib"):
        self.art = joblib.load(path)
        self.modules = {
            "phq9": (PHQ9, self.art["depression"]),
            "gad7": (GAD7, self.art["anxiety"]),
        }

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _vector(inst: Instrument, answers: dict) -> np.ndarray:
        return np.array([[float(answers[i]) if i in answers else np.nan for i in inst.items]])

    def _lambda(self, module: str, sex: str | None, age_band: str | None) -> tuple[float, str]:
        inst, art = self.modules[module]
        table = art["lambda_groups"]
        key = f"{sex}|{age_band}" if module == "phq9" else str(sex)
        if sex in ("female", "male") and key in table:
            return float(table[key]), f"group {key}"
        return float(art["lambda_marginal"]), "all respondents"

    # ------------------------------------------------------------------ main API
    def step(self, module: str, answers: dict, sex: str | None = None, age_band: str | None = None) -> dict:
        inst, art = self.modules[module]
        model = art["model"]
        A = self._vector(inst, answers)
        asked = ~np.isnan(A[0])
        n_asked = int(asked.sum())
        partial = int(np.nansum(A))
        lam, lam_scope = self._lambda(module, sex, age_band)

        safety_idx = inst.safety_item
        safety_flag = safety_idx is not None and asked[safety_idx] and A[0, safety_idx] > 0
        status = int(deterministic_status(A, inst)[0])
        p = float(model.p_pos(A)[0])

        def ask(j, reason):
            item = inst.items[j]
            text, domain = QUESTIONS[item]
            return {"done": False, "module": module, "item": item, "text": text, "domain": domain,
                    "asked": n_asked, "of": inst.J, "reason": reason, "p_positive": round(p, 4)}

        def first_unasked():
            return int(np.where(~asked)[0][0])

        # 1. safety flag or positive screen -> complete the scale
        if safety_flag or status == 1:
            if n_asked < inst.J:
                return ask(first_unasked(), "completing scale (safety)" if safety_flag else
                           "completing scale for severity")
            return self._result(inst, A, "positive" if partial >= inst.cutoff else "negative", "exact",
                                safety_flag, lam, lam_scope, p)
        # 2. negative decision (exact or certified)
        certified = n_asked >= inst.min_items and p <= lam
        if status == -1 or certified or n_asked == inst.J:
            if safety_idx is not None and not asked[safety_idx]:
                return ask(safety_idx, "safety question (always asked)")
            if n_asked == inst.J:
                decision = "positive" if partial >= inst.cutoff else "negative"
                return self._result(inst, A, decision, "exact", safety_flag, lam, lam_scope, p)
            return self._result(inst, A, "negative", "exact" if status == -1 else "certified",
                                safety_flag, lam, lam_scope, p)
        # 3. keep going: most informative unasked item
        j = int(model.next_item(A)[0])
        return ask(j, "most informative next question")

    def _result(self, inst, A, decision, basis, safety_flag, lam, lam_scope, p):
        asked = ~np.isnan(A[0])
        score = int(np.nansum(A))
        complete = bool(asked.all())
        severity = None
        if complete:
            severity = next(lbl for thr, lbl in SEVERITY[inst.name] if score >= thr)
        return {"done": True, "instrument": inst.name, "decision": decision, "basis": basis,
                "items_asked": int(asked.sum()), "items_total": inst.J, "partial_score": score,
                "complete": complete, "severity": severity, "safety_flag": bool(safety_flag),
                "lambda": lam, "lambda_scope": lam_scope, "p_positive": round(p, 4),
                "guarantee": (f"Early stop certified so that, in the calibration population "
                              f"({self.art['depression' if inst.name == 'PHQ-9' else 'anxiety']['calibrated_on']}), "
                              f"at most {int(self.art['alpha']*100)}% of true positives are missed.")
                if basis == "certified" else None}
