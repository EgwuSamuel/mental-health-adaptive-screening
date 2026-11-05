"""
train_v1.py - corrected re-run of the ORIGINAL (v1) synthetic experiment.

Kept only so the numbers in the first manuscript draft can be reproduced and
checked. The data generator and features are unchanged from the original
model/train.py, so the headline numbers are identical. What changed:

  * paths are relative to this file, prints are ASCII (the original crashed on
    Windows at the arrow character, before metadata.json was written);
  * a rule baseline is reported: the synthetic labels are a deterministic
    function of the inputs, so the generating rule scores 100%;
  * feature importance is computed for the DEPLOYED model (Gradient Boosting),
    not the Random Forest as in the original Table 7;
  * every number is written to results_v1.json.

Item naming note: in the original app, "q9" was the GAD-7 item "Feeling
nervous, anxious or on edge" (not PHQ-9 item 9, which asks about self-harm),
and q10/q11 were GAD-7 items 2 and 3. The generator does not depend on item
meaning, so this only affects interpretation, not the numbers.

Run:  python legacy_v1/train_v1.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HERE = Path(__file__).resolve().parent
SEED = 42
N = 3000

FEATURES = [f"q{i}" for i in range(1, 12)] + [
    "duration", "func_impact", "phq9_total", "gad2_total",
    "somatic_cluster", "cognitive_cluster", "core_depression",
    "anxiety_cluster", "duration_phq_interaction",
]  # 20 features (the draft said 19)


def generate_synthetic_data(n=N):
    """Identical to the original generator (same RNG call order)."""
    np.random.seed(SEED)
    rows = []
    for _ in range(n):
        phq = np.random.choice([0, 1, 2, 3], size=9, p=[0.35, 0.30, 0.22, 0.13])
        gad = np.random.choice([0, 1, 2, 3], size=2, p=[0.35, 0.30, 0.22, 0.13])
        duration = np.random.choice([0, 1, 2, 3], p=[0.35, 0.25, 0.25, 0.15])
        func = np.random.choice([0, 1, 2, 3], p=[0.40, 0.30, 0.20, 0.10])
        phq_t, gad_t = int(phq.sum()), int(gad.sum())
        composite = phq_t * [0.7, 0.85, 1.0, 1.15][duration] + gad_t * 1.2 + func * 1.5
        # NOTE: labels use the PHQ cut-offs AND an ad-hoc composite score.
        if phq_t >= 15 or composite >= 22:
            label = 2
        elif phq_t >= 10 or composite >= 13:
            label = 1
        else:
            label = 0
        rows.append(list(phq) + list(gad) + [duration, func, phq_t, gad_t, composite, label])
    cols = [f"q{i}" for i in range(1, 12)] + [
        "duration", "func_impact", "phq9_total", "gad2_total", "composite", "label"]
    return pd.DataFrame(rows, columns=cols)


def engineer_features(df):
    df = df.copy()
    df["somatic_cluster"] = df["q3"] + df["q4"] + df["q5"]
    df["cognitive_cluster"] = df["q7"] + df["q8"]
    df["core_depression"] = df["q1"] + df["q2"]
    df["anxiety_cluster"] = df["q9"] + df["q10"] + df["q11"]
    df["duration_phq_interaction"] = df["duration"] * df["phq9_total"]
    return df


def rule_label(df):
    return np.where((df.phq9_total >= 15) | (df.composite >= 22), 2,
                    np.where((df.phq9_total >= 10) | (df.composite >= 13), 1, 0))


def main():
    df = engineer_features(generate_synthetic_data())
    X, y = df[FEATURES].values, df["label"].values
    idx = np.arange(len(df))
    X_tr, X_te, y_tr, y_te, i_tr, i_te = train_test_split(
        X, y, idx, test_size=0.2, random_state=SEED, stratify=y)

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=5,
            class_weight="balanced", random_state=SEED),
        "SVM (RBF)": Pipeline([("scaler", StandardScaler()), ("clf", SVC(
            kernel="rbf", C=1.0, probability=True, class_weight="balanced", random_state=SEED))]),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.1, random_state=SEED),
        "Logistic Regression": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=SEED))]),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    out = {"n_total": len(df), "n_train": len(y_tr), "n_test": len(y_te), "n_features": len(FEATURES),
           "class_counts": {k: int(v) for k, v in zip(["low", "moderate", "high"], np.bincount(y))},
           "cv": {}}
    print("5-fold CV accuracy (training set)")
    for name, m in models.items():
        s = cross_val_score(m, X_tr, y_tr, cv=cv, scoring="accuracy")
        out["cv"][name] = {"mean": round(float(s.mean()), 4), "std": round(float(s.std()), 4)}
        print(f"  {name:22s} {s.mean():.4f} +/- {s.std():.4f}")

    best = max(out["cv"], key=lambda k: out["cv"][k]["mean"])
    gb = models[best].fit(X_tr, y_tr)
    pred, proba = gb.predict(X_te), gb.predict_proba(X_te)
    rep = classification_report(y_te, pred, target_names=["low", "moderate", "high"], output_dict=True)
    out["selected_model"] = best
    out["test"] = {
        "accuracy": round(accuracy_score(y_te, pred), 4),
        "auc_macro_ovr": round(roc_auc_score(y_te, proba, multi_class="ovr", average="macro"), 4),
        "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
        "per_class": {k: {m: round(rep[k][m], 3) for m in ("precision", "recall", "f1-score")}
                      for k in ("low", "moderate", "high")},
        "macro_f1": round(rep["macro avg"]["f1-score"], 3),
    }
    te = df.iloc[i_te]
    phq_only = np.where(te.phq9_total >= 15, 2, np.where(te.phq9_total >= 10, 1, 0))
    out["baselines_on_test"] = {
        "generating_rule_accuracy": round(float((rule_label(te) == y_te).mean()), 4),
        "phq_cutoff_only_accuracy": round(float((phq_only == y_te).mean()), 4),
    }
    out["share_labels_not_explained_by_phq_cutoffs"] = round(float(
        (np.where(df.phq9_total >= 15, 2, np.where(df.phq9_total >= 10, 1, 0)) != df.label).mean()), 4)

    perm = permutation_importance(gb, X_te, y_te, n_repeats=20, random_state=SEED)
    out["importance_deployed_model"] = {
        "impurity": dict(sorted(((f, round(float(v), 3)) for f, v in zip(FEATURES, gb.feature_importances_)),
                                key=lambda t: -t[1])[:8]),
        "permutation_test_set": dict(sorted(((f, round(float(v), 3)) for f, v in zip(FEATURES, perm.importances_mean)),
                                            key=lambda t: -t[1])[:8]),
    }
    rf = models["Random Forest"].fit(X_tr, y_tr)
    out["importance_random_forest_(what_draft_Table7_showed)"] = dict(sorted(
        ((f, round(float(v), 3)) for f, v in zip(FEATURES, rf.feature_importances_)), key=lambda t: -t[1])[:8])

    (HERE / "results_v1.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("selected_model", "test", "baselines_on_test")}, indent=2))
    print("Saved -> legacy_v1/results_v1.json")


if __name__ == "__main__":
    main()
