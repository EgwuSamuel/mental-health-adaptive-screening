"""
Reproduce every number, table and figure of the study.

    python study/download_data.py   # once
    python study/data_prep.py       # once
    python study/run_all.py         # ~20-30 min on a laptop

Outputs: study/results/*.csv|json, study/results/figures/*.png,
and the deployable screener in models/cas_artifacts.joblib.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from cas.core import (GAD7, NO_STOP, PHQ9, CASModel, crc_lambda, deterministic_status,  # noqa: E402
                      fit_lambdas, make_features, person_lambda, random_masks, summarise,
                      two_stage, without_safety)

warnings.filterwarnings("ignore")
SEED = 42
ALPHA = 0.05
DELTA = 0.10
N_VALIDITY_REPS = 500
N_RECAL_REPS = 300
RES = HERE / "results"
FIG = RES / "figures"
DEV_CYCLES = ["2005-2006", "2007-2008", "2009-2010", "2011-2012", "2013-2014", "2015-2016"]

# reference categorical palette (fixed order) + text inks
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "lines.linewidth": 2,
})

LOG = []


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.append(line)


def split_idx(n, fracs, seed=SEED):
    perm = np.random.default_rng(seed).permutation(n)
    cuts = np.cumsum([int(f * n) for f in fracs[:-1]])
    return np.split(perm, cuts)


# ============================================================ probability-model comparison
def compare_prob_models(inst, X_tr, X_te, seed=SEED):
    rng = np.random.default_rng(seed)
    y_tr_full = (X_tr.sum(1) >= inst.cutoff).astype(int)
    y_te_full = (X_te.sum(1) >= inst.cutoff).astype(int)
    Xm, src = random_masks(X_tr, 6, rng)
    Xt, srct = random_masks(X_te, 3, rng)
    und = deterministic_status(Xt, inst) == 0  # only states where a model is needed
    F, Ft = make_features(Xm, inst), make_features(Xt, inst)
    y, yt = y_tr_full[src], y_te_full[srct]

    def lr_feats(A):
        Z = np.nan_to_num(A, nan=0.0)
        return np.column_stack([Z, np.isnan(A[:, :inst.J]).astype(float)])

    out = []
    from sklearn.ensemble import HistGradientBoostingClassifier
    models = {
        "Histogram gradient boosting (used)": (HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0,
            random_state=seed), lambda A: A),
        "Logistic regression": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), lr_feats),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = (XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.08,
                                           subsample=0.9, tree_method="hist", random_state=seed,
                                           n_jobs=4, verbosity=0), lambda A: A)
    except Exception:  # noqa: BLE001
        pass
    for name, (m, tf) in models.items():
        m.fit(tf(F), y)
        p = m.predict_proba(tf(Ft[und]))[:, 1]
        out.append({"instrument": inst.name, "model": name, "auc": roc_auc_score(yt[und], p),
                    "brier": brier_score_loss(yt[und], p), "n_partial_states": int(und.sum())})
    return pd.DataFrame(out)


# ============================================================ methods table
def threshold_for_sensitivity(X, inst, items, target=0.95):
    s = X[:, items].sum(1)
    y = X.sum(1) >= inst.cutoff
    best = 0
    for t in range(0, 3 * len(items) + 1):
        if (s[y] >= t).mean() >= target:
            best = t
    return best


def methods_rows(dataset, inst, X, Ta, Tf, lams, groups_eval, groups_lam, stage_cfg):
    y = Ta.y
    rows = []

    def add(method, items, missed, note=""):
        r = summarise(items, missed, y, groups_eval)
        rows.append({"dataset": dataset, "instrument": inst.name, "method": method,
                     "n": r["n"], "n_pos": r["n_pos"], "mean_items": r["mean_items"],
                     "items_saved_%": 100 * (1 - r["mean_items"] / inst.J),
                     "sensitivity": r["sensitivity"], "fnr": r["fnr"],
                     "worst_group_fnr": json.dumps({k: round(v, 3) for k, v in r.get("worst_group_fnr", {}).items()}),
                     "note": note})

    add(f"Full {inst.name}", np.full(len(y), inst.J), np.zeros(len(y), bool), "reference")
    has_safety = inst.safety_item is not None
    for label, items, thr in stage_cfg:
        if has_safety:
            it, ms = two_stage(X, inst, items, thr, with_safety=True)
            add(label + " + safety item", it, ms, "two-stage; safety item asked (like-for-like)")
            it, ms = two_stage(X, inst, items, thr)
            add(label + " (no safety item)", it, ms, "two-stage as used in practice; self-harm item skipped")
        else:
            it, ms = two_stage(X, inst, items, thr)
            add(label, it, ms, "two-stage")
    it, ms = Tf.outcome(NO_STOP)
    add("Exact curtailment (standard order)", it, ms)
    it, ms = Ta.outcome(NO_STOP)
    add("Exact curtailment (adaptive order)", it, ms)
    it, ms = Tf.outcome(0.05)
    add("Stochastic curtailment (standard order, p<=0.05)", it, ms, "uncalibrated threshold")
    it, ms = Tf.outcome(lams["fixed"]["marginal"])
    add("CAS - standard order (CRC)", it, ms)
    it, ms = Ta.outcome(lams["adaptive"]["marginal"])
    add("CAS (adaptive, CRC)", it, ms)
    if groups_lam:
        lam = person_lambda(lams["group"], groups_lam, len(y), True)
        it, ms = Ta.outcome(lam)
        add("CAS-G (adaptive, group-wise CRC)", it, ms)
    it, ms = Ta.outcome(lams["pac"]["marginal"])
    add(f"CAS-PAC (adaptive, P(FNR<={ALPHA})>={1-DELTA:.2f})", it, ms)
    if has_safety and "nosafety" in lams:
        Tn = without_safety(Ta)
        it, ms = Tn.outcome(lams["nosafety"]["marginal"])
        add("CAS without safety item (supplementary)", it, ms, "compare with two-stage (no safety item)")
    return rows


# ============================================================ validity under exchangeability
def validity_experiment(T, groups, reps, alpha, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(T.y)
    res = {"marginal": {"overall": []}, "group": {"overall": []}}
    for p in groups:
        for v in np.unique(groups[p]):
            res["marginal"][f"{p}={v}"] = []
            res["group"][f"{p}={v}"] = []
    items_m, items_g = [], []
    for _ in range(reps):
        perm = rng.permutation(n)
        ci, ti = perm[: n // 2], perm[n // 2:]
        Tc_m = T.m_eff[ci][T.y[ci] == 1]
        lam_m = crc_lambda(Tc_m, alpha)
        gl = {"marginal": lam_m, "partitions": {}}
        for p, g in groups.items():
            gl["partitions"][p] = {str(v): crc_lambda(T.m_eff[ci][(T.y[ci] == 1) & (g[ci] == v)], alpha)
                                   for v in np.unique(g)}
        lam_g = person_lambda(gl, groups, n, True)
        for key, lam, store in (("marginal", np.full(n, lam_m), items_m), ("group", lam_g, items_g)):
            items, missed = T.outcome(lam)
            pos = (T.y == 1)
            tmask = np.zeros(n, bool)
            tmask[ti] = True
            res[key]["overall"].append(missed[tmask & pos].mean())
            store.append(items[tmask].mean())
            for p, g in groups.items():
                for v in np.unique(g):
                    sel = tmask & pos & (g == v)
                    res[key][f"{p}={v}"].append(missed[sel].mean() if sel.sum() else np.nan)
    rows = []
    for key in res:
        for grp, vals in res[key].items():
            v = np.array(vals, float)
            rows.append({"lambda": key, "group": grp, "mean_fnr": np.nanmean(v),
                         "p90_fnr": np.nanpercentile(v, 90), "share_reps_fnr_gt_alpha": np.nanmean(v > alpha)})
    rows.append({"lambda": "marginal", "group": "mean_items", "mean_fnr": np.mean(items_m)})
    rows.append({"lambda": "group", "group": "mean_items", "mean_fnr": np.mean(items_g)})
    return pd.DataFrame(rows), res


# ============================================================ local recalibration
def recalibration_curve(T, name, sizes, reps, alpha, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(T.y)
    rows = []
    for N in sizes:
        if N > n // 2:
            continue
        fnrs, items, lams, npos = [], [], [], []
        for _ in range(reps):
            perm = rng.permutation(n)
            ci, ti = perm[:N], perm[N:]
            pos_c = T.y[ci] == 1
            lam = crc_lambda(T.m_eff[ci][pos_c], alpha)
            it, ms = T.outcome(lam)
            pt = T.y[ti] == 1
            fnrs.append(ms[ti][pt].mean())
            items.append(it[ti].mean())
            lams.append(lam)
            npos.append(pos_c.sum())
        fnrs = np.array(fnrs)
        rows.append({"target": name, "n_local": N, "mean_pos_in_local": float(np.mean(npos)),
                     "mean_fnr": fnrs.mean(), "p90_fnr": np.percentile(fnrs, 90),
                     "share_fnr_gt_alpha": (fnrs > alpha).mean(), "mean_items": np.mean(items),
                     "share_no_early_stop": float(np.mean(np.array(lams) == NO_STOP))})
    return rows


# ============================================================ figures
def fig_flow(path):
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.axis("off")
    W, H, Y = 0.175, 0.36, 0.42
    boxes = [
        (0.010, "1. Ask the most\ninformative\nunasked item"),
        (0.208, "2. Update\nP(full score >= 10)\nfrom answers so far"),
        (0.406, "3. Exact stop?\nalready >= 10, or\n10 now unreachable"),
        (0.604, "4. Certified stop?\nP <= lambda for\nthe person's group"),
        (0.802, "5. Safety item\nalways asked;\n> 0: full scale\n+ crisis contacts"),
    ]
    for x, t in boxes:
        ax.add_patch(plt.Rectangle((x, Y), W, H, fc="#eef4fc", ec=C[0], lw=1.2))
        ax.text(x + W / 2, Y + H / 2, t, ha="center", va="center", fontsize=6.4, color=INK, linespacing=1.3)
    for x, _ in boxes[:-1]:
        ax.annotate("", xy=(x + W + 0.023, Y + H / 2), xytext=(x + W + 0.002, Y + H / 2),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.1))
    # loop back: from box 4 bottom to box 1 bottom
    ax.annotate("", xy=(0.010 + W / 2, Y - 0.005), xytext=(0.604 + W / 2, Y - 0.005),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.0, connectionstyle="arc3,rad=-0.25"))
    ax.text(0.39, 0.17, "neither stop: ask another item", ha="center", fontsize=7, color=INK2)
    ax.text(0.5, 0.93, "lambda is set by conformal risk control on calibration data so that at most 5% "
            "of true positives are missed (per group for CAS-G)", ha="center", fontsize=7.2, color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.savefig(path)
    plt.close(fig)


def fig_pareto(curves, path):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
    for ax, (title, cdict, J) in zip(axes, curves):
        for i, (name, pts) in enumerate(cdict.items()):
            pts = sorted(pts)
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", ms=4, color=C[i], label=name)
        xs_all = [x for pts in cdict.values() for x, _ in pts]
        lo, hi = max(0, min(xs_all) - 0.3), max(xs_all) + 0.3
        ax.set_xlim(lo, hi)
        ax.axhline(1 - ALPHA, color=INK2, lw=1, ls="--")
        ax.text(hi - 0.05, 1 - ALPHA - 0.012, "95% target", ha="right", va="top", fontsize=7, color=INK2)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("Mean questions asked per person")
    axes[0].set_ylabel("Sensitivity vs full questionnaire")
    axes[0].set_ylim(0.6, 1.01)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(path)
    plt.close(fig)


def fig_validity(raw, groups_order, path, title):
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    pos = np.arange(len(groups_order))
    for k, (key, col, lab) in enumerate((("marginal", C[1], "Single (marginal) lambda"),
                                         ("group", C[0], "Group-wise lambda (CAS-G)"))):
        data = [np.array(raw[key][g], float) for g in groups_order]
        data = [d[~np.isnan(d)] for d in data]
        bp = ax.boxplot(data, positions=pos + (k - 0.5) * 0.32, widths=0.26, patch_artist=True,
                        showfliers=False, medianprops=dict(color=INK, lw=1))
        for b in bp["boxes"]:
            b.set(facecolor=col, alpha=0.75, edgecolor=col)
        for w in bp["whiskers"] + bp["caps"]:
            w.set(color=col)
        ax.plot([], [], color=col, lw=6, alpha=0.75, label=lab)
        means = [d.mean() for d in data]
        ax.scatter(pos + (k - 0.5) * 0.32, means, marker="D", s=14, color=INK, zorder=3)
    ax.axhline(ALPHA, color=INK2, ls="--", lw=1)
    ax.text(len(groups_order) - 0.5, ALPHA + 0.003, "alpha = 5%", ha="right", fontsize=7, color=INK2)
    ax.set_xticks(pos)
    ax.set_xticklabels([g.split("=", 1)[1].replace("|", "\n") for g in groups_order], fontsize=7)
    ax.set_ylabel("Miss rate among positives\n(500 random calibration splits)")
    ax.set_title(title, fontsize=9, loc="left", color=INK)
    ax.legend(fontsize=7.5, loc="upper left")
    fig.savefig(path)
    plt.close(fig)


def fig_recal(df, path):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for i, (name, g) in enumerate(df.groupby("target", sort=False)):
        axes[0].plot(g.n_local, g.mean_fnr, marker="o", ms=4, color=C[i], label=name)
        axes[1].plot(g.n_local, g.mean_items, marker="o", ms=4, color=C[i], label=name)
    axes[0].axhline(ALPHA, color=INK2, ls="--", lw=1)
    axes[0].set_ylabel("Mean miss rate among positives")
    axes[1].set_ylabel("Mean questions asked")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Local calibration sample size (people)")
    axes[0].set_title("Safety after local recalibration", fontsize=9, loc="left")
    axes[1].set_title("Burden after local recalibration", fontsize=9, loc="left")
    axes[1].legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_item_order(T, inst, labels, path, steps=4):
    M = np.zeros((inst.J, steps))
    for k in range(steps):
        M[:, k] = np.bincount(T.orders[:, k], minlength=inst.J) / len(T.orders)
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(inst.J))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xticks(range(steps))
    ax.set_xticklabels([f"Q{k+1}" for k in range(steps)])
    ax.set_xlabel("Position in the adaptive sequence")
    ax.grid(False)
    for i in range(inst.J):
        for k in range(steps):
            if M[i, k] >= 0.05:
                ax.text(k, i, f"{M[i, k]*100:.0f}%", ha="center", va="center", fontsize=6.5,
                        color="white" if M[i, k] > 0.5 else INK)
    fig.colorbar(im, ax=ax, fraction=0.04, label="Share of people")
    fig.savefig(path)
    plt.close(fig)


PHQ_LABELS = ["1 Anhedonia", "2 Depressed mood", "3 Sleep", "4 Fatigue", "5 Appetite",
              "6 Worthlessness", "7 Concentration", "8 Psychomotor", "9 Self-harm thoughts"]
GAD_LABELS = ["1 Nervous", "2 Can't stop worrying", "3 Worry too much", "4 Trouble relaxing",
              "5 Restless", "6 Irritable", "7 Afraid"]


# ============================================================ main
def run_instrument(inst, dev, externals, group_fn, group_fn_external, stage_cfg_fn, tag):
    log(f"== {inst.name}: development n={len(dev)}")
    X = dev[inst.items].values.astype(int)
    tr, ca, te = split_idx(len(X), [0.5, 0.25, 0.25])
    out = {"split_sizes": {"train": len(tr), "calibration": len(ca), "test": len(te)}}

    log("probability model comparison")
    mc = compare_prob_models(inst, X[tr], X[te])

    log("fitting CAS model")
    t0 = time.time()
    model = CASModel(inst, seed=SEED).fit(X[tr])
    out["fit_seconds"] = round(time.time() - t0, 1)

    log("trajectories: calibration + test")
    Ta_c, Tf_c = model.trajectories(X[ca]), model.trajectories(X[ca], order=list(range(inst.J)))
    Ta_t, Tf_t = model.trajectories(X[te]), model.trajectories(X[te], order=list(range(inst.J)))
    g_c = group_fn(dev.iloc[ca])
    g_t = group_fn(dev.iloc[te])
    lams = {
        "adaptive": fit_lambdas(Ta_c, None, ALPHA),
        "fixed": fit_lambdas(Tf_c, None, ALPHA),
        "group": fit_lambdas(Ta_c, g_c, ALPHA),
        "pac": fit_lambdas(Ta_c, None, ALPHA, "pac", DELTA),
    }
    if inst.safety_item is not None:
        lams["nosafety"] = fit_lambdas(without_safety(Ta_c), None, ALPHA)
    stage_cfg = stage_cfg_fn(X[tr])
    rows = methods_rows(f"{tag} internal test", inst, X[te], Ta_t, Tf_t, lams, g_t, g_t, stage_cfg)

    # alpha / threshold sweeps for the trade-off figure
    stage_names = {stage_cfg[0][0].split(" >=")[0]: stage_cfg[0][1], stage_cfg[-1][0].split(" >=")[0]: stage_cfg[-1][1]}
    curves = {"CAS (adaptive, CRC)": [], "CAS - standard order (CRC)": [],
              "Stochastic curtailment (std order)": []}
    suffix = " two-stage + safety item" if inst.safety_item is not None else " two-stage"
    for nm in stage_names:
        curves[nm + suffix] = []
    for a in (0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20):
        for key, Tc, Tt in (("CAS (adaptive, CRC)", Ta_c, Ta_t), ("CAS - standard order (CRC)", Tf_c, Tf_t)):
            it, ms = Tt.outcome(fit_lambdas(Tc, None, a)["marginal"])
            curves[key].append((it.mean(), 1 - ms[Tt.y == 1].mean()))
    for gma in (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3):
        it, ms = Tf_t.outcome(gma)
        curves["Stochastic curtailment (std order)"].append((it.mean(), 1 - ms[Tf_t.y == 1].mean()))
    for nm, items in stage_names.items():
        for thr in range(1, 6):
            it, ms = two_stage(X[te], inst, items, thr, with_safety=True)
            curves[nm + suffix].append((it.mean(), 1 - ms[Tf_t.y == 1].mean()))

    log("validity experiment (repeated calibration splits)")
    from cas.core import Trajectories
    pool = Trajectories(inst, np.vstack([Ta_c.X, Ta_t.X]), np.vstack([Ta_c.orders, Ta_t.orders]),
                        np.vstack([Ta_c.probs, Ta_t.probs]), np.vstack([Ta_c.psum, Ta_t.psum]))
    g_pool = {k: np.concatenate([g_c[k], g_t[k]]) for k in g_c}
    val_df, val_raw = validity_experiment(pool, g_pool, N_VALIDITY_REPS, ALPHA)

    log("external / shifted datasets")
    ext_T = {}
    for name, df in externals.items():
        Xe = df[inst.items].values.astype(int)
        Ta_e = model.trajectories(Xe)
        Tf_e = model.trajectories(Xe, order=list(range(inst.J)))
        ext_T[name] = Ta_e
        ge = group_fn_external(df)
        rows += methods_rows(name, inst, Xe, Ta_e, Tf_e, lams_external(lams, ge), ge, ge, stage_cfg)

    out["lambdas"] = lams
    out["stage_thresholds"] = [(a, b, c) for a, b, c in stage_cfg]
    return {"model": model, "rows": rows, "model_compare": mc, "curves": curves, "validity": val_df,
            "validity_raw": val_raw, "ext_T": ext_T, "test_T": Ta_t, "info": out, "groups_cal": g_c}


def lams_external(lams, ge):
    """External sets: use only partitions that exist there (no race outside NHANES)."""
    keep = dict(lams)
    keep["group"] = {"marginal": lams["group"]["marginal"],
                     "partitions": {p: t for p, t in lams["group"]["partitions"].items() if p in (ge or {})}}
    return keep


def main():
    RES.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    d = pd.read_pickle(HERE / "data" / "processed" / "all_data.pkl")
    phq = d[d.phq_complete].copy()
    gad = d[d.gad_complete].copy()

    # dataset summary
    summ = []
    for (ds, cyc), g in d.groupby(["dataset", "cycle"]):
        gp, gg = g[g.phq_complete], g[g.gad_complete]
        summ.append({"dataset": ds, "cycle": cyc, "n_records": len(g), "n_phq9_complete": len(gp),
                     "phq9_ge10_%": 100 * (gp.phq_total >= 10).mean() if len(gp) else np.nan,
                     "n_gad7_complete": len(gg),
                     "gad7_ge10_%": 100 * (gg.gad_total >= 10).mean() if len(gg) else np.nan,
                     "female_%": 100 * (g.sex == "female").mean(),
                     "age_mean": g.age.mean()})
    pd.DataFrame(summ).round(2).to_csv(RES / "table1_datasets.csv", index=False)

    # ---------------- depression (PHQ-9): develop on NHANES 2005-2016
    nh = phq[phq.dataset == "NHANES"]
    dev = nh[nh.cycle.isin(DEV_CYCLES)].reset_index(drop=True)

    def g_nh(df):
        return {"sex_age": (df.sex + "|" + df.age_band).values, "race": df.race.values}

    def g_sexage(df):
        return {"sex_age": (df.sex + "|" + df.age_band).values}

    def stage_phq(Xtr):
        t24 = threshold_for_sensitivity(Xtr, PHQ9, [1, 3])
        return [("PHQ-2 >=3 then PHQ-9", [0, 1], 3), ("PHQ-2 >=2 then PHQ-9", [0, 1], 2),
                (f"Items 2+4 >={t24} then PHQ-9 (PHQ-2W style)", [1, 3], t24)]

    ext_dep = {
        "NHANES 2017-2020 (temporal)": nh[nh.cycle == "2017-2020"],
        "NHANES 2021-2023 (temporal, post-COVID)": nh[nh.cycle == "2021-2023"],
        "China students (external)": phq[phq.dataset == "China students"],
        "Mexico medical students (external)": phq[phq.dataset == "Mexico medical students"],
        "Online hobbyist survey (external)": phq[phq.dataset == "Online hobbyist survey"],
    }
    dep = run_instrument(PHQ9, dev, ext_dep, g_nh, g_sexage, stage_phq, "NHANES 2005-2016")

    # ---------------- anxiety (GAD-7): develop on China students
    ch = gad[gad.dataset == "China students"].reset_index(drop=True)

    def g_sex(df):
        return {"sex": df.sex.values}

    def stage_gad(Xtr):
        return [("GAD-2 >=3 then GAD-7", [0, 1], 3), ("GAD-2 >=2 then GAD-7", [0, 1], 2)]

    ext_anx = {
        "Mexico medical students (external)": gad[gad.dataset == "Mexico medical students"],
        "Online hobbyist survey (external)": gad[gad.dataset == "Online hobbyist survey"],
    }
    anx = run_instrument(GAD7, ch, ext_anx, g_sex, g_sex, stage_gad, "China students")

    # ---------------- tables
    methods = pd.DataFrame(dep["rows"] + anx["rows"])
    methods.round(4).to_csv(RES / "table2_methods.csv", index=False)
    pd.concat([dep["model_compare"], anx["model_compare"]]).round(4).to_csv(
        RES / "table_s1_probability_models.csv", index=False)
    v1 = dep["validity"].assign(instrument="PHQ-9")
    v2 = anx["validity"].assign(instrument="GAD-7")
    pd.concat([v1, v2]).round(4).to_csv(RES / "table3_validity.csv", index=False)

    # ---------------- local recalibration (deployment protocol)
    log("local recalibration curves")
    rec = []
    rec += recalibration_curve(dep["ext_T"]["NHANES 2021-2023 (temporal, post-COVID)"],
                               "PHQ-9: NHANES 2021-2023", [100, 200, 400, 800, 1600], N_RECAL_REPS, ALPHA)
    rec += recalibration_curve(dep["ext_T"]["China students (external)"],
                               "PHQ-9: China students", [100, 200, 400, 800, 1600, 3200], N_RECAL_REPS, ALPHA)
    rec += recalibration_curve(dep["ext_T"]["Mexico medical students (external)"],
                               "PHQ-9: Mexico students", [100, 200, 300], N_RECAL_REPS, ALPHA)
    rec += recalibration_curve(anx["ext_T"]["Mexico medical students (external)"],
                               "GAD-7: Mexico students", [100, 200, 300], N_RECAL_REPS, ALPHA)
    rec_df = pd.DataFrame(rec)
    rec_df.round(4).to_csv(RES / "table4_local_recalibration.csv", index=False)

    # ---------------- response-time estimate (China timing data)
    ch_all = d[d.dataset == "China students"]
    sec_phq = float(np.nanmedian(ch_all[[f"phq_time{i}" for i in range(1, 10)]].values))
    sec_gad = float(np.nanmedian(ch_all[[f"gad_time{i}" for i in range(1, 8)]].values))

    # ---------------- figures
    log("figures")
    fig_flow(FIG / "fig1_cas_flow.png")
    curve_rows = [{"instrument": inst_name, "method": meth, "mean_items": x, "sensitivity": y}
                  for inst_name, cur in (("PHQ-9", dep["curves"]), ("GAD-7", anx["curves"]))
                  for meth, pts in cur.items() for x, y in sorted(pts)]
    pd.DataFrame(curve_rows).round(4).to_csv(RES / "table_s2_tradeoff_curves.csv", index=False)
    fig_pareto([("A  Depression (PHQ-9), NHANES internal test", dep["curves"], 9),
                ("B  Anxiety (GAD-7), China internal test", anx["curves"], 7)], FIG / "fig2_tradeoff.png")
    race_groups = [k for k in dep["validity_raw"]["marginal"] if k.startswith("race=")]
    sa_groups = [k for k in dep["validity_raw"]["marginal"] if k.startswith("sex_age=")]
    fig_validity(dep["validity_raw"], race_groups, FIG / "fig3a_validity_race.png",
                 "Depression: miss rate by race/ethnicity (NHANES held-out)")
    fig_validity(dep["validity_raw"], sa_groups, FIG / "fig3b_validity_sex_age.png",
                 "Depression: miss rate by sex and age band (NHANES held-out)")
    fig_recal(rec_df, FIG / "fig4_local_recalibration.png")
    fig_item_order(dep["test_T"], PHQ9, PHQ_LABELS, FIG / "fig5a_item_order_phq.png")
    fig_item_order(anx["test_T"], GAD7, GAD_LABELS, FIG / "fig5b_item_order_gad.png")

    # ---------------- deployable artifacts (sex x age-band groups only: race is not asked in the app)
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    # each partition's lambdas are fitted independently, so the sex x age table is valid on its own
    art = {
        "created": time.strftime("%Y-%m-%d"), "alpha": ALPHA, "sklearn": sklearn.__version__,
        "depression": {"model": dep["model"], "lambda_marginal": dep["info"]["lambdas"]["adaptive"]["marginal"],
                       "lambda_groups": dep["info"]["lambdas"]["group"]["partitions"].get("sex_age", {}),
                       "calibrated_on": "NHANES 2005-2016 calibration split (US adults)"},
        "anxiety": {"model": anx["model"], "lambda_marginal": anx["info"]["lambdas"]["adaptive"]["marginal"],
                    "lambda_groups": anx["info"]["lambdas"]["group"]["partitions"].get("sex", {}),
                    "calibrated_on": "China students calibration split"},
    }
    joblib.dump(art, models_dir / "cas_artifacts.joblib", compress=3)

    summary = {
        "alpha": ALPHA, "delta": DELTA, "seed": SEED, "sklearn": sklearn.__version__,
        "runtime_minutes": round((time.time() - t_start) / 60, 1),
        "median_seconds_per_item_china": {"phq9": sec_phq, "gad7": sec_gad},
        "depression": _jsonable(dep["info"]), "anxiety": _jsonable(anx["info"]),
    }
    (RES / "summary.json").write_text(json.dumps(summary, indent=2))
    (RES / "run_log.txt").write_text("\n".join(LOG))
    log(f"done in {summary['runtime_minutes']} min")


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


if __name__ == "__main__":
    main()
