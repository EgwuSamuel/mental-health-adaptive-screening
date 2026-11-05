"""
Certified Adaptive Screening (CAS) - core library.

The screener asks questionnaire items one at a time. After each answer a model
estimates p = P(full-scale score >= cut-off | answers so far). Screening stops:

  * POSITIVE as soon as the answered items already reach the cut-off (exact);
  * NEGATIVE when even maximal answers to the remaining items cannot reach the
    cut-off (exact), or when p <= lambda (model-based early stop).

lambda is chosen by conformal risk control on a calibration set so that the
miss rate among true positives (false-negative rate, FNR) is <= alpha in
expectation (CRC, Angelopoulos et al. 2024) or with probability >= 1 - delta
(Learn-then-Test style binomial bound). Group-wise (Mondrian) lambdas give the
same guarantee inside every demographic group; with several partitions a person
gets the smallest lambda of the groups they belong to, which keeps the
guarantee for every group of every partition.

Only model-based negative stops can be wrong, so all errors are misses relative
to the full questionnaire; specificity relative to the full scale is 100%.
A designated safety item (PHQ-9 item 9, self-harm) is always asked; any answer
> 0 sends the person to the full questionnaire and crisis information.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import binom
from sklearn.ensemble import HistGradientBoostingClassifier

MAXV = 3  # each item scored 0-3


@dataclass
class Instrument:
    name: str
    items: list[str]
    cutoff: int
    safety_item: int | None = None  # index into items
    min_items: int = 2              # never stop on the model before this many answers

    @property
    def J(self) -> int:
        return len(self.items)


PHQ9 = Instrument("PHQ-9", [f"phq{i}" for i in range(1, 10)], cutoff=10, safety_item=8)
GAD7 = Instrument("GAD-7", [f"gad{i}" for i in range(1, 8)], cutoff=10, safety_item=None)


# --------------------------------------------------------------------------- features
def make_features(A: np.ndarray, inst: Instrument) -> np.ndarray:
    """A: (n, J) answers with NaN for items not yet asked."""
    asked = ~np.isnan(A)
    partial = np.nansum(A, axis=1)
    n_asked = asked.sum(axis=1)
    remaining_max = MAXV * (inst.J - n_asked)
    need = inst.cutoff - partial
    return np.column_stack([A, partial, n_asked, need, need / np.maximum(remaining_max, 1)])


def deterministic_status(A: np.ndarray, inst: Instrument) -> np.ndarray:
    """+1 surely positive, -1 surely negative, 0 undetermined."""
    asked = ~np.isnan(A)
    partial = np.nansum(A, axis=1)
    remaining_max = MAXV * (inst.J - asked.sum(axis=1))
    out = np.zeros(len(A), dtype=int)
    out[partial >= inst.cutoff] = 1
    out[(partial + remaining_max) < inst.cutoff] = -1
    return out


def random_masks(X: np.ndarray, reps: int, rng: np.random.Generator, force_masked: int | None = None):
    """Replicate rows `reps` times, each with a random subset of items hidden."""
    n, J = X.shape
    Xr = np.repeat(X.astype(float), reps, axis=0)
    k = rng.integers(0, J, size=len(Xr))  # number of answered items, 0..J-1
    keys = rng.random((len(Xr), J))
    if force_masked is not None:
        keys[:, force_masked] = 2.0  # always hidden
        k = np.minimum(k, J - 1)
    ranks = keys.argsort(axis=1).argsort(axis=1)
    hide = ranks >= k[:, None]
    Xr[hide] = np.nan
    return Xr, np.repeat(np.arange(n), reps)


# --------------------------------------------------------------------------- models
@dataclass
class CASModel:
    inst: Instrument
    outcome: object = None
    item_models: list = field(default_factory=list)
    seed: int = 42

    def fit(self, X: np.ndarray, reps: int = 10, item_reps: int = 5):
        rng = np.random.default_rng(self.seed)
        y = (X.sum(axis=1) >= self.inst.cutoff).astype(int)
        Xm, src = random_masks(X, reps, rng)
        self.outcome = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0,
            random_state=self.seed).fit(make_features(Xm, self.inst), y[src])
        self.item_models = []
        for j in range(self.inst.J):
            Xm, src = random_masks(X, item_reps, rng, force_masked=j)
            m = HistGradientBoostingClassifier(
                max_iter=150, learning_rate=0.1, max_leaf_nodes=31, random_state=self.seed)
            m.fit(np.delete(make_features(Xm, self.inst), j, axis=1), X[src, j].astype(int))
            self.item_models.append(m)
        return self

    def p_pos(self, A: np.ndarray) -> np.ndarray:
        p = self.outcome.predict_proba(make_features(A, self.inst))[:, 1]
        st = deterministic_status(A, self.inst)
        p[st == 1], p[st == -1] = 1.0, 0.0
        return p

    def item_dist(self, A: np.ndarray, j: int) -> np.ndarray:
        m = self.item_models[j]
        pr = m.predict_proba(np.delete(make_features(A, self.inst), j, axis=1))
        out = np.zeros((len(A), MAXV + 1))
        out[:, m.classes_.astype(int)] = pr
        return out

    # ----------------------------------------------------------------- policy
    def next_item(self, A: np.ndarray) -> np.ndarray:
        """Greedy choice: unasked item with the lowest expected posterior entropy."""
        n, J = A.shape
        asked = ~np.isnan(A)
        score = np.full((n, J), np.inf)
        for j in range(J):
            rows = np.where(~asked[:, j])[0]
            if len(rows) == 0:
                continue
            sub = A[rows]
            pv = self.item_dist(sub, j)
            hyp = np.repeat(sub, MAXV + 1, axis=0)
            hyp[:, j] = np.tile(np.arange(MAXV + 1), len(rows))
            ph = self.p_pos(hyp).reshape(len(rows), MAXV + 1)
            score[rows, j] = (pv * _entropy(ph)).sum(axis=1)
        return score.argmin(axis=1)

    def trajectories(self, X: np.ndarray, order: str | list[int] = "adaptive"):
        """Simulate the question sequence for fully-answered records.

        Returns orders (n, J) and probs/psum (n, J+1): value after k answers.
        """
        n, J = X.shape
        A = np.full((n, J), np.nan)
        orders = np.zeros((n, J), dtype=int)
        probs = np.zeros((n, J + 1))
        psum = np.zeros((n, J + 1))
        probs[:, 0] = self.p_pos(A)
        for k in range(J):
            if isinstance(order, str):
                j = self.next_item(A)
            else:
                j = np.full(n, order[k])
            A[np.arange(n), j] = X[np.arange(n), j]
            orders[:, k] = j
            probs[:, k + 1] = self.p_pos(A)
            psum[:, k + 1] = np.nansum(A, axis=1)
        return Trajectories(self.inst, X, orders, probs, psum)


def _entropy(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


# --------------------------------------------------------------------------- trajectories
@dataclass
class Trajectories:
    inst: Instrument
    X: np.ndarray
    orders: np.ndarray
    probs: np.ndarray
    psum: np.ndarray

    def __post_init__(self):
        inst, n, J = self.inst, len(self.X), self.inst.J
        self.y = (self.X.sum(axis=1) >= inst.cutoff).astype(int)
        k = np.arange(J + 1)
        surely_pos = self.psum >= inst.cutoff
        surely_neg = (self.psum + MAXV * (J - k)[None, :]) < inst.cutoff
        det = surely_pos | surely_neg
        det[:, J] = True
        det[:, 0] = False
        self.k_det = det.argmax(axis=1)  # first k with an exact answer
        # model can stop at k in [min_items, k_det)
        allowed = (k[None, :] >= inst.min_items) & (k[None, :] < self.k_det[:, None])
        self.allowed = allowed
        pr = np.where(allowed, self.probs, np.inf)
        self.m = pr.min(axis=1)  # smallest lambda that makes this person stop on the model
        if inst.safety_item is not None:
            self.safety_flag = self.X[:, inst.safety_item] > 0
        else:
            self.safety_flag = np.zeros(n, dtype=bool)
        # a safety flag always leads to the full questionnaire -> never missed
        self.m_eff = np.where(self.safety_flag, np.inf, self.m)

    def outcome(self, lam):
        """Per-person (items asked, missed) for a scalar or per-person lambda."""
        inst, n, J = self.inst, len(self.X), self.inst.J
        lam = np.broadcast_to(np.asarray(lam, dtype=float), (n,))
        hit = self.allowed & (self.probs <= lam[:, None])
        any_hit = hit.any(axis=1)
        k_stop = np.where(any_hit, hit.argmax(axis=1), self.k_det)
        model_neg = any_hit
        if inst.safety_item is not None:
            asked_safety = (self.orders == inst.safety_item) & (np.arange(J)[None, :] < k_stop[:, None])
            items = k_stop + (~asked_safety.any(axis=1)).astype(int)
        else:
            items = k_stop.copy()
        items = np.where(self.safety_flag, J, items)
        missed = (self.y == 1) & model_neg & ~self.safety_flag
        return items, missed


# --------------------------------------------------------------------------- calibration
NO_STOP = -1.0


def crc_lambda(m_pos: np.ndarray, alpha: float) -> float:
    """Conformal risk control for FNR with loss 1[m <= lambda] on calibration positives."""
    n = len(m_pos)
    c = int(np.floor(alpha * (n + 1) - 1))  # misses allowed
    return _lambda_from_count(m_pos, c)


def pac_lambda(m_pos: np.ndarray, alpha: float, delta: float = 0.1) -> float:
    """Largest lambda whose binomial test rejects FNR > alpha at level delta (fixed sequence)."""
    n = len(m_pos)
    if n == 0:
        return NO_STOP
    cs = np.arange(0, n + 1)
    ok = binom.cdf(cs, n, alpha) <= delta
    if not ok[0]:
        return NO_STOP
    c = int(cs[ok].max())
    return _lambda_from_count(m_pos, c)


def _lambda_from_count(m_pos, c):
    if c < 0:
        return NO_STOP
    finite = np.sort(m_pos[np.isfinite(m_pos)])
    if c >= len(finite):
        return 1.0 - 1e-12  # every positive could be missed without exceeding the budget
    return float(np.nextafter(finite[c], -np.inf))


def fit_lambdas(traj: Trajectories, groups: dict[str, np.ndarray] | None, alpha: float,
                method: str = "crc", delta: float = 0.1):
    """Return {'marginal': lam, partition: {group: lam}}."""
    f = crc_lambda if method == "crc" else (lambda m, a: pac_lambda(m, a, delta))
    pos = traj.y == 1
    out = {"marginal": f(traj.m_eff[pos], alpha), "partitions": {}}
    for pname, g in (groups or {}).items():
        out["partitions"][pname] = {str(v): f(traj.m_eff[pos & (g == v)], alpha) for v in np.unique(g)}
    return out


def person_lambda(lams: dict, groups: dict[str, np.ndarray] | None, n: int, use_groups: bool):
    if not use_groups or not lams["partitions"]:
        return np.full(n, lams["marginal"])
    lam = np.full(n, np.inf)
    for pname, table in lams["partitions"].items():
        g = groups[pname].astype(str)
        lam = np.minimum(lam, np.array([table.get(v, NO_STOP) for v in g]))
    return lam


# --------------------------------------------------------------------------- baselines
def two_stage(X: np.ndarray, inst: Instrument, stage_items: list[int], threshold: float, weights=None,
              with_safety: bool = False):
    """Ask stage items; if (weighted) score >= threshold give the full scale, else stop negative.

    with_safety=True also asks the safety item on a negative first stage (any answer > 0 then
    leads to the full scale), which makes the comparison with CAS like-for-like.
    """
    w = np.ones(len(stage_items)) if weights is None else np.asarray(weights)
    s = X[:, stage_items] @ w
    y = X.sum(axis=1) >= inst.cutoff
    full = s >= threshold
    stage_n = len(stage_items)
    if with_safety and inst.safety_item is not None and inst.safety_item not in stage_items:
        flag = X[:, inst.safety_item] > 0
        full = full | flag
        stage_n += 1
    items = np.where(full, inst.J, stage_n)
    missed = y & ~full
    return items, missed


def without_safety(traj: "Trajectories") -> "Trajectories":
    """Same simulated sessions, but with no forced safety item (supplementary comparison)."""
    inst = Instrument(traj.inst.name, traj.inst.items, traj.inst.cutoff, None, traj.inst.min_items)
    return Trajectories(inst, traj.X, traj.orders, traj.probs, traj.psum)


def summarise(items, missed, y, groups: dict[str, np.ndarray] | None = None):
    pos = y == 1
    res = {"n": int(len(y)), "n_pos": int(pos.sum()),
           "mean_items": float(items.mean()),
           "fnr": float(missed[pos].mean()) if pos.any() else float("nan"),
           "sensitivity": float(1 - missed[pos].mean()) if pos.any() else float("nan"),
           "specificity": 1.0}
    if groups:
        worst = {}
        for pname, g in groups.items():
            per = {}
            for v in np.unique(g):
                sel = pos & (g == v)
                if sel.sum() >= 20:
                    per[str(v)] = {"n_pos": int(sel.sum()), "fnr": float(missed[sel].mean())}
            res[f"by_{pname}"] = per
            if per:
                worst[pname] = max(d["fnr"] for d in per.values())
        res["worst_group_fnr"] = worst
    return res
