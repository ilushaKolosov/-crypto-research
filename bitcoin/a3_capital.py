"""Capital-return version of the position rule.

Step return r_t = P_t (S_{t+1}/S_t - 1) - k |P_t - P_{t-1}|  (fraction of capital, P_0 = 0).
Search ranges restricted to the conditions of Statement 1: d <= 0 <= u.
Outputs: fee sweep (6 test months x 2 optimiser seeds), tuning-test gap, exposure, turnover,
Lo (2002) autocorrelation-adjusted Sharpe, monthly table, block-bootstrap CI of SR(tanh) - SR(sign).
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pickle
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.argv = ["corrected.py"]
src = open("corrected.py", encoding="utf-8").read()
head = src.split("# ---------------- primary model")[0]
exec(head)
c = pickle.load(open("ablation_main.pkl", "rb"))
probs, folds = c["probs"], c["folds"]
pp = pickle.load(open("primary_main.pkl", "rb"))[0]
df = df.assign(pp=pp).dropna(subset=["pp"]).reset_index(drop=True)
close = df["close"].values
months = [str(df.open_time.iloc[te.start])[:7] for _, _, te in folds]
FEES = [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02]
ARM = "regime+stack"


def rule(x, d, u, beta, eps):
    pos = np.zeros_like(x)
    L, S = x >= u, x <= d
    pos[L] = np.tanh(beta * (x[L] - u) ** (1 + eps))
    pos[S] = -np.tanh(beta * (d - x[S]) ** (1 + eps))
    return pos


def ret_k(pos, S, k):
    p = pos[:-1]
    return p * (S[1:] / S[:-1] - 1) - k * np.abs(np.diff(np.r_[0.0, p]))


def ann(r):
    vol = r.std() * np.sqrt(NY)
    return r.mean() * NY, vol, (r.mean() * NY / vol if vol > 0 else np.nan)


def lo_factor(r, L=96):
    r = r - r.mean()
    v = (r ** 2).mean()
    if v == 0:
        return 1.0
    rho = np.array([(r[l:] * r[:-l]).mean() / v for l in range(1, L + 1)])
    w = 1 - np.arange(1, L + 1) / (L + 1)
    return float(np.sqrt(max(1 + 2 * (w * rho).sum(), 1e-6)))


def pos_of(kind, x, prm):
    if kind == "sign":
        return np.sign(x)
    if kind == "linear":
        return np.clip(prm["c"] * x, -1, 1)
    if kind == "tanh_beta":
        return np.tanh(prm["beta"] * x)
    if kind == "tanh_sym":
        return rule(x, -prm["u"], prm["u"], prm["beta"], 0.0)
    if kind == "threshold":
        return np.where(x >= prm["u"], 1.0, np.where(x <= prm["d"], -1.0, 0.0))
    return rule(x, prm["d"], prm["u"], prm["beta"], prm["epsilon"])


def suggest(kind, t):
    if kind == "linear":
        return {"c": t.suggest_float("c", 0.1, 100, log=True)}
    if kind == "tanh_beta":
        return {"beta": t.suggest_float("beta", 0.1, 100, log=True)}
    if kind == "tanh_sym":
        return {"u": t.suggest_float("u", 0.0, 0.5), "beta": t.suggest_float("beta", 0.1, 100, log=True)}
    prm = {"d": t.suggest_float("d", -1.0, 0.0), "u": t.suggest_float("u", 0.0, 1.0)}
    if kind == "full":
        prm.update(beta=t.suggest_float("beta", 1.0, 100.0), epsilon=t.suggest_float("epsilon", 0.0, 10.0))
    return prm


def job(fee, kind, crit, f, seed):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    k = fee / 100
    _, va, te = folds[f]
    xv, xt = 2 * probs[ARM]["val"][f] - 1, 2 * probs[ARM]["test"][f] - 1
    Sv, St = close[va], close[te]
    prm = {}
    if kind != "sign":
        def obj(t):
            r = ret_k(pos_of(kind, xv, suggest(kind, t)), Sv, k)
            if crit == "pnl":
                return r.sum()
            sr = ann(r)[2]
            return -1e9 if not np.isfinite(sr) else sr
        st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=1000 * f + seed))
        st.optimize(obj, n_trials=400)
        prm = st.best_trial.params
    pv, pt = pos_of(kind, xv, prm), pos_of(kind, xt, prm)
    rv, rt = ret_k(pv, Sv, k), ret_k(pt, St, k)
    a = ann(rt)
    return dict(fee=fee, kind=kind, crit=crit, fold=f, month=months[f], seed=seed, prm=prm,
                sr_val=ann(rv)[2], ret=a[0], vol=a[1], sr=a[2], lo=lo_factor(rt),
                turnover=np.abs(np.diff(np.r_[0.0, pt[:-1]])).sum(), expo=np.abs(pt[:-1]).mean(),
                rt=rt.astype(np.float32) if fee in (0.0, 0.01) else None)


KINDS = ("linear", "tanh_beta", "tanh_sym", "threshold", "full")
tasks = [(fee, "sign", "-", f, 0) for fee in FEES for f in range(len(folds))]
tasks += [(fee, kind, crit, f, s) for fee in FEES for f in range(len(folds)) for s in range(2)
          for kind in KINDS for crit in ("sr", "pnl")]
print(f"jobs {len(tasks)}", flush=True)
F = pd.DataFrame(Parallel(n_jobs=-1)(delayed(job)(*t) for t in tasks))
F.to_pickle("a3_capital.pkl")

bh = []
for f, (_, _, te) in enumerate(folds):
    St = close[te]
    r = St[1:] / St[:-1] - 1
    bh.append(dict(month=months[f], ret=ann(r)[0], vol=ann(r)[1], sr=ann(r)[2], lo=lo_factor(r)))
BH = pd.DataFrame(bh)
pd.set_option("display.width", 250)
print("B&H per month:\n", BH.round(3).to_string())
print("B&H mean SR", round(BH.sr.mean(), 2), "mean vol %", round(100 * BH.vol.mean(), 1), "months>0", int((BH.sr > 0).sum()))

for v, nd in (("sr", 2), ("sr_val", 2)):
    print(f"\n{v} mean\n", F.pivot_table(index=["kind", "crit"], columns="fee", values=v, aggfunc="mean").round(nd).to_string())
print("\nshare of (month, seed) with SR>0\n",
      F.pivot_table(index=["kind", "crit"], columns="fee", values="sr", aggfunc=lambda s: (s > 0).mean()).round(2).to_string())
for v in ("ret", "vol"):
    print(f"\n{v} % mean\n", (100 * F.pivot_table(index=["kind", "crit"], columns="fee", values=v, aggfunc="mean")).round(1).to_string())
for v in ("expo", "turnover", "lo"):
    print(f"\n{v} mean\n", F.pivot_table(index=["kind", "crit"], columns="fee", values=v, aggfunc="mean").round(3).to_string())

S = F[F.crit.isin(["sr", "-"])]
print("\nSR by month, k=0.01%, crit sr (mean over seeds)\n",
      S[S.fee == 0.01].pivot_table(index="kind", columns="month", values="sr", aggfunc="mean").round(2).to_string())
print("\nSR by month, k=0, crit sr\n",
      S[S.fee == 0.0].pivot_table(index="kind", columns="month", values="sr", aggfunc="mean").round(2).to_string())
print("\nLo-adjusted mean SR (sr / lo)\n",
      S.assign(sra=S.sr / S.lo).pivot_table(index="kind", columns="fee", values="sra", aggfunc="mean").round(2).to_string())

rng = np.random.default_rng(0)


def pooled(kind, fee, seed):
    q = S[(S.kind == kind) & (S.fee == fee) & (S.seed == (0 if kind == "sign" else seed))].sort_values("fold")
    return np.concatenate(q.rt.values).astype(float)


for fee in (0.0, 0.01):
    for other in ("tanh_beta", "linear"):
        for seed in (0, 1):
            a, b = pooled(other, fee, seed), pooled("sign", fee, 0)
            sa, sb = ann(a)[2], ann(b)[2]
            Lb, nb = 96, len(a) // 96
            d = []
            for _ in range(2000):
                idx = (rng.integers(0, len(a) - Lb, nb)[:, None] + np.arange(Lb)).ravel()
                d.append(ann(a[idx])[2] - ann(b[idx])[2])
            lo, hi = np.percentile(d, [2.5, 97.5])
            print(f"BOOT k={fee}% {other} seed{seed} vs sign: pooled SR {sa:.2f} vs {sb:.2f}, diff {sa - sb:.2f}, "
                  f"95% CI [{lo:.2f}; {hi:.2f}], P(diff<=0)={np.mean(np.array(d) <= 0):.3f}", flush=True)
        dm = (S[(S.kind == other) & (S.fee == fee)].groupby("fold").sr.mean().values
              - S[(S.kind == "sign") & (S.fee == fee)].sort_values("fold").sr.values)
        print(f"MONTHLY k={fee}% {other} - sign: mean {dm.mean():.2f} ± {dm.std(ddof=1) / np.sqrt(len(dm)):.2f}, "
              f"wins {int((dm > 0).sum())}/{len(dm)}")

tb = S[(S.kind == "tanh_beta") & (S.fee == 0.01)]
print("\ntanh_beta k=0.01 tuned beta per (month, seed):", [round(p["beta"], 2) for p in tb.prm])
print("DONE")
