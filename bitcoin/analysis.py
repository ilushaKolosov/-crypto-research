"""Regime centroids, elbow, persistence, predictability by regime (causal regimes, 6 test months).
Fee sweep (break-even commission) for the decision rules, walk-forward tuning on the preceding month."""
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
rules = src.split("# ---------------- decision rule ----------------")[1].split("close = df[")[0]
exec(head)
fit_part = df.loc[:first_eval - 1]
c = pickle.load(open("ablation_main.pkl", "rb"))
probs, folds = c["probs"], c["folds"]
pp = pickle.load(open("primary_main.pkl", "rb"))[0]
df = df.assign(pp=pp).dropna(subset=["pp"]).reset_index(drop=True)
exec(rules)
pd.set_option("display.width", 250)

# ---------------- regimes ----------------
cent = pd.DataFrame(km.cluster_centers_, columns=CLUS)
cent["share_fit"] = fit_part["market_mode"].value_counts(normalize=True).sort_index().values
cent["up_rate_fit"] = fit_part.groupby("market_mode")["target"].mean().sort_index().values
print("CENTROIDS (standardised):\n", cent.round(3).to_string())

Xs = sc.transform(fit_part[CLUS])
sse = [KMeans(n_clusters=k, random_state=42, n_init=3).fit(Xs).inertia_ for k in range(1, 11)]
print("ELBOW SSE:", [round(s) for s in sse])
print("ELBOW rel drop:", [round(1 - b / a, 3) for a, b in zip(sse, sse[1:])])

m = df["market_mode"].values
runs = pd.Series((m[1:] != m[:-1]).cumsum())
rl = pd.DataFrame({"mode": m[1:], "run": runs}).groupby("run").agg(mode=("mode", "first"), n=("mode", "size"))
print("MEAN RUN LENGTH (bars):", rl.groupby("mode")["n"].mean().round(2).to_dict())
cont = (np.sign(df["close_return"]) == np.sign(df["close_return"].shift(-1)))
print("CONTINUATION RATE fit part by regime:", cont.loc[:first_eval - 1].groupby(fit_part["market_mode"]).mean().round(4).to_dict())


def auc_se(auc, n1, n0):
    q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
    return np.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2) + (n0 - 1) * (q2 - auc ** 2)) / (n1 * n0))


for part in ("val", "test"):
    idx = np.concatenate([np.arange(sl.start, sl.stop) for sl in (f[1] if part == "val" else f[2] for f in folds)])
    sub = df.iloc[idx][["market_mode", "target"]].copy()
    for arm in ("base", "regime+stack"):
        sub[arm] = np.concatenate(probs[arm][part])
    rows = []
    for g, s in sub.groupby("market_mode"):
        y = s["target"].values
        for arm in ("base", "regime+stack"):
            a = roc_auc_score(y, s[arm])
            rows.append(dict(mode=g, arm=arm, n=len(s), share=len(s) / len(sub), auc=a,
                             se=auc_se(a, y.sum(), len(y) - y.sum()), acc=accuracy_score(y, s[arm] > 0.5)))
    y = sub["target"].values
    for arm in ("base", "regime+stack"):
        a = roc_auc_score(y, sub[arm])
        rows.append(dict(mode="ALL", arm=arm, n=len(sub), share=1.0, auc=a, se=auc_se(a, y.sum(), len(y) - y.sum()),
                         acc=accuracy_score(y, sub[arm] > 0.5)))
    print(f"\nAUC BY REGIME ({part}, pooled over 6 folds):\n", pd.DataFrame(rows).round(4).to_string())

idx = np.concatenate([np.arange(f[2].start, f[2].stop) for f in folds])
sub = df.iloc[idx][["market_mode", "target", "open_time"]].copy()
sub["p"] = np.concatenate(probs["base"]["test"])
sub["month"] = np.repeat(np.arange(len(folds)), BLOCK)
print("\nAUC BY REGIME x MONTH (base, test):\n",
      sub.groupby(["market_mode", "month"]).apply(lambda s: roc_auc_score(s.target, s.p)).unstack().round(3).to_string())

# ---------------- fee sweep ----------------
close = df["close"].values
FEES = [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02]


def job(fee_pct, kind, crit, f, seed, arm):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    k = fee_pct / 100
    tr, va, te = folds[f]
    xv, xt = 2 * probs[arm]["val"][f] - 1, 2 * probs[arm]["test"][f] - 1
    Sv, St = close[va], close[te]

    def pnl_k(pos, S):
        p = pos[:-1]
        return p * (S[1:] - S[:-1]) - S[:-1] * np.abs(np.diff(np.r_[0.0, p])) * k

    def pos_of(x, prm):
        if kind == "linear":
            return np.clip(prm["c"] * x, -1, 1)
        if kind == "threshold":
            return np.where(x >= prm["u"], 1.0, np.where(x <= prm["d"], -1.0, 0.0))
        return rule(x, prm["d"], prm["u"], prm["beta"], prm["epsilon"], True)

    def obj(t):
        if kind == "linear":
            prm = {"c": t.suggest_float("c", 0.1, 100, log=True)}
        else:
            d = t.suggest_float("d", -1.0, 1.0)
            prm = {"d": d, "u": t.suggest_float("u", d, 1.0)}
            if kind != "threshold":
                prm.update(beta=t.suggest_float("beta", 1.0, 100.0), epsilon=t.suggest_float("epsilon", 0.01, 10.0))
        r = pnl_k(pos_of(xv, prm), Sv)
        if crit == "pnl":
            return r.sum()
        sr = stats_(r)[2]
        return -1e9 if not np.isfinite(sr) else sr

    if kind == "sign":
        pv, pt = np.sign(xv), np.sign(xt)
    else:
        st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=1000 * f + seed))
        st.optimize(obj, n_trials=400)
        prm = st.best_trial.params
        pv, pt = pos_of(xv, prm), pos_of(xt, prm)
    rv, rt = pnl_k(pv, Sv), pnl_k(pt, St)
    return dict(fee=fee_pct, kind=kind, crit=crit, fold=f, seed=seed, arm=arm,
                sr_val=stats_(rv)[2], ret=stats_(rt)[0], vol=stats_(rt)[1], sr=stats_(rt)[2],
                turnover=np.abs(np.diff(np.r_[0.0, pt[:-1]])).sum(), expo=np.abs(pt).mean())


tasks = [(fee, kind, crit, f, s, "regime+stack")
         for fee in FEES for f in range(len(folds)) for s in range(2)
         for kind, crits in (("sign", ("-",)), ("linear", ("pnl", "sr")), ("threshold", ("pnl", "sr")), ("tanh_cont", ("pnl", "sr")))
         for crit in crits if not (kind == "sign" and s > 0)]
print(f"\nfee sweep: {len(tasks)} jobs", flush=True)
out = Parallel(n_jobs=14, verbose=0)(delayed(job)(*t) for t in tasks)
F = pd.DataFrame(out)
bh = []
for f, (tr, va, te) in enumerate(folds):
    St = close[te]
    bh.append(stats_(St[1:] - St[:-1]))
F.to_pickle("fee_sweep.pkl")
print("B&H per month SR:", [round(b[2], 2) for b in bh], "mean", round(np.mean([b[2] for b in bh]), 2),
      "vol mean", round(np.mean([b[1] for b in bh])))
print("\nFEE SWEEP mean test SR (6 months x seeds):\n",
      F.pivot_table(index=["kind", "crit"], columns="fee", values="sr", aggfunc="mean").round(2).to_string())
print("\nFEE SWEEP months with SR>0 (share):\n",
      F.pivot_table(index=["kind", "crit"], columns="fee", values="sr", aggfunc=lambda s: (s > 0).mean()).round(2).to_string())
print("\nFEE SWEEP mean annual return:\n",
      F.pivot_table(index=["kind", "crit"], columns="fee", values="ret", aggfunc="mean").round(0).to_string())
print("\nFEE SWEEP mean vol:\n",
      F.pivot_table(index=["kind", "crit"], columns="fee", values="vol", aggfunc="mean").round(0).to_string())
print("\nFEE SWEEP mean turnover per month:\n",
      F.pivot_table(index=["kind", "crit"], columns="fee", values="turnover", aggfunc="mean").round(0).to_string())
print("\nFEE SWEEP mean val SR (in-sample):\n",
      F.pivot_table(index=["kind", "crit"], columns="fee", values="sr_val", aggfunc="mean").round(2).to_string())
print("DONE")
