"""Simple baselines next to CatBoost, per-regime models, monthly AUC table.

Same causal regimes, same 6 walk-forward test months and folds as corrected.py (ablation_main.pkl).
Arms added:
  reversal   score = -close_return (sign of the last return, no fitting);
  trend3     score = -trend_strength_3 (no fitting);
  regime_freq  P(up | regime) estimated on the training part;
  logit      logistic regression on the standardised base features (fit on the training part);
  per_regime separate CatBoost per regime, same fixed hyperparameters, 3 seeds.
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pickle
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.argv = ["corrected.py"]
src = open("corrected.py", encoding="utf-8").read()
head = src.split("# ---------------- primary model")[0]
exec(head)
c = pickle.load(open("ablation_main.pkl", "rb"))
probs, folds, R = c["probs"], c["folds"], c["R"]
pp = pickle.load(open("primary_main.pkl", "rb"))[0]
df = df.assign(pp=pp).dropna(subset=["pp"]).reset_index(drop=True)
delong = None
exec("def delong" + src.split("def delong")[1].split("\n\n\n")[0])
HP = dict(iterations=1000, learning_rate=0.05655, depth=6, l2_leaf_reg=10.7376, bagging_temperature=0.6119,
          random_strength=0.9061, loss_function="Logloss", eval_metric="AUC", use_best_model=True,
          early_stopping_rounds=50, verbose=0, thread_count=-1)

months = [str(df.open_time.iloc[te.start])[:7] for _, _, te in folds]
scores = {k: [] for k in ("reversal", "trend3", "regime_freq", "logit", "per_regime")}
ys = []
for f, (tr, va, te) in enumerate(folds):
    T, V, E = df.iloc[tr], df.iloc[va], df.iloc[te]
    ys.append(E["target"].values)
    scores["reversal"].append(-E["close_return"].values)
    scores["trend3"].append(-E["trend_strength_3"].values)
    freq = T.groupby("market_mode")["target"].mean()
    scores["regime_freq"].append(E["market_mode"].map(freq).values)
    sc_ = StandardScaler().fit(T[BASE])
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(sc_.transform(T[BASE]), T["target"])
    scores["logit"].append(lr.predict_proba(sc_.transform(E[BASE]))[:, 1])
    pr = np.full(len(E), np.nan)
    for j in sorted(df["market_mode"].unique()):
        mt, mv, me = (T.market_mode == j).values, (V.market_mode == j).values, (E.market_mode == j).values
        if me.sum() == 0:
            continue
        ps = []
        for s in SEEDS:
            m = CatBoostClassifier(**HP, random_seed=s)
            m.fit(T.loc[mt, BASE], T.loc[mt, "target"], eval_set=(V.loc[mv, BASE], V.loc[mv, "target"]))
            ps.append(m.predict_proba(E.loc[me, BASE])[:, 1])
        pr[me] = np.mean(ps, 0)
    scores["per_regime"].append(pr)
    P(f"fold {f} {months[f]} done")

for a in ("base", "regime", "stack", "regime+stack"):
    scores[a] = probs[a]["test"]

rows = []
for a, lst in scores.items():
    for f, (s, y) in enumerate(zip(lst, ys)):
        rows.append(dict(arm=a, month=months[f], auc=roc_auc_score(y, s), acc=np.mean((s > np.median(s)) == y)))
M = pd.DataFrame(rows)
pd.set_option("display.width", 250)
P("\nMONTHLY AUC:\n", M.pivot(index="arm", columns="month", values="auc").round(4).to_string())
P("\nMEAN ± SD over months:\n", M.groupby("arm").auc.agg(["mean", "std"]).round(4).to_string())

yall = np.concatenate(ys)
pall = {a: np.concatenate(v) for a, v in scores.items()}
for a in scores:
    P(f"POOLED {a}: AUC {roc_auc_score(yall, pall[a]):.4f}")
for a, b in (("base", "reversal"), ("base", "trend3"), ("base", "logit"), ("logit", "reversal"),
             ("per_regime", "base"), ("base", "regime_freq"), ("reversal", "regime_freq")):
    A, B, z, p = delong(yall, pall[a], pall[b])
    dm = M[M.arm == a].set_index("month").auc - M[M.arm == b].set_index("month").auc
    P(f"DELONG {a} vs {b}: {A:.4f} vs {B:.4f}, z={z:.2f}, p={p:.4f}; monthly diff {dm.mean():+.4f} ± "
      f"{dm.std(ddof=1) / np.sqrt(len(dm)):.4f}, wins {int((dm > 0).sum())}/{len(dm)}")
pickle.dump(dict(M=M, scores=scores, ys=ys, months=months), open("a2_baselines.pkl", "wb"))
P("DONE")
