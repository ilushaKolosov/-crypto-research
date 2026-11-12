# -*- coding: utf-8 -*-
"""Probe: how non-causal denoising inflates the test ROC-AUC of an ML classifier.

Latent log-price is a random walk, so the sign of the next latent increment is
unpredictable and any honest classifier has ROC-AUC = 0.5 on it.
"""
import os
import sys
import json
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import leak_model as exp_bi

N = 12000
TRAIN = 0.7
LAGS = 10
M = 10


def lag_features(series, lags=LAGS):
    d = np.diff(series, prepend=series[0])
    cols = [np.roll(d, k) for k in range(lags)]
    return np.column_stack(cols)


def run(seed):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0.0, exp_bi.SIG, N))
    y = x + rng.normal(0.0, exp_bi.NOISE, N)
    target = (np.roll(np.diff(x, append=x[-1]), 0) > 0).astype(int)  # sign(x_{t+1}-x_t)
    split = int(TRAIN * N)
    q, r = exp_bi.SIG ** 2, exp_bi.NOISE ** 2

    kf, ks = exp_bi.kalman_local_level(y, q, r)
    variants = {
        "raw": y,
        "dwt_global": exp_bi.dwt_full(y),
        "dwt_per_split": np.concatenate([exp_bi.dwt_full(y[:split]), exp_bi.dwt_full(y[split:])]),
        "dwt_causal": exp_bi.dwt_causal(y),
        "rts_global": ks,
        "rts_per_split": np.concatenate([
            exp_bi.kalman_local_level(y[:split], q, r)[1],
            exp_bi.kalman_local_level(y[split:], q, r)[1],
        ]),
        "kf_causal": kf,
    }
    out = {}
    valid = np.arange(LAGS, N - 1)
    tr = valid[valid < split]
    te = valid[valid >= split + LAGS]
    for name, s in variants.items():
        X = lag_features(s)
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0)
        clf.fit(X[tr], target[tr])
        p = clf.predict_proba(X[te])[:, 1]
        out[name] = roc_auc_score(target[te], p)
    return out


if __name__ == "__main__":
    allres = [run(2000 + s) for s in range(M)]
    names = list(allres[0].keys())
    summary = {}
    for n in names:
        v = np.array([a[n] for a in allres])
        summary[n] = {"auc_mean": float(v.mean()), "auc_std": float(v.std())}
        print(f"{n:14s} test ROC-AUC = {v.mean():.3f} ± {v.std():.3f}")
    with open(os.path.join(HERE, "table2.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
