# -*- coding: utf-8 -*-
"""Дополнение к таблице 1: разброс по 50 прогонам, t-критерий против 1/2, доля прогонов выше 1/2
и корреляция rho, вычисленная по весам линейного оператора (без латентного ряда)."""
import json
import os

import numpy as np
from scipy import stats

from leak_model import SIG, NOISE, N, simulate, dwt_full, dwt_causal, kalman_local_level, strat_metrics

HERE = os.path.dirname(os.path.abspath(__file__))


def per_run(M=50):
    keys = ["dwt_full", "kf_smoother", "dwt_causal", "kf_filter"]
    acc = {k: [] for k in keys}
    for s in range(M):
        x, y = simulate(1000 + s)
        kf, ks = kalman_local_level(y, SIG ** 2, NOISE ** 2)
        ser = {"dwt_full": dwt_full(y), "kf_smoother": ks, "dwt_causal": dwt_causal(y), "kf_filter": kf}
        for k, d in ser.items():
            acc[k].append(strat_metrics(d, y, x)[3])
    out = {}
    for k, a in acc.items():
        a = np.array(a)
        t, p = stats.ttest_1samp(a, 0.5)
        out[k] = dict(mean=float(a.mean()), sd=float(a.std(ddof=1)), se=float(a.std(ddof=1) / np.sqrt(len(a))),
                      min=float(a.min()), max=float(a.max()), n_above=int((a > 0.5).sum()), t=float(t), p=float(p),
                      z_sd=float((a.mean() - 0.5) / a.std(ddof=1)))
    return out


def linear_weights(op, t):
    """Строка весов приращения оператора в точке t: b_k = a_{t,k} - a_{t-1,k}."""
    A = np.empty((2, N))
    for k in range(N):
        e = np.zeros(N)
        e[k] = 1.0
        z = op(e)
        A[0, k], A[1, k] = z[t - 1], z[t]
    return A[1] - A[0]


def rho_from_weights(b, t):
    B = np.cumsum(b[::-1])[::-1]
    var = SIG ** 2 * np.sum(B[1:] ** 2) + NOISE ** 2 * np.sum(b ** 2)
    return float(SIG * B[t + 1] / np.sqrt(var))


def acc_from_rho(r):
    return 0.5 + np.arcsin(r) / np.pi


if __name__ == "__main__":
    res = {"per_run": per_run()}
    t = N // 2
    smoother = lambda e: kalman_local_level(e, SIG ** 2, NOISE ** 2)[1]
    filt = lambda e: kalman_local_level(e, SIG ** 2, NOISE ** 2)[0]
    for name, op in (("kf_smoother", smoother), ("kf_filter", filt)):
        b = linear_weights(op, t)
        r = rho_from_weights(b, t)
        res[f"rho_weights_{name}"] = dict(rho=r, acc=float(acc_from_rho(r)), future_weight=float(b[t + 1:].sum()))
    print(json.dumps(res, ensure_ascii=False, indent=2))
    json.dump(res, open(os.path.join(HERE, "a1_extra.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
