# -*- coding: utf-8 -*-
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import leak_model as exp_bi

rows = {"dwt_full": [], "kf_smoother": [], "kf_filter": [], "dwt_causal": []}
acc = {k: [] for k in rows}
for s in range(50):
    x, y = exp_bi.simulate(1000 + s)
    kf, ks = exp_bi.kalman_local_level(y, exp_bi.SIG ** 2, exp_bi.NOISE ** 2)
    series = {"dwt_full": exp_bi.dwt_full(y), "kf_smoother": ks, "kf_filter": kf}
    if s < 10:
        series["dwt_causal"] = exp_bi.dwt_causal(y)
    for k, d in series.items():
        dd = np.diff(d)[:-1]
        fx = np.diff(x)[1:]
        rows[k].append(np.corrcoef(dd, fx)[0, 1])
        acc[k].append(np.mean(np.sign(dd) == np.sign(fx)))
for k in rows:
    rho = float(np.mean(rows[k]))
    pred = 0.5 + np.arcsin(rho) / np.pi
    print(f"{k:12s} rho={rho:.4f} Sheppard ACC={pred*100:.1f}% empirical ACC={np.mean(acc[k])*100:.1f}%")
