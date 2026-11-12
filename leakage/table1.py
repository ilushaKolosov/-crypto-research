# -*- coding: utf-8 -*-
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import leak_model as exp_bi

res = exp_bi.monte_carlo(M=50)
for k, v in res.items():
    print(
        f"{k:12s} RMSE={v['rmse_mean']*1e3:.3f}e-3 red={v['rmse_reduction_pct']:.1f}% "
        f"ACC_true={v['acc_true_mean']*100:.1f}±{v['acc_true_std']*100:.1f} "
        f"Sharpe={v['sharpe_mean']:+.1f}±{v['sharpe_std']:.1f}"
    )
with open(os.path.join(HERE, "table1.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
