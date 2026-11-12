"""Robustness of the position rule: tune each rule with the holding-cost penalty (as in the original notebook),
evaluate on the next month under both cost models; 3 optimiser seeds per month."""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pickle
import sys

import numpy as np
import pandas as pd

sys.argv = ["corrected.py"]
src = open("corrected.py", encoding="utf-8").read()
head = src.split("# ---------------- primary model")[0]
rules = src.split("# ---------------- decision rule ----------------")[1].split("close = df[")[0]
exec(head)
c = pickle.load(open("ablation_main.pkl", "rb"))
probs, folds = c["probs"], c["folds"]
pp = pickle.load(open("primary_main.pkl", "rb"))[0]
df = df.assign(pp=pp).dropna(subset=["pp"]).reset_index(drop=True)
exec(rules)

close = df["close"].values
rows = []
for arm in ("regime+stack", "base"):
    for f, (tr, va, te) in enumerate(folds):
        xv, xt = 2 * probs[arm]["val"][f] - 1, 2 * probs[arm]["test"][f] - 1
        Sv, St = close[va], close[te]
        month = str(df.open_time.iloc[te.start])[:10]
        bh = St[1:] - St[:-1]
        rows.append(dict(arm=arm, month=month, kind="buy&hold", crit="-", seed=0, eval="-",
                         **dict(zip(("ret", "vol", "sr"), stats_(bh))), trades=0.0, expo=1.0))
        for kind in ("tanh_orig", "tanh_cont", "threshold", "linear"):
            for crit in ("pnl", "sr"):
                for seed in range(3):
                    prm = tune(xv, Sv, crit, "hold", kind, seed=100 * f + seed)
                    pos = apply(xt, prm, kind)
                    for ev in ("hold", "change"):
                        r = pnl(pos, St, ev)
                        rows.append(dict(arm=arm, month=month, kind=kind, crit=crit, seed=seed, eval=ev,
                                         **dict(zip(("ret", "vol", "sr"), stats_(r))),
                                         trades=np.abs(np.diff(np.r_[0.0, pos[:-1]])).sum(), expo=np.abs(pos).mean()))
        print(arm, month, "done", flush=True)
T = pd.DataFrame(rows)
T.to_pickle("rule_check.pkl")
pd.set_option("display.width", 250)
g = T.groupby(["arm", "kind", "crit", "eval"])
print(g.agg(ret=("ret", "mean"), vol=("vol", "mean"), sr_mean=("sr", "mean"), sr_med=("sr", "median"),
            sr_pos=("sr", lambda s: f"{int((s > 0).sum())}/{len(s)}"), trades=("trades", "mean"), expo=("expo", "mean"))
      .round(2).to_string())
print(T[(T.arm == "regime+stack") & (T["eval"] != "hold")].pivot_table(index=["kind", "crit"], columns="month", values="sr").round(2).to_string())
print("DONE")
