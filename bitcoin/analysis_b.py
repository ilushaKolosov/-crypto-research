import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
﻿src = open("analysis.py", encoding="utf-8").read()
pre = src.split("# ---------------- regimes ----------------")[0]
job = src.split("FEES = [")[1].split("tasks = [")[0]
job = "FEES = [" + job
job = job.replace('''        if kind == "threshold":
            return np.where(x >= prm["u"], 1.0, np.where(x <= prm["d"], -1.0, 0.0))
        return rule''', '''        if kind == "threshold":
            return np.where(x >= prm["u"], 1.0, np.where(x <= prm["d"], -1.0, 0.0))
        if kind == "tanh_beta":
            return np.tanh(prm["beta"] * x)
        if kind == "tanh_sym":
            return rule(x, -prm["u"], prm["u"], prm["beta"], 0.0, True)
        return rule''')
job = job.replace('''        if kind == "linear":
            prm = {"c": t.suggest_float("c", 0.1, 100, log=True)}''', '''        if kind == "linear":
            prm = {"c": t.suggest_float("c", 0.1, 100, log=True)}
        elif kind == "tanh_beta":
            prm = {"beta": t.suggest_float("beta", 0.1, 100, log=True)}
        elif kind == "tanh_sym":
            prm = {"u": t.suggest_float("u", 0.0, 0.5), "beta": t.suggest_float("beta", 0.1, 100, log=True)}''')
exec(pre)
close = df["close"].values
exec(job)
tasks = [(fee, kind, crit, f, s, "regime+stack") for fee in FEES for f in range(len(folds)) for s in range(2)
         for kind in ("tanh_beta", "tanh_sym") for crit in ("pnl", "sr")]
out = Parallel(n_jobs=14)(delayed(job)(*t) for t in tasks)
G = pd.DataFrame(out)
F = pd.concat([pd.read_pickle("fee_sweep.pkl"), G], ignore_index=True)
F.to_pickle("fee_sweep_all.pkl")
for v in ("sr", "sr_val"):
    print(v, "\n", F.pivot_table(index=["kind", "crit"], columns="fee", values=v, aggfunc="mean").round(2).to_string())
print("pos\n", F.pivot_table(index=["kind", "crit"], columns="fee", values="sr", aggfunc=lambda s: (s > 0).mean()).round(2).to_string())
print("vol\n", F.pivot_table(index=["kind", "crit"], columns="fee", values="vol", aggfunc="mean").round(0).to_string())
print("ret\n", F.pivot_table(index=["kind", "crit"], columns="fee", values="ret", aggfunc="mean").round(0).to_string())
print("DONE")
