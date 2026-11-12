"""Tables and the capital-return figure from a3_capital.pkl."""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pickle

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

F = pd.read_pickle("a3_capital.pkl")
months = pickle.load(open("ablation_main.pkl", "rb"))["R"].groupby("fold").month.first().tolist()
F["start"] = F.fold.map(dict(enumerate(months)))
S = F[F.crit.isin(["sr", "-"])]
pd.set_option("display.width", 250)

for fee in (0.0, 0.01):
    t = S[S.fee == fee].pivot_table(index="kind", columns="start", values="sr", aggfunc="mean")
    print(f"\nSR by test month, k={fee}%, SR criterion\n", t.round(2).to_string())
    for other in ("tanh_beta", "linear"):
        d = (t.loc[other] - t.loc["sign"]).values
        tt = stats.ttest_1samp(d, 0.0)
        print(f"  {other} - sign: mean {d.mean():.2f}, se {d.std(ddof=1) / np.sqrt(6):.2f}, t={tt.statistic:.2f}, "
              f"p(two-sided)={tt.pvalue:.4f}, wins {int((d > 0).sum())}/6, sign-test p(two-sided)={2 * 0.5 ** 6 if (d > 0).all() else np.nan}")

g = S[S.fee == 0.01].groupby("kind")
print("\nk=0.01% SR crit: mean ret%, vol%, expo, turnover, SR, SR_val, gap, Lo-adj SR")
print(pd.DataFrame({"ret": 100 * g.ret.mean(), "vol": 100 * g.vol.mean(), "expo": g.expo.mean(),
                    "turnover": g.turnover.mean(), "sr": g.sr.mean(), "sr_val": g.sr_val.mean(),
                    "gap": g.sr_val.mean() - g.sr.mean(), "sr_lo": (S[S.fee == 0.01].sr / S[S.fee == 0.01].lo).groupby(S[S.fee == 0.01].kind).mean()}).round(2).to_string())
P = F[(F.fee == 0.01) & (F.crit == "pnl")].groupby("kind")
print("\nk=0.01% PnL crit: ret%, vol%, SR, expo\n",
      pd.DataFrame({"ret": 100 * P.ret.mean(), "vol": 100 * P.vol.mean(), "sr": P.sr.mean(), "expo": P.expo.mean()}).round(2).to_string())

NAMES = {"sign": "По знаку", "tanh_beta": "tanh(βx)", "linear": "Линейное с насыщением",
         "threshold": "Пороговое", "tanh_sym": "Симметричное", "full": "Полное (2)"}
STY = {"sign": ("black", "o", "-"), "tanh_beta": ("tab:red", "s", "-"), "linear": ("tab:orange", "^", "--"),
       "threshold": ("tab:blue", "v", ":"), "tanh_sym": ("tab:green", "D", "-."), "full": ("tab:purple", "x", "--")}
m = S.pivot_table(index="fee", columns="kind", values="sr", aggfunc="mean")
fig, ax = plt.subplots(figsize=(7.0, 3.9))
for k, nm in NAMES.items():
    c, mk, ls = STY[k]
    ax.plot(m.index, m[k], color=c, marker=mk, ls=ls, lw=1.4, ms=4.5, label=nm)
ax.axhline(0, color="0.5", lw=0.8)
ax.set_xlabel("Ставка издержек k, % от оборота")
ax.set_ylabel("Средний SR на тесте (6 мес.)")
fmt = lambda v, _: f"{v:g}".replace(".", ",")
ax.xaxis.set_major_formatter(FuncFormatter(fmt))
ax.yaxis.set_major_formatter(FuncFormatter(fmt))
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, ncol=2, loc="upper right")
fig.tight_layout()
fig.savefig("a3_fig2_capital.png", dpi=300)
print("\nfig saved; mean SR table\n", m.round(2).to_string())
