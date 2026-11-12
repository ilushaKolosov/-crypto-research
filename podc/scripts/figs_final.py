import csv
import glob
import json
import math
import os
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter

W = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIG = os.path.join(W, "figs")
os.makedirs(FIG, exist_ok=True)
S = json.load(open(os.path.join(W, "summary.json")))
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11})
comma = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))

ROUT = ["PoDCRouter", "EpidemicRouter", "MaxPropRouter", "ProphetRouter",
        "SprayAndWaitRouter", "FirstContactRouter", "DirectDeliveryRouter"]
LBL = {"PoDCRouter": "Предложенный протокол", "EpidemicRouter": "Epidemic", "MaxPropRouter": "MaxProp",
       "ProphetRouter": "PRoPHET", "SprayAndWaitRouter": "Spray and Wait",
       "FirstContactRouter": "First Contact", "DirectDeliveryRouter": "Direct Delivery"}
COL = ["#1f4e79", "#7f7f7f", "#a5a5a5", "#c55a11", "#ed7d31", "#f4b183", "#e7e6e6"]
NS = (50, 100, 150)


def mb(key):
    v = S[key]
    return (v["d_bytes_data"][0] + v["d_bytes_ctrl"][0]) / 1e6


# Figure 2: delivery and traffic
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.5))
width = 0.11
for i, r in enumerate(ROUT):
    xs = [j + (i - 3) * width for j in range(3)]
    dv = [S[f"F1_{r}_{n}"]["d_delivery_ratio"] for n in NS]
    axs[0].bar(xs, [v[0] for v in dv], width, yerr=[v[1] for v in dv], capsize=1.5,
               color=COL[i], edgecolor="black", linewidth=0.4, label=LBL[r], error_kw={"elinewidth": 0.6})
    axs[1].bar(xs, [mb(f"F1_{r}_{n}") for n in NS], width, color=COL[i], edgecolor="black", linewidth=0.4)
for ax, yl in ((axs[0], "Доля доставленных сообщений"), (axs[1], "Передано по сети, МБ")):
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"N = {n}" for n in NS])
    ax.set_ylabel(yl)
    ax.yaxis.set_major_formatter(comma)
    ax.grid(axis="y", alpha=0.3)
axs[0].set_title("а", fontsize=11)
axs[1].set_title("б", fontsize=11)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, ncol=4, fontsize=8.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.savefig(os.path.join(FIG, "fig2_base.png"), dpi=300)
plt.close(fig)

# Figure 3: attacks
ATT = [("F1_EpidemicRouter_100", "Epidemic, без атаки"), ("F2_Epi_bh", "Epidemic, поглотители"),
       ("F1_PoDCRouter_100", "Базовый режим, без атаки"), ("F2_PoDC_bh", "Базовый режим, поглотители"),
       ("F2_PoDC_promo", "Базовый режим, самонакрутка"), ("F2_PoDC_spoof", "Базовый режим, подделка личности"),
       ("F1_PoDCstrict_100", "Строгий режим, без атаки"), ("F2_strict_bh", "Строгий режим, поглотители"),
       ("F2_strict_promo", "Строгий режим, самонакрутка")]
cols = ["#bfbfbf", "#7f7f7f", "#bdd7ee", "#2e75b6", "#1f4e79", "#203864", "#f8cbad", "#ed7d31", "#c55a11"]
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.6), sharey=True)
for ax, metric, xl in ((axs[0], "d_delivery_ratio", "Доля доставленных сообщений"),
                       (axs[1], "d_absorbed_by_malicious", "Доля передач данных злоумышленникам")):
    vals = [S[k][metric] for k, _ in ATT]
    ax.barh(range(len(ATT)), [v[0] for v in vals], xerr=[v[1] for v in vals], capsize=1.5, color=cols,
            edgecolor="black", linewidth=0.4, error_kw={"elinewidth": 0.6})
    ax.set_xlabel(xl, fontsize=10)
    ax.xaxis.set_major_formatter(comma)
    ax.grid(axis="x", alpha=0.3)
axs[0].set_yticks(range(len(ATT)))
axs[0].set_yticklabels([l for _, l in ATT], fontsize=9)
axs[0].invert_yaxis()
axs[0].set_title("а", fontsize=11)
axs[1].set_title("б", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig3_attacks_v2.png"), dpi=300)
plt.close(fig)

# Figure 4: long run, attack stops at 1800 s
LAM, STOP, BIN = 0.001, 1800, 600
agg = defaultdict(lambda: [0.0, 0.0, [], []])
for f in glob.glob(os.path.join(W, "the-one", "reports", "podc", "L_strict_promostop_s*_series.csv")):
    for r in csv.DictReader(open(f)):
        b = int(float(r["time"])) // BIN * BIN
        a = agg[b]
        a[0] += float(r["forwards"]); a[1] += float(r["forwards_to_attackers"])
        a[2].append(float(r["attacker_work"])); a[3].append(float(r["honest_work"]))
bs = sorted(agg)
mid = [b + BIN / 2 for b in bs]
frac = [agg[b][1] / agg[b][0] for b in bs]
aw = [st.mean(agg[b][2]) for b in bs]
hw = [st.mean(agg[b][3]) for b in bs]
w_stop = aw[bs.index(STOP)] if STOP in bs else aw[3]
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.3))
axs[0].semilogy(mid, aw, "o-", color="#1f4e79", ms=4, label="злоумышленник")
axs[0].semilogy(mid, hw, "s--", color="#7f7f7f", ms=4, label="честный узел")
anchor = bs.index(5400)
tt = [t for t in range(3000, 9001, 100)]
axs[0].semilogy(tt, [aw[anchor] * math.exp(-LAM * (t - mid[anchor])) for t in tt], ":", color="black",
                label="наклон по формуле (8)")
axs[0].axvline(STOP, color="#c00000", linewidth=0.8)
axs[0].set_xlabel("Время моделирования, с")
axs[0].set_ylabel("Работа в реестрах честных узлов")
axs[0].legend(fontsize=8, frameon=False)
axs[0].grid(alpha=0.3, which="both")
axs[0].set_title("а", fontsize=11)
axs[1].bar(mid, frac, BIN * 0.8, color="#ed7d31", edgecolor="black", linewidth=0.4)
axs[1].axvline(STOP, color="#c00000", linewidth=0.8)
axs[1].set_xlabel("Время моделирования, с")
axs[1].set_ylabel("Доля передач злоумышленникам")
axs[1].yaxis.set_major_formatter(comma)
axs[1].grid(axis="y", alpha=0.3)
axs[1].set_title("б", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig4_decay_v3.png"), dpi=300)
plt.close(fig)
json.dump({"bins": bs, "frac": frac, "attacker_work": aw, "honest_work": hw},
          open(os.path.join(W, "long_series.json"), "w"))

# Figure 1: scheme
fig, ax = plt.subplots(figsize=(7.4, 3.0))
ax.set_xlim(0, 10)
ax.set_ylim(0, 4.2)
ax.axis("off")
nodes = [("S", "отправитель", 0.9), ("R1", "ретранслятор", 3.6), ("R2", "ретранслятор", 6.3), ("D", "получатель", 9.0)]
for name, role, x in nodes:
    ax.add_patch(plt.Circle((x, 3.3), 0.38, color="#bdd7ee", ec="black", lw=0.8))
    ax.text(x, 3.3, name, ha="center", va="center", fontsize=12, weight="bold")
    ax.text(x, 2.72, role, ha="center", va="center", fontsize=9)
for (_, _, x1), (_, _, x2) in zip(nodes, nodes[1:]):
    ax.annotate("", xy=(x2 - 0.45, 3.3), xytext=(x1 + 0.45, 3.3),
                arrowprops=dict(arrowstyle="->", lw=1.2))
ents = [("e0: S, подпись S,\nсвязь с началом (2)", 0.25), ("e1: R1, подпись R1,\nхеш e0", 2.8),
        ("e2: R2, подпись R2,\nхеш e1", 5.35), ("e3: D, подпись D,\nхеш e2", 7.9)]
for txt, x in ents:
    ax.add_patch(FancyBboxPatch((x, 1.55), 1.85, 0.75, boxstyle="round,pad=0.03", fc="#f2f2f2", ec="black", lw=0.6))
    ax.text(x + 0.925, 1.925, txt, ha="center", va="center", fontsize=8)
ax.annotate("", xy=(1.0, 0.75), xytext=(9.0, 0.75), arrowprops=dict(arrowstyle="->", lw=1.2, ls="--"))
ax.text(5.0, 0.95, "подтверждение доставки: цепочка e0...e3 распространяется по сети", ha="center", fontsize=9)
ax.text(5.0, 0.3, "каждый узел проверяет цепочку сам и начисляет вклад: R2 получает 1, R1 получает 1/2, S и D не получают",
        ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig1_scheme_v2.png"), dpi=300)
plt.close(fig)
print("figs ok")


