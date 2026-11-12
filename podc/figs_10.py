# -*- coding: utf-8 -*-
"""Рисунки 2 и 3 по 10 прогонам. Отрезки: половина 95 % ДИ Стьюдента."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
W = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figs")
os.makedirs(FIG, exist_ok=True)
d = pd.concat([pd.read_csv(os.path.join(W, "final_results.csv")),
               pd.read_csv(os.path.join(W, "extra_results.csv"))], ignore_index=True)
d = d[d.d_delivery_ratio.notna()].copy()
d["key"] = d["name"].str.rsplit("_s", n=1).str[0]
d["mb"] = (d.d_bytes_data.fillna(0) + d.d_bytes_ctrl.fillna(0)) / 1e6

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11})
comma = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))


def mc(key, col):
    v = d.loc[d.key == key, col].astype(float).to_numpy()
    v = v[~np.isnan(v)]
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return float(v.mean()), float(h)


ROUT = ["PoDCRouter", "EpidemicRouter", "MaxPropRouter", "ProphetRouter",
        "SprayAndWaitRouter", "FirstContactRouter", "DirectDeliveryRouter"]
LBL = {"PoDCRouter": "Предложенный протокол", "EpidemicRouter": "Epidemic", "MaxPropRouter": "MaxProp",
       "ProphetRouter": "PRoPHET", "SprayAndWaitRouter": "Spray and Wait",
       "FirstContactRouter": "First Contact", "DirectDeliveryRouter": "Direct Delivery"}
COL = ["#1f4e79", "#7f7f7f", "#a5a5a5", "#c55a11", "#ed7d31", "#f4b183", "#e7e6e6"]
NS = (50, 100, 150)

fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.5))
width = 0.11
for i, r in enumerate(ROUT):
    xs = [j + (i - 3) * width for j in range(3)]
    dv = [mc(f"F1_{r}_{n}", "d_delivery_ratio") for n in NS]
    mv = [mc(f"F1_{r}_{n}", "mb") for n in NS]
    axs[0].bar(xs, [v[0] for v in dv], width, yerr=[v[1] for v in dv], capsize=1.5,
               color=COL[i], edgecolor="black", linewidth=0.4, label=LBL[r], error_kw={"elinewidth": 0.6})
    axs[1].bar(xs, [v[0] for v in mv], width, yerr=[v[1] for v in mv], capsize=1.5,
               color=COL[i], edgecolor="black", linewidth=0.4, error_kw={"elinewidth": 0.6})
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
fig.savefig(os.path.join(FIG, "fig2_base_10.png"), dpi=300)
plt.close(fig)

ATT = [("F1_EpidemicRouter_100", "Epidemic, без атаки"), ("F2_Epi_bh", "Epidemic, поглотители"),
       ("F1_PoDCRouter_100", "Базовый режим, без атаки"), ("F2_PoDC_bh", "Базовый режим, поглотители"),
       ("F2_PoDC_promo", "Базовый режим, самонакрутка"), ("F2_PoDC_spoof", "Базовый режим, подделка личности"),
       ("F1_PoDCstrict_100", "Строгий режим, без атаки"), ("F2_strict_bh", "Строгий режим, поглотители"),
       ("F2_strict_promo", "Строгий режим, самонакрутка")]
cols = ["#bfbfbf", "#7f7f7f", "#bdd7ee", "#2e75b6", "#1f4e79", "#203864", "#f8cbad", "#ed7d31", "#c55a11"]
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.6), sharey=True)
for ax, metric, xl in ((axs[0], "d_delivery_ratio", "Доля доставленных сообщений"),
                       (axs[1], "d_absorbed_by_malicious", "Доля передач данных злоумышленникам")):
    vals = [mc(k, metric) for k, _ in ATT]
    ax.barh(range(len(ATT)), [v[0] for v in vals], xerr=[v[1] for v in vals], capsize=1.5, color=cols,
            edgecolor="black", linewidth=0.4, error_kw={"elinewidth": 0.6})
    ax.set_xlabel(xl, fontsize=10)
    ax.xaxis.set_major_formatter(comma)
    ax.grid(axis="x", alpha=0.3)
axs[0].set_yticks(range(len(ATT)))
axs[0].set_yticklabels([lab for _, lab in ATT], fontsize=9)
axs[0].invert_yaxis()
axs[0].set_title("а", fontsize=11)
axs[1].set_title("б", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig3_attacks_10.png"), dpi=300)
plt.close(fig)
print("figs ok")
