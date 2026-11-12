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

W = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIG = os.path.join(W, "figs")
os.makedirs(FIG, exist_ok=True)
rows = list(csv.DictReader(open(os.path.join(W, "final_results.csv"), encoding="utf-8")))


def group(prefix):
    return [r for r in rows if r["name"].rsplit("_s", 1)[0] == prefix]


def ms(rs, col):
    v = [float(r[col]) for r in rs if r.get(col) not in (None, "", "NaN")]
    if not v:
        return float("nan"), float("nan")
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)


METRICS = ["d_delivery_ratio", "d_overhead_data", "d_overhead_total", "d_latency_avg",
           "d_hopcount_avg", "d_absorbed_by_malicious", "d_bytes_ctrl", "d_bytes_data",
           "p_invalid_acks", "p_acks_processed", "p_decrypted", "p_delivered",
           "p_avg_proof_bytes", "p_gini_work"]
summary = {}
for key in sorted({r["name"].rsplit("_s", 1)[0] for r in rows}):
    rs = group(key)
    summary[key] = {m: ms(rs, m) for m in METRICS}
    summary[key]["runs"] = len(rs)
    bc, bd = ms(rs, "d_bytes_ctrl")[0], ms(rs, "d_bytes_data")[0]
    summary[key]["ctrl_byte_share"] = bc / (bc + bd) if bd == bd and bd > 0 else float("nan")
json.dump(summary, open(os.path.join(W, "summary.json"), "w"), indent=1)

ROUT = ["PoDCRouter", "EpidemicRouter", "MaxPropRouter", "ProphetRouter",
        "SprayAndWaitRouter", "FirstContactRouter", "DirectDeliveryRouter"]
LBL = {"PoDCRouter": "PoDC", "EpidemicRouter": "Epidemic", "MaxPropRouter": "MaxProp",
       "ProphetRouter": "PRoPHET", "SprayAndWaitRouter": "Spray and Wait",
       "FirstContactRouter": "First Contact", "DirectDeliveryRouter": "Direct Delivery"}
with open(os.path.join(W, "table_base.txt"), "w", encoding="utf-8") as fh:
    for n in (50, 100, 150):
        for r in ROUT:
            s = summary[f"F1_{r}_{n}"]
            fh.write(f"{n}\t{LBL[r]}\t" + "\t".join(
                f"{s[m][0]:.4f}±{s[m][1]:.4f}" for m in
                ("d_delivery_ratio", "d_overhead_data", "d_overhead_total", "d_latency_avg", "d_hopcount_avg"))
                + f"\tctrl_bytes={s['ctrl_byte_share']:.4f}\n")
    for k in sorted(summary):
        if not k.startswith("F1_"):
            s = summary[k]
            fh.write(k + "\t" + "\t".join(f"{m}={s[m][0]:.4f}±{s[m][1]:.4f}" for m in METRICS
                                         if s[m][0] == s[m][0]) + "\n")

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11})
NS = (50, 100, 150)
for metric, ylabel, fname in (("d_delivery_ratio", "Доля доставленных сообщений", "fig_delivery.png"),
                              ("d_overhead_data", "Пересылок данных на одно доставленное", "fig_overhead.png"),
                              ("d_latency_avg", "Средняя задержка, с", "fig_latency.png")):
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    width = 0.11
    for i, r in enumerate(ROUT):
        vals = [summary[f"F1_{r}_{n}"][metric] for n in NS]
        xs = [j + (i - 3) * width for j in range(3)]
        ax.bar(xs, [v[0] for v in vals], width, yerr=[v[1] for v in vals], capsize=2,
               label=LBL[r], color=["#1f4e79", "#7f7f7f", "#a5a5a5", "#c55a11", "#ed7d31", "#f4b183", "#d9d9d9"][i],
               edgecolor="black", linewidth=0.4)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"N = {n}" for n in NS])
    ax.set_ylabel(ylabel)
    ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), dpi=200)
    plt.close(fig)

# attack bar chart
ATT = [("F1_EpidemicRouter_100", "Epidemic,\nбез атаки"), ("F2_Epi_bh", "Epidemic,\nпоглотители"),
       ("F1_PoDCRouter_100", "PoDC,\nбез атаки"), ("F2_PoDC_bh", "PoDC,\nпоглотители"),
       ("F2_PoDC_promo", "PoDC,\nпоглотители +\nсамонакрутка"),
       ("F1_PoDCstrict_100", "PoDC строгий,\nбез атаки"), ("F2_strict_bh", "PoDC строгий,\nпоглотители"),
       ("F2_strict_promo", "PoDC строгий,\nпоглотители +\nсамонакрутка")]
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.6))
for ax, metric, ylabel in ((axs[0], "d_delivery_ratio", "Доля доставленных сообщений"),
                           (axs[1], "d_absorbed_by_malicious", "Доля передач данных поглотителям")):
    vals = [summary[k][metric] for k, _ in ATT]
    ax.barh(range(len(ATT)), [v[0] for v in vals], xerr=[v[1] for v in vals], capsize=2,
            color=["#a5a5a5", "#7f7f7f", "#9dc3e6", "#2e75b6", "#1f4e79", "#f4b183", "#ed7d31", "#c55a11"],
            edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(len(ATT)))
    ax.set_yticklabels([l for _, l in ATT] if ax is axs[0] else [""] * len(ATT), fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(ylabel, fontsize=9)
    ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_attacks.png"), dpi=200)
plt.close(fig)

# time series: strict mode, attack stops at 1800 s
series = defaultdict(lambda: defaultdict(list))
for f in glob.glob(os.path.join(W, "the-one", "reports", "podc", "F2_strict_promostop_s*_series.csv")):
    for r in csv.DictReader(open(f)):
        t = int(float(r["time"]))
        for c in ("attacker_work", "honest_work", "forwards", "forwards_to_attackers"):
            series[t][c].append(float(r[c]))
ts = sorted(series)
BIN5 = 300
agg = defaultdict(lambda: [0.0, 0.0, [], []])
for t in ts:
    b = t // BIN5 * BIN5
    agg[b][0] += sum(series[t]["forwards"])
    agg[b][1] += sum(series[t]["forwards_to_attackers"])
    agg[b][2].append(st.mean(series[t]["attacker_work"]))
    agg[b][3].append(st.mean(series[t]["honest_work"]))
bs = sorted(agg)
frac = [agg[b][1] / agg[b][0] if agg[b][0] else 0 for b in bs]
aw = [st.mean(agg[b][2]) for b in bs]
hw = [st.mean(agg[b][3]) for b in bs]
fig, ax1 = plt.subplots(figsize=(7.0, 3.4))
ax1.bar([b + BIN5 / 2 for b in bs], frac, BIN5 * 0.8, color="#ed7d31", edgecolor="black", linewidth=0.4,
        label="Доля передач данных атакующим")
ax1.set_xlabel("Время моделирования, с")
ax1.set_ylabel("Доля передач атакующим")
ax2 = ax1.twinx()
ax2.plot([b + BIN5 / 2 for b in bs], aw, "o-", color="#1f4e79", label="Работа атакующего глазами честных узлов")
ax2.plot([b + BIN5 / 2 for b in bs], hw, "s--", color="#7f7f7f", label="Работа честного узла")
ax2.set_ylabel("Ненормированная работа W")
ax1.axvline(1800, color="black", linestyle=":", linewidth=1)
ax1.text(1820, max(frac) * 0.95, "прекращение атаки", fontsize=9)
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, fontsize=8, frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_decay.png"), dpi=200)
plt.close(fig)
with open(os.path.join(W, "series_strict_promostop.txt"), "w") as fh:
    for b, f_, a, h in zip(bs, frac, aw, hw):
        fh.write(f"{b}\t{f_:.4f}\t{a:.4f}\t{h:.4f}\n")
print("ok")
