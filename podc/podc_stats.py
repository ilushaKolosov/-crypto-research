"""PoDC: 10 seeds, 95% CI, paired tests by seed, ablation (filter off), TTL = 30 min sensitivity."""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

W = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
D = pd.concat([pd.read_csv(os.path.join(W, "final_results.csv")), pd.read_csv(os.path.join(W, "extra_results.csv"))],
              ignore_index=True).drop_duplicates("name", keep="last")
# F1_MaxPropRouter_150_s4: MaxProp comparator crashed the JVM (TimSort). The report files are empty.
D = D[D["d_delivery_ratio"].notna()].copy()
D["key"] = D["name"].str.rsplit("_s", n=1).str[0]
D["mb"] = (D["d_bytes_data"].fillna(0) + D["d_bytes_ctrl"].fillna(0)) / 1e6
D["fwd_per_del"] = D["d_overhead_data"]
out = {}


def ci(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
    return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0, ci=float(h), n=int(len(v)))


def summ(key, cols=("d_delivery_ratio", "mb", "d_latency_avg", "d_hopcount_avg", "d_absorbed_by_malicious",
                    "fwd_per_del", "d_overhead_total", "p_invalid_acks", "p_avg_proof_bytes")):
    g = D[D.key == key]
    return {c: ci(g[c]) for c in cols if c in g and g[c].notna().any()}


def paired(ka, kb, col):
    a = D[D.key == ka].set_index("seed")[col]
    b = D[D.key == kb].set_index("seed")[col]
    s = a.index.intersection(b.index)
    d = (a[s] - b[s]).values
    t = stats.ttest_rel(a[s], b[s])
    w = stats.wilcoxon(a[s], b[s]) if np.any(d != 0) else None
    return dict(n=int(len(s)), diff=float(d.mean()), ci=float(stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))),
                rel=float(d.mean() / b[s].mean()), p_t=float(t.pvalue), p_w=float(w.pvalue) if w else 1.0,
                wins=int((d > 0).sum()))


ROUT = ["PoDCRouter", "EpidemicRouter", "MaxPropRouter", "ProphetRouter", "SprayAndWaitRouter",
        "FirstContactRouter", "DirectDeliveryRouter"]
for n in (50, 100, 150):
    for r in ROUT:
        out[f"F1_{r}_{n}"] = summ(f"F1_{r}_{n}")
    for r in ("ProphetRouter", "EpidemicRouter", "MaxPropRouter", "SprayAndWaitRouter"):
        for col in ("d_delivery_ratio", "mb", "d_latency_avg", "d_hopcount_avg"):
            out[f"PAIR_{n}_PoDC_vs_{r}_{col}"] = paired(f"F1_PoDCRouter_{n}", f"F1_{r}_{n}", col)
for k in ("F1_PoDCstrict_100", "F2_Epi_bh", "F2_PoDC_bh", "F2_PoDC_promo", "F2_strict_bh", "F2_strict_promo",
          "F2_strict_promostop", "F2_PoDC_spoof", "A_nofilter_100", "A_nofilter_bh"):
    out[k] = summ(k)
for r in ROUT:
    out[f"T30_{r}_100"] = summ(f"T30_{r}_100")
for r in ("ProphetRouter", "EpidemicRouter", "MaxPropRouter"):
    for col in ("d_delivery_ratio", "mb"):
        out[f"PAIR_T30_PoDC_vs_{r}_{col}"] = paired("T30_PoDCRouter_100", f"T30_{r}_100", col)
for a, b in (("F1_PoDCRouter_100", "A_nofilter_100"), ("A_nofilter_100", "F1_EpidemicRouter_100"),
             ("F1_PoDCstrict_100", "F1_PoDCRouter_100"), ("F2_PoDC_bh", "A_nofilter_bh"),
             ("F2_strict_bh", "F2_PoDC_bh"), ("F2_strict_promo", "F1_PoDCstrict_100"), ("F2_PoDC_promo", "F1_PoDCRouter_100"),
             ("F2_PoDC_bh", "F1_PoDCRouter_100"), ("F2_Epi_bh", "F1_EpidemicRouter_100")):
    for col in ("d_delivery_ratio", "mb", "d_absorbed_by_malicious"):
        if D[D.key == a][col].notna().any() and D[D.key == b][col].notna().any():
            out[f"PAIR_{a}_vs_{b}_{col}"] = paired(a, b, col)
for k in ("F3_delta0.3", "F3_delta0.7", "F3_lambda0.0001", "F3_lambda0.01", "F3_ackUnl"):
    out[k] = summ(k)
json.dump(out, open(os.path.join(W, "podc_stats.json"), "w", encoding="utf-8"), indent=1)


def f(x, nd=3):
    return f"{x['mean']:.{nd}f} ± {x['ci']:.{nd}f} (n={x['n']})"


for k, v in out.items():
    if k.startswith("PAIR"):
        print(f"{k}: diff {v['diff']:+.4f} ± {v['ci']:.4f} ({100 * v['rel']:+.1f}%), p_t={v['p_t']:.4f}, p_w={v['p_w']:.4f}, wins {v['wins']}/{v['n']}")
    else:
        print(k, " | ".join(f"{c}: {f(x, 3 if c != 'mb' else 0)}" for c, x in v.items()))
