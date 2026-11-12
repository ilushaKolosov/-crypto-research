from runner import *
from final_runs_cfg import att, STRICT, PROMO

BASE = ["EpidemicRouter", "ProphetRouter", "SprayAndWaitRouter", "MaxPropRouter",
        "FirstContactRouter", "DirectDeliveryRouter", "PoDCRouter"]
NOFILTER = {"PoDCRouter.dropThreshold": "1.0"}
TTL30 = {"Group.msgTtl": "30"}

jobs = []
for s in (6, 7, 8, 9, 10):
    for n in (50, 100, 150):
        for r in BASE:
            jobs.append((f"F1_{r}_{n}_s{s}", r, n, s, {}))
    jobs.append((f"F1_PoDCstrict_100_s{s}", "PoDCRouter", 100, s, dict(STRICT)))
    jobs.append((f"F2_Epi_bh_s{s}", "EpidemicRouter", 100, s, att("BlackholeEpidemicRouter")))
    jobs.append((f"F2_PoDC_bh_s{s}", "PoDCRouter", 100, s, att("SybilPoDCRouter")))
    jobs.append((f"F2_PoDC_promo_s{s}", "PoDCRouter", 100, s, att("SybilPoDCRouter", PROMO)))
    jobs.append((f"F2_strict_bh_s{s}", "PoDCRouter", 100, s, att("SybilPoDCRouter", STRICT)))
    jobs.append((f"F2_strict_promo_s{s}", "PoDCRouter", 100, s, att("SybilPoDCRouter", dict(STRICT, **PROMO))))
    jobs.append((f"F2_strict_promostop_s{s}", "PoDCRouter", 100, s,
                 att("SybilPoDCRouter", dict(STRICT, **PROMO, **{"SybilPoDCRouter.attackStop": "1800"}))))
    jobs.append((f"F2_PoDC_spoof_s{s}", "PoDCRouter", 100, s,
                 att("SybilPoDCRouter", dict(PROMO, **{"SybilPoDCRouter.spoofIdentity": "true"}))))
for s in range(1, 11):
    jobs.append((f"A_nofilter_100_s{s}", "PoDCRouter", 100, s, dict(NOFILTER)))
    jobs.append((f"A_nofilter_bh_s{s}", "PoDCRouter", 100, s, att("SybilPoDCRouter", NOFILTER)))
    for r in BASE:
        jobs.append((f"T30_{r}_100_s{s}", r, 100, s, dict(TTL30)))

print(len(jobs), "jobs", flush=True)
run_all(jobs, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "extra_results.csv"), 13)
print("DONE", flush=True)
