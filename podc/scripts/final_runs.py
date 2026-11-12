from runner import *

SEEDS = (1, 2, 3, 4, 5)
BASE = ["EpidemicRouter", "ProphetRouter", "SprayAndWaitRouter", "MaxPropRouter",
        "FirstContactRouter", "DirectDeliveryRouter", "PoDCRouter"]
STRICT = {"PoDCRouter.minContribution": "0.01", "PoDCRouter.explore": "0.1"}


def att(router, extra=None):
    e = {"Scenario.nrofHostGroups": "7", "Group7.groupID": "x", "Group7.nrofHosts": "10",
         "Group7.router": router, "Events1.hosts": "0,100", "DataStatsReport.maliciousPrefix": "x"}
    e.update(extra or {})
    return e


PROMO = {"SybilPoDCRouter.fakeAckInterval": "10"}

jobs = []
for n in (50, 100, 150):
    for r in BASE:
        for s in SEEDS:
            jobs.append((f"F1_{r}_{n}_s{s}", r, n, s, {}))
for s in SEEDS:
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
    for d in ("0.3", "0.7"):
        jobs.append((f"F3_delta{d}_s{s}", "PoDCRouter", 100, s, {"PoDCRouter.dropThreshold": d}))
    for lam in ("0.0001", "0.01"):
        jobs.append((f"F3_lambda{lam}_s{s}", "PoDCRouter", 100, s, {"PoDCRouter.decayLambda": lam}))
    jobs.append((f"F3_ackUnl_s{s}", "PoDCRouter", 100, s, {"PoDCRouter.ackBudget": "0"}))

print(len(jobs), "jobs", flush=True)
run_all(jobs, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "final_results.csv"), 13)
print("DONE", flush=True)
