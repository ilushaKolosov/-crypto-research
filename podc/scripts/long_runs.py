from runner import *
from final_runs_cfg import att, STRICT, PROMO

jobs = []
for s in (1, 2, 3, 4, 5):
    e = att("SybilPoDCRouter", dict(STRICT, **PROMO, **{"SybilPoDCRouter.attackStop": "1800",
                                                          "Scenario.endTime": "9000"}))
    jobs.append((f"L_strict_promostop_s{s}", "PoDCRouter", 100, s, e))
run_all(jobs, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "long_results.csv"), 5)
print("DONE", flush=True)
