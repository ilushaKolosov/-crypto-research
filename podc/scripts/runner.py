import csv
import itertools
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONE = os.environ.get("THE_ONE_HOME", os.path.join(ROOT, "the-one"))
RUNS = os.path.join(ROOT, "runs")
CP = "target;lib/ECLA.jar;lib/DTNConsoleConnection.jar"

GROUPS = {50: (14, 15, 15), 100: (31, 31, 32), 150: (48, 48, 48), 126: (40, 40, 40)}


def scenario_text(name, router, n, seed, extra):
    p, c, w = GROUPS[n]
    lines = [
        f"Scenario.name = {name}",
        "Scenario.endTime = 3600",
        f"Group.router = {router}",
        f"Group1.nrofHosts = {p}",
        f"Group2.nrofHosts = {c}",
        f"Group3.nrofHosts = {w}",
        "Events1.interval = 4,10",
        "Events1.size = 8k,20k",
        f"Events1.hosts = 0,{n}",
        f"Events1.prefix = M{seed}_",
        f"MovementModel.rngSeed = {seed}",
        "Report.nrofReports = 2",
        "Report.report1 = MessageStatsReport",
        "Report.report2 = DataStatsReport",
        f"Report.reportDir = {RUNS.replace(os.sep, '/')}/{name}/",
    ]
    lines += [f"{k} = {v}" for k, v in extra.items()]
    return "\n".join(lines) + "\n"


def parse(path):
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path, encoding="utf-8", errors="replace"):
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def run_one(job):
    name, router, n, seed, extra = job
    d = os.path.join(RUNS, name)
    os.makedirs(d, exist_ok=True)
    data_file = os.path.join(d, f"{name}_DataStatsReport.txt")
    if not os.path.exists(data_file):
        cfg = os.path.join(d, "settings.txt")
        with open(cfg, "w", encoding="utf-8") as fh:
            fh.write(scenario_text(name, router, n, seed, extra))
        t0 = time.time()
        with open(os.path.join(d, "stdout.txt"), "w") as log:
            subprocess.run(
                ["java", "-Xmx1G", "-cp", CP, "core.DTNSim", "-b", "1", cfg],
                cwd=ONE, stdout=log, stderr=subprocess.STDOUT)
        print(f"{name}: {time.time() - t0:.0f}s", flush=True)
    row = {"name": name, "router": router, "n": n, "seed": seed}
    row.update({f"x_{k}": v for k, v in extra.items()})
    ds = parse(data_file)
    ms = parse(os.path.join(d, f"{name}_MessageStatsReport.txt"))
    row.update({f"d_{k}": v for k, v in ds.items() if k != "scenario"})
    ps = parse(os.path.join(ONE, "reports", "podc", f"{name}_summary.txt"))
    row.update({f"p_{k}": v for k, v in ps.items()})
    row.update({f"m_{k}": ms.get(k) for k in
                ("created", "delivered", "delivery_prob", "overhead_ratio",
                 "latency_avg", "hopcount_avg")})
    return row


def run_all(jobs, out_csv, workers=12):
    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(run_one, jobs))
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k[0] != "n" and k != "name", k))
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return rows
