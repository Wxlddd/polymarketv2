"""Parallel parameter sweep over recorded tick files.

Usage:
    uv run python scratch/sweep_backtest.py [--set train|test] [--configs name,name,...] [--workers 6]

Each (config, file) pair runs `run_backtest.py` in its own process with env-var overrides and
its own LOG_DIR, so parallel runs never collide. Results are appended to scratch/sweep_results.csv
and a pivot (config x file -> net_pnl) is printed.
"""
import argparse, csv, glob, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FILES = {
    "train": [
        "logs/2026-05-28/merton/live_1779986166/ticks.parquet",  # 128k ticks, 6 cycles
        "logs/2026-05-27/merton/live_1779887341/ticks.parquet",  # 213k
        "logs/2026-05-27/merton/live_1779888432/ticks.parquet",  # 242k
    ],
    "test": [
        "logs/2026-05-27/merton/live_1779837191/ticks.parquet",  # 388k, 13 settles
        "logs/2026-05-26/merton/live_1779826699/ticks.parquet",  # 461k
    ],
}

CONFIGS = {
    "baseline":      {},
    "maker_only":    {"TAKER_ENABLED": "False"},
    "eps04":         {"MM_TAKER_EDGE_EPSILON": "0.04"},
    "eps04_k05":     {"MM_TAKER_EDGE_EPSILON": "0.04", "KELLY_FRACTION": "0.05"},
    "ema15":         {"EMA_HALFLIFE_SEC": "15"},
    "no_hawkes":     {"HAWKES_KAPPA_SELF": "0", "HAWKES_KAPPA_CROSS": "0"},
    "inv200":        {"MM_MAX_INVENTORY": "200"},
    "gamma10":       {"MM_RISK_AVERSION": "10"},
    "wide_quotes":   {"MM_MIN_FEE_BUFFER": "0.015", "MM_TOXICITY_BUFFER": "0.015"},
    "legacy":        {"STRATEGY_NAME": "legacy_merton"},
    "conservative":  {"TAKER_ENABLED": "False", "MM_MIN_FEE_BUFFER": "0.015", "MM_TOXICITY_BUFFER": "0.015", "MM_RISK_AVERSION": "10"},
    "unwind03":      {"MM_UNWIND_THRESHOLD": "0.03"},
}


def run_job(name, overrides, file_path, seed, out_root):
    log_dir = os.path.join(out_root, name, os.path.basename(os.path.dirname(file_path)))
    os.makedirs(log_dir, exist_ok=True)
    env = {**os.environ, **overrides, "LOG_DIR": log_dir, "BACKTEST_SEED": str(seed), "PYTHONIOENCODING": "utf-8"}
    t0 = time.time()
    summaries = glob.glob(os.path.join(log_dir, "**", "summary.json"), recursive=True)
    rc = 0
    if not summaries:  # resume: a finished job leaves a summary.json behind, skip it
        with open(os.path.join(log_dir, "stdout.log"), "w", encoding="utf-8") as out:
            rc = subprocess.run([sys.executable, "run_backtest.py", "--file", file_path], cwd=ROOT, env=env,
                                stdout=out, stderr=subprocess.STDOUT).returncode
        summaries = glob.glob(os.path.join(log_dir, "**", "summary.json"), recursive=True)
    row = {"config": name, "file": os.path.basename(os.path.dirname(file_path)), "rc": rc, "secs": round(time.time() - t0)}
    if summaries:
        s = json.load(open(summaries[-1], encoding="utf-8"))
        row.update({k: (round(s[k], 2) if isinstance(s[k], (int, float)) else s[k]) for k in ("net_pnl", "max_drawdown_pct", "total_trades", "win_rate_pct", "profit_factor") if k in s and s[k] is not None})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="train", choices=FILES)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(ROOT, "scratch", "sweep_runs"))
    a = ap.parse_args()

    jobs = [(c, CONFIGS[c], f) for c in a.configs.split(",") for f in FILES[a.set]]
    print(f"{len(jobs)} jobs on {a.workers} workers")
    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(lambda j: run_job(j[0], j[1], j[2], a.seed, a.out), jobs))

    res_path = os.path.join(ROOT, "scratch", "sweep_results.csv")
    new = not os.path.exists(res_path)
    keys = ["config", "file", "rc", "secs", "net_pnl", "max_drawdown_pct", "total_trades", "win_rate_pct", "profit_factor"]
    with open(res_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        if new: w.writeheader()
        w.writerows(rows)

    files = [os.path.basename(os.path.dirname(f)) for f in FILES[a.set]]
    print(f"\n{'config':<14}" + "".join(f"{f[-10:]:>12}" for f in files) + f"{'TOTAL':>10}{'trades':>8}{'maxDD%':>8}")
    for c in a.configs.split(","):
        r = {x["file"]: x for x in rows if x["config"] == c}
        pnls = [r[f].get("net_pnl") for f in files]
        tot = sum(p for p in pnls if p is not None)
        trades = sum(r[f].get("total_trades", 0) for f in files)
        dd = max((r[f].get("max_drawdown_pct", 0) for f in files), default=0)
        print(f"{c:<14}" + "".join(f"{(p if p is not None else 'ERR'):>12}" for p in pnls) + f"{tot:>10.0f}{trades:>8}{dd:>8.1f}")


if __name__ == "__main__":
    main()
