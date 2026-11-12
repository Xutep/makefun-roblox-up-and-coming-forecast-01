"""roblox-up-and-coming-forecast-01.

Forecast Roblox's Up-and-Coming chart top 10, place by place, for the 72
hours after the history ends. Monte Carlo rank-diffusion model, seeded so
two runs are byte-identical. Standard library only, no network.

Reads $MAKEFUN_INPUT_DIR (else /input): history.ndjson + task.json.
Writes $MAKEFUN_OUTPUT_DIR (else /output): forecast.json.

  mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import random
from datetime import datetime, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

SEED = 20261113
NSIM = 800
SIGMA = 5.0           # rank-change sd per sqrt(day); tuned on backtests (4-6 all ~equal)
W_AGE = 1.0           # weight on the age-based drift (full weight: best in tuning)
TARGET_SUM = 0.80     # each place's named probabilities sum to this; rest is unnamed
MIN_P = 0.001         # drop rows below this before scaling
MAX_AGE_DAYS_ABSENT = 7  # games gone longer than this are dropped


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def age_drift(age_days):
    """Mean rank change per day by game age, measured from the history:
    brand-new games climb fast, older games slide toward the 28-day exit."""
    if age_days < 1:
        return -7.0
    if age_days < 2:
        return -2.0
    if age_days < 8:
        return 4.0
    return 2.0


def load_history(path):
    snaps = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                snaps.append(json.loads(line))
    return snaps


def main():
    with open(os.path.join(INPUT_DIR, "task.json"), "r", encoding="utf-8") as f:
        task = json.load(f)
    places = int(task.get("places", 10))
    hours = int(task.get("window", {}).get("slots", 72))
    output_name = task.get("output", "forecast.json")

    snaps = load_history(os.path.join(INPUT_DIR, "history.ndjson"))
    if not snaps:
        raise SystemExit("empty history")
    last_t = parse_time(snaps[-1]["at"])
    last_rank = {}
    for row in snaps[-1].get("up_and_coming") or []:
        last_rank[int(row["universe_id"])] = int(row["rank"])
    chart_size = len(last_rank)

    # Per-game rank series over the whole history.
    series = {}
    for snap in snaps:
        tt = parse_time(snap["at"])
        for row in snap.get("up_and_coming") or []:
            uid = int(row["universe_id"])
            series.setdefault(uid, []).append((tt, int(row["rank"])))

    games = []  # (universe_id, rank0, drift ranks/day)
    for uid, pts in series.items():
        first_seen = pts[0][0]
        last_seen = pts[-1][0]
        age = (last_t - first_seen).total_seconds() / 86400.0
        if uid in last_rank:
            rank0 = float(last_rank[uid])
        else:
            days_absent = (last_t - last_seen).total_seconds() / 86400.0
            if days_absent > MAX_AGE_DAYS_ABSENT:
                continue
            rank0 = chart_size + 5.0 + 5.0 * days_absent
        # Drift is the age-based drift only. Recent rank velocity was
        # measured and dropped: drift autocorrelation is only ~0.2, and
        # adding velocity hurt both backtests. Player-count growth was
        # also measured: correlation with rank change ~-0.02, i.e. none.
        drift = W_AGE * age_drift(age)
        # A game cannot climb or fall without bound in a day.
        drift = max(-25.0, min(25.0, drift))
        games.append((uid, rank0, drift))
    games.sort()

    rng = random.Random(SEED)
    counts = {k: {} for k in range(1, places + 1)}
    sim_rank_sum = {}
    sim_rank_n = {}
    denom = NSIM * hours
    for _sim in range(NSIM):
        for h in range(1, hours + 1):
            td = h / 24.0
            sd = SIGMA * math.sqrt(td)
            vals = []
            for uid, rank0, drift in games:
                vals.append((rank0 + drift * td + rng.gauss(0.0, sd), uid))
            vals.sort()
            for k in range(min(places, len(vals))):
                uid = vals[k][1]
                d = counts[k + 1]
                d[uid] = d.get(uid, 0) + 1
            for _x, uid in vals[:15]:
                sim_rank_sum[uid] = sim_rank_sum.get(uid, 0.0) + _x
                sim_rank_n[uid] = sim_rank_n.get(uid, 0) + 1

    probabilities = []
    p_top = {}
    for k in range(1, places + 1):
        rows = {}
        total = 0.0
        for uid, c in counts[k].items():
            p = c / denom
            if p >= MIN_P:
                rows[uid] = p
                total += p
        if total > 0:
            scale = TARGET_SUM / total
            for uid in rows:
                rows[uid] *= scale
                p_top[uid] = p_top.get(uid, 0.0) + rows[uid]
        for uid, p in sorted(rows.items()):
            probabilities.append(
                {"universe_id": uid, "place": k, "p": round(p, 6)})
    # top10: most likely to be in the top 10, ties by mean simulated rank.
    top10 = sorted(
        p_top,
        key=lambda u: (-p_top[u],
                       sim_rank_sum.get(u, 1e18) / max(1, sim_rank_n.get(u, 1))),
    )[:places]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, output_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"probabilities": probabilities, "top10": top10}))
        f.write("\n")


if __name__ == "__main__":
    main()
