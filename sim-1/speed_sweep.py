import asyncio
import csv
import json
from statistics import mean
from typing import List

import matplotlib.pyplot as plt

from autopilot import Autopilot


async def run_avg_for_speed(speed: float, trials: int = 4) -> float:
    corners = ["NE", "NW", "SE", "SW"]
    collisions: List[int] = []
    for i in range(trials):
        c = corners[i % len(corners)]
        pilot = Autopilot(corner=c, enable_moving_obstacles=True, obstacle_speed=speed)
        res = await pilot.run()
        print(json.dumps(res, indent=2))
        collisions.append(res.get("collisions", 0))
    return float(mean(collisions)) if collisions else 0.0


async def main():
    import argparse
    p = argparse.ArgumentParser(description="Sweep obstacle speeds and plot average collisions")
    p.add_argument("--speeds", type=float, nargs="*", default=[0.02, 0.04, 0.06, 0.08, 0.10])
    p.add_argument("--output_csv", type=str, default="speed_vs_collisions.csv")
    p.add_argument("--output_png", type=str, default="speed_vs_collisions.png")
    args = p.parse_args()

    rows = []
    for s in args.speeds:
        avg = await run_avg_for_speed(s)
        rows.append((s, avg))
        print(f"Speed {s:.3f} -> avg collisions {avg:.3f}")

    # Write CSV
    with open(args.output_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["speed", "avg_collisions"])
        w.writerows(rows)
    print(f"Saved {args.output_csv}")

    # Plot
    xs = [r[0] for r in rows]
    ys = [r[1] for r in rows]
    plt.figure(figsize=(6,4))
    plt.plot(xs, ys, marker="o")
    plt.title("Obstacle speed vs. average collisions")
    plt.xlabel("Obstacle speed")
    plt.ylabel("Average collisions")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(args.output_png, dpi=150)
    print(f"Saved {args.output_png}")


if __name__ == "__main__":
    asyncio.run(main())


