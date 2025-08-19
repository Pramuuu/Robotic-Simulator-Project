import asyncio
import json
from statistics import mean

from autopilot import Autopilot


async def run_trials(corners, moving=False, speed=0.0):
    results = []
    for c in corners:
        pilot = Autopilot(corner=c, enable_moving_obstacles=moving, obstacle_speed=speed)
        res = await pilot.run()
        results.append(res)
        print(json.dumps(res, indent=2))
    avg_collisions = mean(r["collisions"] for r in results)
    print("Average collisions:", avg_collisions)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--moving", action="store_true")
    p.add_argument("--speed", type=float, default=0.0)
    args = p.parse_args()
    corners = ["NE", "NW", "SE", "SW"]
    asyncio.run(run_trials(corners, moving=args.moving, speed=args.speed))


