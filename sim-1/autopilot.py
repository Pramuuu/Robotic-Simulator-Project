import asyncio
import base64
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import requests
import websockets
import cv2


FLOOR_HALF = 50.0
GOAL_MARGIN = 5.0


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def normalize_vec(x: float, z: float) -> Tuple[float, float]:
    n = math.hypot(x, z)
    if n < 1e-6:
        return (0.0, 0.0)
    return (x / n, z / n)


def rotate_y(x: float, z: float, angle_rad: float) -> Tuple[float, float]:
    s, c = math.sin(angle_rad), math.cos(angle_rad)
    # Rotation around +Y axis (right-handed): x' = x*c + z*s, z' = -x*s + z*c
    return (x * c + z * s, -x * s + z * c)


def corner_to_coords(corner: str, margin: float = GOAL_MARGIN) -> Dict[str, float]:
    c = corner.upper()
    if c in ("NE", "EN", "TR"):
        return {"x": FLOOR_HALF - margin, "y": 0.0, "z": -(FLOOR_HALF - margin)}
    if c in ("NW", "WN", "TL"):
        return {"x": -(FLOOR_HALF - margin), "y": 0.0, "z": -(FLOOR_HALF - margin)}
    if c in ("SE", "ES", "BR"):
        return {"x": FLOOR_HALF - margin, "y": 0.0, "z": FLOOR_HALF - margin}
    # SW default
    return {"x": -(FLOOR_HALF - margin), "y": 0.0, "z": FLOOR_HALF - margin}


@dataclass
class RobotState:
    x: float = 0.0
    z: float = 0.0
    prev_x: float = 0.0
    prev_z: float = 0.0
    collisions: int = 0
    goal_reached: bool = False

    def update_position(self, x: float, z: float) -> None:
        self.prev_x, self.prev_z = self.x, self.z
        self.x, self.z = x, z

    @property
    def heading(self) -> Tuple[float, float]:
        dx, dz = self.x - self.prev_x, self.z - self.prev_z
        hx, hz = normalize_vec(dx, dz)
        if hx == 0.0 and hz == 0.0:
            return (0.0, 1.0)  # arbitrary default forward (positive Z)
        return (hx, hz)


class Autopilot:
    def __init__(self, server_http: str = "http://localhost:5000", server_ws: str = "ws://localhost:8080",
                 corner: Optional[str] = None, step: float = 2.0, max_seconds: int = 300,
                 enable_moving_obstacles: bool = False, obstacle_speed: float = 0.0,
                 verbose: bool = True):
        self.http = server_http.rstrip("/")
        self.ws_url = server_ws
        self.corner = corner or random.choice(["NE", "NW", "SE", "SW"]) 
        self.goal = corner_to_coords(self.corner)
        self.step = step
        self.max_seconds = max_seconds
        self.state = RobotState()
        self.enable_moving_obstacles = enable_moving_obstacles
        self.obstacle_speed = obstacle_speed
        self.verbose = verbose

        # Async message queues
        self.q_messages: "asyncio.Queue[dict]" = asyncio.Queue()
        self.q_captures: "asyncio.Queue[dict]" = asyncio.Queue()

    def log(self, *args):
        if self.verbose:
            print(*args, flush=True)

    # ---------------- HTTP helpers ----------------
    def post(self, path: str, payload: Optional[dict] = None) -> dict:
        url = f"{self.http}{path}"
        r = requests.post(url, json=payload or {}, timeout=10)
        r.raise_for_status()
        return r.json()

    def get(self, path: str) -> dict:
        url = f"{self.http}{path}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()

    # ---------------- CV ----------------
    def compute_steer_angle_deg(self, b64_data_url: str) -> float:
        # data URL: "data:image/png;base64,...."
        comma = b64_data_url.find(',')
        b64 = b64_data_url[comma + 1:] if comma >= 0 else b64_data_url
        img_bytes = base64.b64decode(b64)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return 0.0

        h, w = img.shape[:2]
        # Focus on lower half where obstacles ahead appear
        roi = img[h // 2:h, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Green obstacles threshold (tune as needed)
        lower = np.array([40, 50, 50])
        upper = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

        thirds = w // 3
        left = mask[:, :thirds]
        center = mask[:, thirds:2 * thirds]
        right = mask[:, 2 * thirds:]

        l_score = float(left.mean())
        c_score = float(center.mean())
        r_score = float(right.mean())

        # If center is blocked, steer to the clearer side
        steer = 0.0
        if c_score > 5.0:  # threshold; 0-255 scale
            if l_score < r_score:
                steer = -min(45.0, 60.0 * (c_score / 255.0))
            else:
                steer = min(45.0, 60.0 * (c_score / 255.0))
        else:
            # Slight bias away from denser side even when center is okay
            if abs(l_score - r_score) > 5.0:
                steer = (r_score - l_score) * 0.05  # positive => steer right

        return steer

    # ---------------- Control ----------------
    async def ws_reader(self, ws: websockets.WebSocketClientProtocol):
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            # Push all to general queue
            await self.q_messages.put(msg)
            t = msg.get("type")
            if t == "capture_image_response":
                await self.q_captures.put(msg)
            elif t == "collision" and msg.get("collision"):
                self.state.collisions += 1
                self.log("[event] collision -> count:", self.state.collisions)
            elif t == "goal_reached":
                self.state.goal_reached = True
                self.log("[event] goal_reached")

    async def wait_for_capture(self, timeout: float = 5.0) -> Optional[dict]:
        try:
            return await asyncio.wait_for(self.q_captures.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    def send_goal(self):
        self.log(f"[cmd] set goal at corner {self.corner} -> {self.goal}")
        self.post("/goal", {"corner": self.corner})

    def maybe_enable_moving_obstacles(self):
        if self.enable_moving_obstacles and self.obstacle_speed > 0:
            self.log(f"[cmd] enable moving obstacles, speed={self.obstacle_speed}")
            self.post("/obstacles/motion", {"enabled": True, "speed": float(self.obstacle_speed)})

    def send_move_to(self, x: float, z: float):
        x = clamp(x, -(FLOOR_HALF - 1), FLOOR_HALF - 1)
        z = clamp(z, -(FLOOR_HALF - 1), FLOOR_HALF - 1)
        try:
            self.post("/move", {"x": x, "z": z})
            self.log(f"[cmd] move to -> ({x:.2f}, {z:.2f})")
        except Exception as e:
            self.log("[warn] move failed (is simulator tab open?)", e)

    def request_capture(self):
        try:
            self.post("/capture", {})
        except Exception as e:
            self.log("[warn] capture failed (no simulator connected yet?)", e)

    def distance_to_goal(self) -> float:
        dx = self.goal["x"] - self.state.x
        dz = self.goal["z"] - self.state.z
        return math.hypot(dx, dz)

    async def drive(self):
        # Reset server & simulator state
        try:
            self.post("/reset")
        except Exception:
            pass

        self.send_goal()
        self.maybe_enable_moving_obstacles()

        start = time.time()
        # Seed initial heading roughly toward goal
        self.state.update_position(0.0, 0.0)
        last_dir = normalize_vec(self.goal["x"] - self.state.x, self.goal["z"] - self.state.z)

        self.log("[run] driving... (press Ctrl+C to stop)")
        while not self.state.goal_reached and (time.time() - start) < self.max_seconds:
            # Ask for a capture and wait for it
            self.request_capture()
            cap = await self.wait_for_capture(timeout=3.0)
            if not cap:
                self.log("[warn] capture timeout; retrying")
                continue

            pos = cap.get("position") or {}
            cur_x = float(pos.get("x", self.state.x))
            cur_z = float(pos.get("z", self.state.z))
            self.state.update_position(cur_x, cur_z)

            # Goal vector
            gdx, gdz = self.goal["x"] - cur_x, self.goal["z"] - cur_z
            gdx, gdz = normalize_vec(gdx, gdz)

            # If we have a valid previous motion, use it as current heading
            hx, hz = self.state.heading
            if hx == 0.0 and hz == 0.0:
                hx, hz = gdx, gdz

            # Blend goal and heading for smoother motion
            dir_x, dir_z = normalize_vec(0.6 * gdx + 0.4 * hx, 0.6 * gdz + 0.4 * hz)

            # CV-based steering angle (deg). Negative => steer left
            steer_deg = self.compute_steer_angle_deg(cap.get("image", ""))
            dir_x, dir_z = rotate_y(dir_x, dir_z, math.radians(steer_deg))

            # Take a short step
            nx = cur_x + dir_x * self.step
            nz = cur_z + dir_z * self.step
            last_dir = (dir_x, dir_z)
            self.log(f"[step] pos=({cur_x:.2f},{cur_z:.2f}) steer={steer_deg:.1f}° next=({nx:.2f},{nz:.2f}) d_goal={self.distance_to_goal():.2f}")

            # Move toward waypoint
            self.send_move_to(nx, nz)

            # Exit if close enough
            if self.distance_to_goal() < 3.0:
                break

            await asyncio.sleep(0.15)

        # Final status
        return {
            "goal": self.goal,
            "position": {"x": self.state.x, "z": self.state.z},
            "collisions": self.state.collisions,
            "goal_reached": self.state.goal_reached or (self.distance_to_goal() < 3.0),
        }

    async def run(self) -> dict:
        self.log(f"[ws] connecting to {self.ws_url} ...")
        async with websockets.connect(self.ws_url) as ws:
            self.log("[ws] connected")
            reader = asyncio.create_task(self.ws_reader(ws))
            try:
                result = await self.drive()
            finally:
                reader.cancel()
            return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Autonomous controller for the robot simulator")
    parser.add_argument("--corner", type=str, default=None, help="Corner for goal: NE|NW|SE|SW")
    parser.add_argument("--step", type=float, default=2.0, help="Step size per command")
    parser.add_argument("--moving", action="store_true", help="Enable moving obstacles")
    parser.add_argument("--speed", type=float, default=0.0, help="Obstacle speed when --moving is set")
    parser.add_argument("--no-verbose", action="store_true", help="Disable verbose logs")
    args = parser.parse_args()

    pilot = Autopilot(corner=args.corner, step=args.step,
                      enable_moving_obstacles=args.moving, obstacle_speed=args.speed,
                      verbose=not args.no_verbose)
    result = asyncio.run(pilot.run())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()


