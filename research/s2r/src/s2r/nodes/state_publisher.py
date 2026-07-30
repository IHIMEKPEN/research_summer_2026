"""Publishes robot state + lightweight map frames for GUI monitoring."""

from __future__ import annotations
from s2r.robot import G1_DOF

import math
import time

import numpy as np

from s2r.core.node import Node
from s2r.core.schemas import MapFrame, RobotState, Topic
from s2r.core.zmq_bus import Publisher, Subscriber


class StatePublisherNode(Node):
    """In sim mode, integrates joint commands into a simple plant.

    In real/hybrid mode, replace `_integrate_sim` with hardware state reads.
    """

    name = "state_publisher"

    def setup(self) -> None:
        z = self.zmq_cfg
        robot = self.cfg.get("robot", {})
        self.n_joints = int(robot.get("n_joints", G1_DOF))
        self.hz = float(robot.get("control_hz", 100))
        self.mode = str(self.cfg.get("pipeline", {}).get("mode", "sim"))
        self._q = np.zeros(self.n_joints, dtype=np.float64)
        self._dq = np.zeros(self.n_joints, dtype=np.float64)
        self._cmd = self._q.copy()
        self._xy = np.array([0.0, 0.0], dtype=np.float64)
        self._yaw = 0.0

        self.cmd_sub = Subscriber(z["joint_cmd_pub"], topics=[Topic.JOINT_CMD], zmq_cfg=z, conflate=True)
        self.state_pub = Publisher(z["state_pub"], zmq_cfg=z, source=self.name, conflate=True)
        self.map_pub = Publisher(z["map_pub"], zmq_cfg=z, source=self.name, conflate=True)
        self.gui_pub = Publisher(z["gui_pub"], zmq_cfg=z, bind=False, source=self.name)

    def _integrate_sim(self, dt: float) -> None:
        # First-order joint tracking plant
        alpha = min(1.0, 12.0 * dt)
        self._dq = (self._cmd - self._q) / max(dt, 1e-4)
        self._q = (1 - alpha) * self._q + alpha * self._cmd
        # Fake base motion from joint 0/1 for mapping visualization
        self._yaw = float(self._q[0])
        speed = 0.15 * float(self._q[1])
        self._xy[0] += speed * math.cos(self._yaw) * dt
        self._xy[1] += speed * math.sin(self._yaw) * dt

    def _landmarks(self) -> list[list[float]]:
        # Circling demo landmarks around origin
        pts = []
        for i in range(8):
            a = i * (math.pi / 4)
            pts.append([1.5 * math.cos(a), 1.5 * math.sin(a)])
        return pts

    def _grid(self) -> list[list[float]]:
        # Tiny 16x16 occupancy toy grid with robot blob
        n = 16
        g = np.zeros((n, n), dtype=np.float64)
        cx = int(np.clip((self._xy[0] + 2) / 4 * (n - 1), 0, n - 1))
        cy = int(np.clip((self._xy[1] + 2) / 4 * (n - 1), 0, n - 1))
        for i in range(max(0, cx - 1), min(n, cx + 2)):
            for j in range(max(0, cy - 1), min(n, cy + 2)):
                g[j, i] = 1.0
        return g.tolist()

    def step(self) -> None:
        t0 = time.perf_counter()
        env = self.cmd_sub.recv(timeout_ms=0)
        if env is not None:
            q = env.payload.get("q")
            if q is not None:
                self._cmd = np.asarray(q, dtype=np.float64)

        dt = 1.0 / self.hz
        if self.mode in {"sim", "hybrid"}:
            self._integrate_sim(dt)

        # Fake EE from average joints
        ee = [
            float(0.4 * math.cos(self._q[0])),
            float(0.4 * math.sin(self._q[0])),
            float(0.5 + 0.1 * self._q[2]),
        ]
        state = RobotState(
            joint_pos=self._q.tolist(),
            joint_vel=self._dq.tolist(),
            ee_pos=ee,
            mode=self.mode,
        ).model_dump()
        self.state_pub.publish(Topic.STATE, state)
        self.gui_pub.publish(Topic.STATE, state)

        if self._ticks % max(1, int(self.hz / 10)) == 0:
            mf = MapFrame(
                robot_xy=self._xy.tolist(),
                robot_yaw=self._yaw,
                landmarks=self._landmarks(),
                grid=self._grid(),
                resolution=0.25,
            ).model_dump()
            self.map_pub.publish(Topic.MAP, mf)
            self.gui_pub.publish(Topic.MAP, mf)

        # Keep loop rate
        self.rate_sleep(self.hz, t0)

    def teardown(self) -> None:
        self.cmd_sub.close()
        self.state_pub.close()
        self.map_pub.close()
        self.gui_pub.close()

    def run(self, hz: float | None = None) -> None:
        # step() already rate-limits
        super().run(hz=None)
