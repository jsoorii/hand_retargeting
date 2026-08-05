"""
glove_source.py
===============
Glove-agnostic source layer for thumb retargeting.

The contract every source implements is a single quantity:

    T_dorsum_nail  --  the nail/sensor pose expressed in the dorsum frame

That is the narrowest interface that any hand-tracking glove can satisfy, which
is what makes swapping hardware cheap. Everything glove-specific (units, axis
convention, array ordering, solver bypass) is confined to a Source class.

Sources, in order of how little processing sits between sensor and output:

  1. ManusRawSensorSource   -- MANUS Metagloves Pro `raw_sensor` array.
                               Sensor poses are already expressed relative to
                               the glove source, and the glove source module
                               sits on the back of the hand -- so this IS
                               T_dorsum_nail, with the Advanced Hand Solver
                               completely bypassed. Preferred.
  2. ManusRawSkeletonSource -- 21-node raw skeleton, hand node vs thumb node.
                               Solver output. Fallback for legacy Quantum
                               gloves, and useful as an A/B reference.
  3. GenericRelativeSource / GenericAbsoluteSource
                            -- drop-in points for a future glove.

Consequences of bypassing the solver (read these)
-------------------------------------------------
* No per-user hand scaling from Core at all. Our own calibration (spherical
  fit for CMC + length, Procrustes on the functional poses) becomes the *only*
  scaling mechanism. That is fine -- it is what we designed -- but it means the
  calibration is now mandatory, not a refinement.

* The sensor clips onto the fingertip, so its mount offset changes slightly
  every time the glove is donned. `ThumbCalibration.T_nail_pad` now has to
  absorb per-donning variation, not just a fixed hardware offset. Run the
  10-second calibration sweep at the START OF EVERY SESSION rather than once
  per user.

* EMF tracking is sensitive to metal, motors and electronics -- i.e. exactly a
  robot workcell. The solver used to smooth some of that away; now it lands
  directly on the filter input. `JumpGate` below rejects the impulse-type
  outliers that One-Euro would otherwise chase.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence

import numpy as np

from thumb_retarget import make_T, inv_T, log_so3


# ----------------------------------------------------------------------------
# common types
# ----------------------------------------------------------------------------


@dataclass
class ThumbPose:
    T_dorsum_nail: np.ndarray     # (4,4)
    stamp: float                  # [s], monotonic


class ThumbPoseSource(Protocol):
    def latest(self) -> Optional[ThumbPose]:
        """Most recent accepted pose, or None before the first valid packet."""
        ...


def quat_to_R(q_xyzw) -> np.ndarray:
    x, y, z, w = np.asarray(q_xyzw, dtype=float).reshape(4)
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w),     s * (x * z + y * w)],
        [s * (x * y + z * w),     1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w),     s * (y * z + x * w),     1 - s * (x * x + y * y)],
    ])


def axis_remap(x_to: str, y_to: str, z_to: str) -> np.ndarray:
    """Map glove axes onto robot axes. Right-handed -> right-handed only;
    a handedness flip is a mirror and must be fixed at the glove/Core side."""
    basis = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]),
             "z": np.array([0, 0, 1.0])}
    cols = []
    for spec in (x_to, y_to, z_to):
        sign = -1.0 if spec.startswith("-") else 1.0
        cols.append(sign * basis[spec.lstrip("+-")])
    M = np.column_stack(cols)
    if abs(np.linalg.det(M) - 1.0) > 1e-6:
        raise ValueError("axis_remap must be a proper rotation (det=+1)")
    return M


# ----------------------------------------------------------------------------
# outlier rejection -- runs BEFORE the One-Euro filter
# ----------------------------------------------------------------------------


class JumpGate:
    """Reject physically impossible pose jumps (EMF spikes, packet corruption).

    A low-pass filter cannot save you from these: One-Euro specifically raises
    its cutoff when it sees high velocity, so a spike is exactly the input it
    passes through fastest. Gate first, filter second.

    Consecutive rejections are counted; after `max_reject` in a row the gate
    gives up and accepts, on the assumption that the hand really did move fast
    (or tracking re-acquired at a new pose) rather than the stream being noisy.
    """

    def __init__(self, max_lin_vel: float = 3.0,      # [m/s]
                 max_ang_vel: float = 30.0,           # [rad/s]
                 max_reject: int = 5):
        self.max_lin_vel = max_lin_vel
        self.max_ang_vel = max_ang_vel
        self.max_reject = max_reject
        self._prev: Optional[ThumbPose] = None
        self._n_reject = 0
        self.reject_count = 0          # cumulative, for diagnostics

    def reset(self) -> None:
        self._prev = None
        self._n_reject = 0

    def __call__(self, pose: ThumbPose) -> Optional[ThumbPose]:
        T = pose.T_dorsum_nail
        if not np.all(np.isfinite(T)):
            return None
        if self._prev is None:
            self._prev = pose
            return pose

        dt = max(pose.stamp - self._prev.stamp, 1e-6)
        dp = np.linalg.norm(T[:3, 3] - self._prev.T_dorsum_nail[:3, 3]) / dt
        dR = np.linalg.norm(
            log_so3(self._prev.T_dorsum_nail[:3, :3].T @ T[:3, :3])) / dt

        if (dp > self.max_lin_vel or dR > self.max_ang_vel) \
                and self._n_reject < self.max_reject:
            self._n_reject += 1
            self.reject_count += 1
            return None                # hold previous

        self._n_reject = 0
        self._prev = pose
        return pose


# ----------------------------------------------------------------------------
# 1) MANUS Metagloves Pro -- raw sensor path (solver bypassed)
# ----------------------------------------------------------------------------


@dataclass
class ManusRawSensorConfig:
    # Index of the thumb sensor inside the `raw_sensor` array.
    # None -> must be established once via ThumbSensorIdentifier.
    thumb_sensor_index: Optional[int] = None
    # Identity if the glove-side coordinate system already matches the robot.
    R_glove_to_robot: np.ndarray = field(default_factory=lambda: np.eye(3))
    unit_scale: float = 1.0            # 1.0 = metres


class ManusRawSensorSource:
    """Consumes the `raw_sensor` array from a ManusGlove message.

    `raw_sensor[i]` is a Pose expressed relative to the glove source, and the
    glove source module is mounted on the back of the hand. So no dorsum node,
    no subtraction and no solver is involved: the pose IS T_dorsum_nail.

    `raw_sensor_orientation` (the raw wrist orientation) is deliberately NOT
    used -- it would only be needed to lift these poses into a world frame,
    which the retargeter never requires.
    """

    def __init__(self, cfg: ManusRawSensorConfig = ManusRawSensorConfig(),
                 gate: Optional[JumpGate] = None):
        self.cfg = cfg
        self.gate = gate if gate is not None else JumpGate()
        self._latest: Optional[ThumbPose] = None

    def on_packet(self, raw_sensor: Sequence, stamp: Optional[float] = None
                  ) -> None:
        """`raw_sensor`: sequence of objects with .position (x,y,z) and
        .orientation (x,y,z,w), i.e. geometry_msgs/Pose."""
        idx = self.cfg.thumb_sensor_index
        if idx is None:
            raise RuntimeError(
                "thumb_sensor_index is not set. The ordering of `raw_sensor` "
                "is not documented and may change -- run ThumbSensorIdentifier "
                "once and store the result.")
        if idx >= len(raw_sensor):
            return
        s = raw_sensor[idx]
        M, sc = self.cfg.R_glove_to_robot, self.cfg.unit_scale

        p = M @ (np.array([s.position.x, s.position.y, s.position.z]) * sc)
        R = M @ quat_to_R((s.orientation.x, s.orientation.y,
                           s.orientation.z, s.orientation.w)) @ M.T

        t = time.monotonic() if stamp is None else float(stamp)
        accepted = self.gate(ThumbPose(make_T(R, p), t))
        if accepted is not None:
            self._latest = accepted

    def latest(self) -> Optional[ThumbPose]:
        return self._latest


class ThumbSensorIdentifier:
    """Establishes which entry of `raw_sensor` is the thumb, without relying on
    an undocumented array ordering.

    Procedure: ask the user to move ONLY the thumb for a few seconds while
    holding the other fingers still, feed every packet in, then read `result()`.
    The thumb is the sensor with the largest positional variance. Store the
    index in config; re-run only if the glove firmware or Core version changes.
    """

    def __init__(self):
        self._samples: list[np.ndarray] = []

    def feed(self, raw_sensor: Sequence) -> None:
        self._samples.append(np.array(
            [[s.position.x, s.position.y, s.position.z] for s in raw_sensor]))

    def result(self, min_samples: int = 200) -> tuple[int, float]:
        """Returns (index, confidence). Confidence is the ratio between the
        winner's motion and the runner-up's; below ~2.0 the user probably moved
        other fingers too, so re-run rather than trusting it."""
        if len(self._samples) < min_samples:
            raise RuntimeError(f"need >= {min_samples} samples, "
                               f"have {len(self._samples)}")
        X = np.stack(self._samples)                      # (T, N, 3)
        motion = X.std(axis=0).sum(axis=1)               # (N,)
        order = np.argsort(motion)[::-1]
        best, second = motion[order[0]], motion[order[1]]
        return int(order[0]), float(best / max(second, 1e-9))


# ----------------------------------------------------------------------------
# 2) MANUS raw skeleton path (solver output) -- fallback / A-B reference
# ----------------------------------------------------------------------------

_CHAIN_KEYS = ("thumb", "index", "middle", "ring", "pinky", "hand")
_JOINT_KEYS = ("metacarpal", "proximal", "intermediate", "distal", "tip")


def _norm(value: str, keys: Sequence[str]) -> str:
    v = str(value).lower()
    for k in keys:
        if k in v:
            return k
    return "unknown"


@dataclass
class RawNode:
    node_id: int
    side: str
    chain: str
    joint: str
    position: np.ndarray
    quat_xyzw: np.ndarray


def normalize_node(node_id, side, chain_type, joint_type, position,
                   quat_xyzw) -> RawNode:
    return RawNode(int(node_id), _norm(side, ("left", "right")),
                   _norm(chain_type, _CHAIN_KEYS), _norm(joint_type, _JOINT_KEYS),
                   np.asarray(position, float).reshape(3),
                   np.asarray(quat_xyzw, float).reshape(4))


class ManusRawSkeletonSource:
    """Hand node vs thumb node, differenced. MANUS warns that raw skeleton node
    IDs are not stable, so roles are resolved by (side, chain, joint) at
    startup rather than hardcoded."""

    def __init__(self, side: str = "right",
                 orientation_from: str = "distal", position_from: str = "tip",
                 R_glove_to_robot: Optional[np.ndarray] = None,
                 unit_scale: float = 1.0, gate: Optional[JumpGate] = None):
        self.side = side
        self.orientation_from = orientation_from
        self.position_from = position_from
        self.M = np.eye(3) if R_glove_to_robot is None else R_glove_to_robot
        self.unit_scale = unit_scale
        self.gate = gate if gate is not None else JumpGate()
        self._ids: dict[str, int] = {}
        self._latest: Optional[ThumbPose] = None

    @staticmethod
    def dump_nodes(nodes: Sequence[RawNode]) -> str:
        lines = ["id   side   chain      joint"]
        for n in sorted(nodes, key=lambda x: x.node_id):
            lines.append(f"{n.node_id:<4} {n.side:<6} {n.chain:<10} {n.joint}")
        return "\n".join(lines)

    def resolve(self, nodes: Sequence[RawNode]) -> None:
        found: dict[str, int] = {}
        for n in nodes:
            if n.side != self.side:
                continue
            if n.chain == "hand":
                found["hand"] = n.node_id
            elif n.chain == "thumb":
                found[n.joint] = n.node_id
        for role in ("hand", self.orientation_from, self.position_from):
            if role not in found:
                raise RuntimeError(
                    f"node role '{role}' not found; run dump_nodes() to see "
                    f"the actual chain/joint strings for this Core version")
        self._ids = found

    def on_packet(self, nodes: Sequence[RawNode],
                  stamp: Optional[float] = None) -> None:
        if not self._ids:
            self.resolve(nodes)
        by_id = {n.node_id: n for n in nodes}
        try:
            n_hand = by_id[self._ids["hand"]]
            n_rot = by_id[self._ids[self.orientation_from]]
            n_pos = by_id[self._ids[self.position_from]]
        except KeyError:
            return
        M, s = self.M, self.unit_scale
        T_hand = make_T(M @ quat_to_R(n_hand.quat_xyzw) @ M.T,
                        M @ (n_hand.position * s))
        T_nail = make_T(M @ quat_to_R(n_rot.quat_xyzw) @ M.T,
                        M @ (n_pos.position * s))
        t = time.monotonic() if stamp is None else float(stamp)
        accepted = self.gate(ThumbPose(inv_T(T_hand) @ T_nail, t))
        if accepted is not None:
            self._latest = accepted

    def latest(self) -> Optional[ThumbPose]:
        return self._latest


# ----------------------------------------------------------------------------
# 3) generic drop-in points for a future glove
# ----------------------------------------------------------------------------


class GenericRelativeSource:
    """For any glove that reports the thumb sensor relative to a hand-mounted
    module. Push T_dorsum_nail directly."""

    def __init__(self, gate: Optional[JumpGate] = None):
        self.gate = gate if gate is not None else JumpGate()
        self._latest: Optional[ThumbPose] = None

    def push(self, T_dorsum_nail: np.ndarray,
             stamp: Optional[float] = None) -> None:
        t = time.monotonic() if stamp is None else float(stamp)
        acc = self.gate(ThumbPose(np.asarray(T_dorsum_nail, float), t))
        if acc is not None:
            self._latest = acc

    def latest(self) -> Optional[ThumbPose]:
        return self._latest


class GenericAbsoluteSource(GenericRelativeSource):
    """For gloves that only report world-frame poses (e.g. optical mocap)."""

    def push_absolute(self, T_world_dorsum, T_world_nail,
                      stamp: Optional[float] = None) -> None:
        self.push(inv_T(np.asarray(T_world_dorsum, float))
                  @ np.asarray(T_world_nail, float), stamp)


# ----------------------------------------------------------------------------
# control-rate reader + pipeline
# ----------------------------------------------------------------------------


@dataclass
class ReadResult:
    pose: ThumbPose
    dt: float
    fresh: bool
    fault: bool


class SourceReader:
    """Bridges the glove rate (~120 Hz) to the control rate. Computes dt from
    real timestamps so drops and jitter do not corrupt the One-Euro velocity
    estimate or the joint rate limiter."""

    def __init__(self, source: ThumbPoseSource,
                 stale_after: float = 0.10, fault_after: float = 0.50):
        self.source = source
        self.stale_after = stale_after
        self.fault_after = fault_after
        self._last_step_t: Optional[float] = None

    def read(self, now: Optional[float] = None) -> Optional[ReadResult]:
        pose = self.source.latest()
        if pose is None:
            return None
        t = time.monotonic() if now is None else float(now)
        dt = 1e-3 if self._last_step_t is None else max(t - self._last_step_t, 1e-6)
        self._last_step_t = t
        age = t - pose.stamp
        return ReadResult(pose, dt, age < self.stale_after,
                          age > self.fault_after)


class ThumbPipeline:
    """Call `tick()` once per control cycle."""

    def __init__(self, retargeter, reader: SourceReader):
        self.rt = retargeter
        self.reader = reader

    def tick(self, now: Optional[float] = None):
        r = self.reader.read(now)
        if r is None:
            return None
        if r.fault:
            self.rt.disengage()        # stream lost -> freeze
            return None
        return self.rt.step_relative(r.pose.T_dorsum_nail, r.dt)


# ----------------------------------------------------------------------------
# ROS2 wiring sketch (MANUS ROS2 package, ManusGlove msg @120 Hz)
# ----------------------------------------------------------------------------

ROS2_SKETCH = '''
import rclpy
from rclpy.node import Node
from manus_ros2_msgs.msg import ManusGlove     # verify exact pkg/msg name

class ManusProNode(Node):
    def __init__(self, source, topic="/manus_glove_0"):
        super().__init__("thumb_source")
        self.source = source
        self.create_subscription(ManusGlove, topic, self._cb, 10)

    def _cb(self, msg):
        # raw_sensor is Pro-series only; raw_sensor_count tells you how many.
        if msg.raw_sensor_count == 0:
            self.get_logger().warn("no raw_sensor data -- Pro series and "
                                   "Core Integrated mode required")
            return
        self.source.on_packet(msg.raw_sensor)
'''


if __name__ == "__main__":
    from thumb_retarget import (ThumbKinematics, ThumbCalibration,
                                ThumbRetargeter, PositionErrorMetric, exp_so3)

    kin = ThumbKinematics(L1=0.030, L2=0.025, L3=0.020)
    rt = ThumbRetargeter(kin, ThumbCalibration(scale=1.0))
    rt.engage()

    src = GenericRelativeSource()
    pipe = ThumbPipeline(rt, SourceReader(src))

    dt, metric = 1.0 / 500.0, PositionErrorMetric()
    rng = np.random.default_rng(0)
    for i in range(1000):
        t = i * dt
        if i % 4 == 0:                                   # glove at 125 Hz
            p = np.array([0.055 + 0.012 * np.sin(2 * np.pi * 0.4 * t),
                          0.006 * np.sin(2 * np.pi * 0.25 * t),
                          0.012 * np.sin(2 * np.pi * 0.35 * t)])
            if rng.random() < 0.01:                      # simulated EMF spike
                p += rng.normal(0, 0.05, 3)
            R = exp_so3(np.array([0.0, 0.5 * np.sin(2 * np.pi * 0.3 * t), 0.0]))
            src.push(make_T(R, p), stamp=t)
        out = pipe.tick(now=t)
        if out is not None:
            metric.update(out)
    print(metric.summary(), "| spikes rejected:", src.gate.reject_count)
