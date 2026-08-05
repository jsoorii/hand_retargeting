"""
hand_retarget.py
================
Whole-hand retargeting on top of `thumb_retarget`.

Structural note
---------------
A finger here is AA + parallel-3R. That is the SAME structure as the thumb's
yaw + parallel-3R: one joint whose axis is perpendicular to the three parallel
flexion axes, placed proximal to them. So `ThumbKinematics` / `ThumbIK` apply
verbatim -- only link lengths, joint limits and `R_off` differ. Define each
digit's base frame so that

    +z  = the AA (spread) axis
    -y  = the three parallel flexion axes
    +x  = the neutral pointing direction

and the closed-form solver needs no changes. `Finger1P3R` below is just an
alias; the code does not care whether you call the first joint yaw or pitch.

Retargeting policy
------------------
Thumb:   tracked absolutely, as accurately as the hardware allows.
Fingers: a blend of
           (a) absolute scaled position, and
           (b) a position defined relative to the THUMB'S ACHIEVED fingertip,
with the blend weight rising as the human's thumb-finger distance shrinks.

Two things this gets right that a naive version does not:

1. The relative anchor is the thumb's ACHIEVED pose (FK after clamping and
   rate limiting), never its commanded target. When the thumb saturates -- which
   is exactly when a pinch is being attempted -- anchoring on the command would
   place the finger relative to a pose the thumb never reached, and the pinch
   fails precisely when it matters most.

2. Human pinch distance is not zero. Nail- or pad-frame separation at a firm
   human pinch is typically 15-25 mm, and the robot's own pads have thickness
   too. `PinchMap` maps human separation onto robot separation with both
   contact offsets accounted for, so "human contact" maps to "robot contact"
   rather than to interpenetration or to a permanent gap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from thumb_retarget import (ThumbKinematics, ThumbRetargeter, RetargetOutput,
                            OneEuroFilter3D, RotationLPF, make_T, inv_T)

Finger1P3R = ThumbKinematics          # same structure, different numbers

FINGERS = ("index", "middle", "ring", "pinky")
DIGITS = ("thumb",) + FINGERS


# ----------------------------------------------------------------------------
# distance remapping and blending
# ----------------------------------------------------------------------------


@dataclass
class PinchMap:
    """Human thumb-finger separation -> desired robot separation.

    d_robot = d_contact_robot + scale * max(0, d_human - d_contact_human)

    so that d_human == d_contact_human maps exactly onto robot contact, and
    larger openings scale linearly from there.
    """

    d_contact_human: float = 0.020    # [m] measured in the pinch calibration pose
    d_contact_robot: float = 0.016    # [m] sum of robot pad radii + clearance
    scale: float = 1.0                # hand-size ratio, from calibration

    def __call__(self, d_human: float) -> float:
        return self.d_contact_robot + self.scale * max(
            0.0, d_human - self.d_contact_human)


@dataclass
class FingerBlend:
    """Weight on the thumb-relative target as a function of human separation.

    Far apart -> absolute mapping, so an open hand looks natural.
    Close      -> relative mapping, so pinch geometry is preserved.

    Ring and pinky rarely participate in precision pinches, so their `w_max` is
    lower: forcing them to chase thumb-relative targets distorts the whole hand
    posture for no benefit.
    """

    w_max: float = 1.0
    d_half: float = 0.045             # [m] human separation at w = w_max/2
    k: float = 120.0                  # [1/m] sigmoid steepness

    def weight(self, d_human: float) -> float:
        return self.w_max / (1.0 + np.exp(self.k * (d_human - self.d_half)))


DEFAULT_BLENDS = {
    "index": FingerBlend(w_max=1.0),
    "middle": FingerBlend(w_max=0.9),
    "ring": FingerBlend(w_max=0.5, d_half=0.035),
    "pinky": FingerBlend(w_max=0.3, d_half=0.030),
}


# ----------------------------------------------------------------------------
# calibration and mounting
# ----------------------------------------------------------------------------


@dataclass
class DigitConfig:
    kin: ThumbKinematics
    T_palm_base: np.ndarray            # palm frame -> this digit's base frame
    T_nail_pad: np.ndarray = field(default_factory=lambda: make_T(
        np.eye(3), np.array([0.0, 0.0, -0.018])))
    R_pad_tip: np.ndarray = field(default_factory=lambda: np.eye(3))
    blend: Optional[FingerBlend] = None       # None for the thumb


@dataclass
class HandCalibration:
    T_dorsum_palm: np.ndarray = field(default_factory=lambda: np.eye(4))
    scale: float = 1.0                 # robot hand / human hand
    pinch: dict = field(default_factory=lambda: {
        f: PinchMap() for f in FINGERS})

    def sane(self) -> bool:
        return 0.7 <= self.scale <= 1.4


# ----------------------------------------------------------------------------
# hand retargeter
# ----------------------------------------------------------------------------


@dataclass
class HandOutput:
    per_digit: dict           # name -> RetargetOutput
    pinch_dist: dict          # finger -> achieved robot thumb-finger distance
    blend_w: dict             # finger -> blend weight used


class HandRetargeter:
    def __init__(self, digits: dict[str, DigitConfig],
                 calib: HandCalibration = HandCalibration(),
                 qd_max: Optional[np.ndarray] = None):
        assert "thumb" in digits, "a thumb config is required"
        self.cfg = digits
        self.calib = calib
        self.rt = {
            name: ThumbRetargeter(d.kin, calib=_dummy_calib(), qd_max=qd_max)
            for name, d in digits.items()
        }
        # Filtering happens in the PALM frame, before blending, so each digit
        # gets its own filter pair here rather than using the ones inside
        # ThumbRetargeter.
        self.pos_f = {n: OneEuroFilter3D(min_cutoff=1.0, beta=0.05)
                      for n in digits}
        self.rot_f = {n: RotationLPF(cutoff=2.0) for n in digits}

    # -- session control ------------------------------------------------------
    def engage(self) -> None:
        for n in self.rt:
            self.rt[n].engage()
            self.pos_f[n].reset()
            self.rot_f[n].reset()

    def disengage(self) -> None:
        for r in self.rt.values():
            r.disengage()

    # -- helpers --------------------------------------------------------------
    def _to_palm(self, name: str, T_dorsum_nail: np.ndarray):
        d = self.cfg[name]
        T = self.calib.T_dorsum_palm @ T_dorsum_nail @ d.T_nail_pad
        return T[:3, 3].copy(), T[:3, :3] @ d.R_pad_tip

    def _to_base(self, name: str, p_palm: np.ndarray, R_palm: np.ndarray):
        T = self.cfg[name].T_palm_base
        return T[:3, :3] @ p_palm + T[:3, 3], T[:3, :3] @ R_palm

    def _from_base(self, name: str, p_base: np.ndarray) -> np.ndarray:
        Ti = inv_T(self.cfg[name].T_palm_base)
        return Ti[:3, :3] @ p_base + Ti[:3, 3]

    # -- one control-rate step ------------------------------------------------
    def step(self, poses: dict[str, np.ndarray], dt: float) -> HandOutput:
        """`poses`: digit name -> T_dorsum_nail (4x4), as produced by the glove
        source layer. Missing digits are skipped."""
        s = self.calib.scale

        # 1) everything into the palm frame, unscaled and scaled
        p_h_raw, p_h, R_h = {}, {}, {}
        for name in poses:
            if name not in self.cfg:
                continue
            p, R = self._to_palm(name, poses[name])
            p_h_raw[name] = p                       # human metric, for distances
            p_h[name] = self.pos_f[name](p * s, dt)  # scaled + filtered
            R_h[name] = self.rot_f[name](R, dt)

        out: dict[str, RetargetOutput] = {}
        pinch: dict[str, float] = {}
        weights: dict[str, float] = {}

        # 2) thumb first -- absolute, and it defines the anchor
        pb, Rb = self._to_base("thumb", p_h["thumb"], R_h["thumb"])
        out["thumb"] = self.rt["thumb"].solve_target(pb, Rb, dt)
        anchor = self._from_base("thumb", out["thumb"].p_achieved)

        # 3) fingers -- blend absolute with thumb-relative
        for name in FINGERS:
            if name not in p_h:
                continue
            d_cfg = self.cfg[name]
            blend = d_cfg.blend or DEFAULT_BLENDS.get(name, FingerBlend())

            v_raw = p_h_raw[name] - p_h_raw["thumb"]
            d_human = float(np.linalg.norm(v_raw))
            if d_human < 1e-6:
                u = np.array([0.0, 0.0, 1.0])
            else:
                u = v_raw / d_human

            d_robot = self.calib.pinch[name](d_human)
            p_rel = anchor + d_robot * u
            p_abs = p_h[name]

            w = float(blend.weight(d_human))
            p_target = (1.0 - w) * p_abs + w * p_rel
            weights[name] = w

            pb, Rb = self._to_base(name, p_target, R_h[name])
            out[name] = self.rt[name].solve_target(pb, Rb, dt)

            pinch[name] = float(np.linalg.norm(
                self._from_base(name, out[name].p_achieved) - anchor))

        return HandOutput(out, pinch, weights)


def _dummy_calib():
    """HandRetargeter feeds targets straight into `solve_target`, so the
    per-digit ThumbCalibration is never consulted."""
    from thumb_retarget import ThumbCalibration
    return ThumbCalibration()


# ----------------------------------------------------------------------------
# calibration helper: estimate the human contact distance
# ----------------------------------------------------------------------------


class PinchCalibrator:
    """Feed frames recorded while the user holds a firm thumb-to-finger pinch.
    The 5th percentile (not the minimum) is used, so a single tracking glitch
    cannot collapse the estimate."""

    def __init__(self):
        self._d: dict[str, list[float]] = {f: [] for f in FINGERS}

    def feed(self, p_palm: dict[str, np.ndarray]) -> None:
        if "thumb" not in p_palm:
            return
        for f in FINGERS:
            if f in p_palm:
                self._d[f].append(
                    float(np.linalg.norm(p_palm[f] - p_palm["thumb"])))

    def result(self, finger: str, min_samples: int = 100) -> float:
        d = np.asarray(self._d[finger])
        if d.size < min_samples:
            raise RuntimeError(f"need >= {min_samples} samples for {finger}")
        return float(np.percentile(d, 5))


# ----------------------------------------------------------------------------
# NOTE -- inter-finger collision.
# Adjacent fingers can cross when AA saturates or when the blend pulls a finger
# sideways. Each digit's IK already accepts a `self_collision_fn` that is
# evaluated inside the beta scan, so the cheapest fix is a closure that tests
# this finger's link capsules against the neighbours' capsules at their
# PREVIOUS commanded q. One frame of lag is harmless at teleoperation speeds
# and keeps the digits decoupled (no joint solve).
# ----------------------------------------------------------------------------


if __name__ == "__main__":
    from thumb_retarget import exp_so3

    def mount(p, yaw=0.0):
        c, s_ = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1.0]])
        return inv_T(make_T(R, np.asarray(p, float)))

    thumb = DigitConfig(ThumbKinematics(L1=0.030, L2=0.025, L3=0.020),
                        mount([0.00, -0.030, 0.010], yaw=-0.9))
    digits = {"thumb": thumb}
    for i, name in enumerate(FINGERS):
        L = [0.045, 0.040, 0.035, 0.030][i]
        digits[name] = DigitConfig(
            Finger1P3R(L1=L * 0.45, L2=L * 0.32, L3=L * 0.23,
                       q_min=np.deg2rad([-15, -10, 0, 0]),
                       q_max=np.deg2rad([15, 90, 110, 80])),
            mount([0.02 * (i - 1.5), 0.075, 0.0]))

    hand = HandRetargeter(digits, HandCalibration(scale=1.0))
    hand.engage()

    dt = 1.0 / 500.0
    for i in range(400):
        t = i * dt
        close = 0.5 * (1 - np.cos(2 * np.pi * 0.5 * t))     # pinch cycle
        poses = {
            "thumb": make_T(np.eye(3), np.array([0.010, -0.010, 0.045])
                            + close * np.array([0.005, 0.030, 0.010])),
        }
        for j, name in enumerate(FINGERS):
            poses[name] = make_T(np.eye(3),
                                 np.array([0.02 * (j - 1.5), 0.075, 0.090])
                                 - close * np.array([0.0, 0.010, 0.045]))
        o = hand.step(poses, dt)

    print("blend w :", {k: round(v, 2) for k, v in o.blend_w.items()})
    print("pinch mm:", {k: round(v * 1e3, 1) for k, v in o.pinch_dist.items()})
    print("thumb err mm:", round(o.per_digit["thumb"].pos_error * 1e3, 1))
