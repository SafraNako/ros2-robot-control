"""A discrete PID controller with a filtered derivative on the measurement and anti-windup by conditional integration."""
from __future__ import annotations

import math


class PID:
    """``u = kp e + ki integral(e) - kd d(filtered measurement)/dt`` limited to ``[out_min, out_max]``.

    * the derivative acts on the measurement (no kick when the setpoint jumps) through a first-order filter with time constant ``tau_d``;
    * the integral stops growing while the output is saturated and the error would push it further in (conditional integration),
      which is what keeps the overshoot after a long saturation small.
    """

    def __init__(self, kp, ki=0.0, kd=0.0, out_min=-math.inf, out_max=math.inf, tau_d=0.02, anti_windup=True):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_min, self.out_max, self.tau_d, self.anti_windup = out_min, out_max, tau_d, anti_windup
        self.reset()

    def reset(self):
        self.integral = 0.0
        self._filtered = None
        self._last = None

    def update(self, setpoint, measurement, dt):
        if dt <= 0:
            raise ValueError("dt must be positive")
        error = setpoint - measurement
        if self._filtered is None:
            self._filtered = measurement
        alpha = dt / (self.tau_d + dt)
        previous = self._filtered
        self._filtered += alpha * (measurement - self._filtered)
        derivative = (self._filtered - previous) / dt
        unsat = self.kp * error + self.ki * (self.integral + error * dt) - self.kd * derivative
        out = min(max(unsat, self.out_min), self.out_max)
        saturated = unsat != out
        pushing = (unsat > self.out_max and error > 0) or (unsat < self.out_min and error < 0)
        if not (self.anti_windup and saturated and pushing):
            self.integral += error * dt
        out = min(max(self.kp * error + self.ki * self.integral - self.kd * derivative, self.out_min), self.out_max)
        self._last = out
        return out
