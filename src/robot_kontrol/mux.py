"""A priority multiplexer for velocity commands with a watchdog: the highest-priority fresh command wins, silence means stop."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CmdMux:
    """Sources are registered with a priority (higher wins) and a timeout (s); a command older than its timeout is ignored.

    ``select`` returns the winning command, or a stop ``(0, 0)`` when every source has timed out (the watchdog).
    """

    sources: dict = field(default_factory=dict)         # name -> (priority, timeout)
    _last: dict = field(default_factory=dict)           # name -> (time, v, w)

    def add_source(self, name, priority, timeout):
        self.sources[name] = (priority, timeout)

    def update(self, name, v, w, now):
        if name not in self.sources:
            raise KeyError(f"unknown source {name!r}")
        self._last[name] = (now, v, w)

    def select(self, now):
        """``(v, w, source)``; ``source`` is ``None`` when the watchdog has stopped the robot."""
        best = None
        for name, (priority, timeout) in self.sources.items():
            if name not in self._last:
                continue
            t, v, w = self._last[name]
            if now - t > timeout:
                continue
            if best is None or priority > best[0]:
                best = (priority, name, v, w)
        if best is None:
            return 0.0, 0.0, None
        return best[2], best[3], best[1]
