"""Timing that is both precise in-process and meaningful across processes.

Neither standard clock is sufficient on its own:

``time.time()``       comparable across processes, but on Windows its
                      granularity is 15.6 ms and the smallest observable delta
                      is around 0.5 ms. In-process delivery is far faster than
                      that, so measuring it this way reports zero.

``time.perf_counter()``  sub-microsecond, monotonic, and the right tool for a
                      single process. Its epoch is arbitrary and process-local,
                      so a perf_counter value published by one process is
                      meaningless when subtracted in another. A distributed
                      latency number built from it is not wrong by a constant,
                      it is noise.

So every timestamp carries both, plus the id of the process that took it.
``elapsed`` then uses the precise clock when both ends ran in the same process
and the comparable clock when they did not, and says which it used. A benchmark
that silently mixed the two would produce numbers nobody could interpret.
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from typing import Optional, Tuple

#: Identifies this process for the lifetime of the interpreter. Includes the
#: pid for readability and a uuid because pids are reused.
PROCESS_ID = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"

#: Smallest interval the wall clock can distinguish here. Used to flag results
#: that fall below what the platform can actually resolve.
WALL_RESOLUTION_S = time.get_clock_info("time").resolution


@dataclass(frozen=True)
class Stamp:
    """A moment, recorded on both clocks."""

    wall: float
    perf: float
    origin: str = PROCESS_ID

    @property
    def local(self) -> bool:
        return self.origin == PROCESS_ID


def now() -> Stamp:
    return Stamp(wall=time.time(), perf=time.perf_counter())


def elapsed(start: Stamp, end: Optional[Stamp] = None) -> Tuple[float, bool]:
    """Seconds between two stamps, and whether the precise clock was used.

    Returns (seconds, precise). ``precise`` is False when the two stamps came
    from different processes and the coarse wall clock had to be used, which is
    the caller's cue to treat a sub-millisecond result as unresolvable rather
    than real.
    """
    finish = end or now()
    if start.local and finish.local:
        return finish.perf - start.perf, True
    return finish.wall - start.wall, False


def resolvable(seconds: float, precise: bool) -> bool:
    """Whether a measured interval is above the clock's noise floor."""
    return precise or seconds >= WALL_RESOLUTION_S
