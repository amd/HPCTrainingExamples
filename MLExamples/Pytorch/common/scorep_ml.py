"""Opt-in Score-P user-region helper for the PyTorch benchmarks.

`region(name)` is a context manager that records a Score-P user region **only** when
the process runs under `python -m scorep` with `SCOREP_ML=1` in the environment
(both set by `common/scorep_launch.sh`). In every other case it is a zero-cost
no-op, so the benchmarks run completely unchanged in normal use.

We deliberately drive Score-P with `--nopython` (see scorep_launch.sh) and rely on
these hand-placed regions: automatic Python instrumentation of PyTorch intercepts
every Python call and is far too heavy to be usable.
"""
import os
from contextlib import contextmanager

_ENABLED = os.environ.get("SCOREP_ML", "0") == "1"
if _ENABLED:
    try:
        import scorep.user as _su
    except Exception:
        _ENABLED = False


@contextmanager
def region(name):
    """Record `name` as a Score-P user region when enabled, else do nothing."""
    if _ENABLED:
        with _su.region(name):
            yield
    else:
        yield
