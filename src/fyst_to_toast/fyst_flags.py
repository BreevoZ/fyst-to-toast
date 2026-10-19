"""FYST scan classification enums, copied from FYST's semantics.

Source: fyst-trajectories/src/fyst_trajectories/trajectory.py:80-83,
commit f095c5176ab5f8b53aa19a549dc871ef2e91feeb (v0.10).
These plain integers are FYST enum values, not TOAST bitmask values.
The optional FYST contract test checks that the copied values still agree.
"""

import numpy as np

SCAN_FLAG_UNCLASSIFIED = 0
SCAN_FLAG_SCIENCE = 1
SCAN_FLAG_TURNAROUND = 2
SCAN_FLAG_RETUNE = 3


def to_toast_flags(raw, *, turnaround_mask, invalid_mask):
    """Map FYST enum values to a TOAST uint8 shared-flag bitmask.

    SCIENCE -> 0, TURNAROUND -> turnaround_mask, any other value -> invalid_mask.
    The TOAST mask values are passed in, so this module never imports TOAST.
    """
    raw = np.asarray(raw)
    flags = np.zeros(raw.shape, dtype=np.uint8)
    flags[raw == SCAN_FLAG_TURNAROUND] = turnaround_mask
    flags[~np.isin(raw, (SCAN_FLAG_SCIENCE, SCAN_FLAG_TURNAROUND))] = invalid_mask
    return flags
