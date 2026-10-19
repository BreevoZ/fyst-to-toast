"""FYST scan classification enums, copied from FYST's semantics.

Source: fyst-trajectories/src/fyst_trajectories/trajectory.py:80-83,
commit f095c5176ab5f8b53aa19a549dc871ef2e91feeb (v0.10).
These plain integers are FYST enum values, not TOAST bitmask values.
The optional FYST contract test checks that the copied values still agree.
"""

SCAN_FLAG_UNCLASSIFIED = 0
SCAN_FLAG_SCIENCE = 1
SCAN_FLAG_TURNAROUND = 2
SCAN_FLAG_RETUNE = 3
