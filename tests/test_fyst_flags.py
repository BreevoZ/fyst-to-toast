"""The single FYST enum -> TOAST bitmask mapping shared by writer and SimFYST."""

import numpy as np

from fyst_to_toast.fyst_flags import (
    SCAN_FLAG_RETUNE,
    SCAN_FLAG_SCIENCE,
    SCAN_FLAG_TURNAROUND,
    SCAN_FLAG_UNCLASSIFIED,
    to_toast_flags,
)


def test_each_enum_value_maps_to_one_mask():
    raw = np.array(
        [
            SCAN_FLAG_UNCLASSIFIED,
            SCAN_FLAG_SCIENCE,
            SCAN_FLAG_TURNAROUND,
            SCAN_FLAG_RETUNE,
            7,  # unknown values are invalid too (the loader rejects them earlier)
        ],
        dtype=np.int8,
    )
    flags = to_toast_flags(raw, turnaround_mask=4, invalid_mask=1)
    assert flags.dtype == np.uint8
    assert flags.tolist() == [1, 0, 4, 1, 1]


def test_turnaround_mask_is_configurable():
    raw = np.array([SCAN_FLAG_SCIENCE, SCAN_FLAG_TURNAROUND], dtype=np.int8)
    assert to_toast_flags(raw, turnaround_mask=8, invalid_mask=1).tolist() == [0, 8]
