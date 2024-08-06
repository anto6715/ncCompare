from typing import NamedTuple

import numpy as np

PASSED = True
FAILED = False


class CompareResult(NamedTuple):
    relative_error: float = np.nan
    min_diff: float = np.nan
    max_diff: float = np.nan
    mask_equal: bool = False
    variable: str = ""
    description: str = "-"
