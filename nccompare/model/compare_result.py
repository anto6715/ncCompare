from pathlib import Path
from typing import NamedTuple

PASSED = True
FAILED = False


class CompareResult(NamedTuple):
    relative_error: float = "-"
    min_diff: float = "-"
    max_diff: float = "-"
    mask_equal: bool = "-"
    file1: Path = "-"
    file2: Path = "-"
    variable: str = "-"
    description: str = "-"
