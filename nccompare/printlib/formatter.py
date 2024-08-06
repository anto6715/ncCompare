from typing import Any

import numpy as np
from rich import box
from rich.console import Console
from rich.table import Table

from nccompare.model import CompareResult
from nccompare.model.comparison import Comparison

COLUMNS = ["RESULT", "MIN DIFF", "MAX DIFF", "REL ERR", "MASK", "VAR", "DESCR"]

FAILED = "[red] :x: FAILED"
PASSED = "[green] :heavy_check_mark: PASSED"


def get_result(cr: CompareResult) -> str:
    if (
        float(cr.min_diff) == 0.0
        and float(cr.max_diff) == 0.0
        and cr.mask_equal
        and float(cr.relative_error) == 0.0
    ):
        return PASSED

    return FAILED


def print_comparison(comparison: Comparison) -> None:
    console = Console()

    table = Table(
        show_header=True,
        header_style="bold blue",
        title=f"{comparison.reference_file} vs {comparison.comparison_file}",
        box=box.SIMPLE,
    )
    for column in COLUMNS:
        table.add_column(column)

    if comparison.exception is not None:
        table.add_row(
            FAILED, *["-" for _ in range(len(COLUMNS) - 1)], str(comparison.exception)
        )
    else:
        for c in comparison:
            result = get_result(c)
            table.add_row(
                f"{result}",
                render(c.min_diff),
                render(c.max_diff),
                render(c.relative_error),
                render(c.mask_equal),
                render(c.variable),
                render(c.description),
            )

    console.print(table)


def render(value: Any):
    if isinstance(value, bool):
        return str(value)

    if isinstance(value, np.timedelta64):
        return f"{value.view('int64'):.2e}"
    try:
        return f"{value:.2e}"
    except Exception:
        return str(value)
