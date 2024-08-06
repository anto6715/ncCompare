from rich import box
from rich.console import Console
from rich.table import Table

from nccompare.model.comparison import Comparison

COLUMNS = ["RESULT", "MIN DIFF", "MAX DIFF", "REL ERR", "MASK EQUAL" "VAR", "DESCR"]


def print_comparison(to_print: Comparison) -> None:
    console = Console()

    table = Table(
        show_header=True,
        header_style="bold magenta",
        title=f"{to_print.reference_file} vs {to_print.reference_file}",
        box=box.SIMPLE_HEAD,
    )
    for column in COLUMNS:
        table.add_column(column, justify="center")

    for res in to_print:
        table.add_row(*[str(field) for field in res])

    console.print(table)
