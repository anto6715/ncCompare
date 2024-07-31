from pathlib import Path
from typing import List, Optional

from nccompare.model import CompareResult


class Comparison:
    def __init__(self, reference_file: Path, comparison_file: Path):
        self.reference_file = reference_file
        self.comparison_file = comparison_file
        self._compare_results: List[CompareResult] = []
        self._exception: Optional[Exception] = None

    def __len__(self) -> int:
        return len(self._compare_results)

    def __getitem__(self, position) -> CompareResult:
        return self._compare_results[position]

    def append(self, result: CompareResult) -> None:
        self._compare_results.append(result)

    def extend(self, results: List[CompareResult]) -> None:
        self._compare_results.extend(results)

    def set_exception(self, exception: Exception) -> None:
        self._exception = exception

    def __str__(self) -> str:
        result_count = len(self._compare_results)

        title = f"Comparison between {self.reference_file} and {self.comparison_file}"
        subtitle = f"\n\t- Number of results: {result_count}"
        if result_count > 0:
            result_info = "\n\t- Results:"
            for result in self._compare_results:
                result_info += f"\n\t\t- {result}"
        else:
            result_info = ""
        exception_info = f"\n\t- Exception: {self._exception}" if self._exception else ""

        return (
            f"{title}"
            f"{subtitle}"
            f"{result_info}"
            f"{exception_info}"
            "\n"
        )

