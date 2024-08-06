#!/usr/bin/env python

import logging
from pathlib import Path
from typing import List

from nccompare import compare
from nccompare.utils.regex import find_file_matches

# settings
logger = logging.getLogger("nccompare")


def execute(
    folder1: Path,
    folder2: Path,
    filter_name: str,
    common_pattern: str,
    variables: List[str],
    last_time_step: bool,
):
    ########################
    # INPUT FILES
    ########################
    reference_input_files = load_files(folder1, filter_name)
    comparison_input_files = load_files(folder2, filter_name)

    ########################
    # FILES TO COMPARE
    ########################
    files_to_compare = find_file_matches(
        reference_input_files, comparison_input_files, common_pattern
    )

    ########################
    # COMPARISON
    ########################
    for result in compare.compare(files_to_compare, variables, last_time_step):
        print(result)


def load_files(directory: Path, filter_name: str) -> List[Path]:
    """Load all files within a directory if they match the filter name"""
    return [f for f in directory.glob(filter_name) if f.is_file()]
