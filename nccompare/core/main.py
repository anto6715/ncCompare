#!/usr/bin/env python

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import xarray as xr

import nccompare.conf as settings
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
    results = []
    errors_found = 0
    for reference, compares in files_to_compare.items():
        for cmp in compares:
            df = pd.DataFrame(
                [],
                columns=[
                    "Relative error",
                    "Min Diff",
                    "Max Diff",
                    "Mask Equal",
                    "Reference File",
                    "Comparison File",
                    "Variable",
                    "Description",
                ],
            )
            result = compare.compare_files(reference, cmp, variables=variables, last_time_step=last_time_step)
            for row_data in result:
                df.loc[len(df)] = list(row_data)
            df_to_print = df.drop(["Comparison File", "Reference File"], axis=1)
            print(f"\n- Reference file: {reference}")
            print(f"- Comparison file: {cmp}")
            print(df_to_print.to_string(index=False))
            # if (df["Result"] == "FAILED").any():
            #     errors_found += 1
            # results.append(df)
        #
        # if errors_found > 0:
        #     exit(1)


def load_files(directory: Path, filter_name: str) -> List[Path]:
    """Load all files within a directory if they match the filter name"""
    return [f for f in directory.glob(filter_name) if f.is_file()]
