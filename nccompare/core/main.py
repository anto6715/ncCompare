#!/usr/bin/env python

import logging
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import xarray as xr

import nccompare.conf as settings
from nccompare.model import CompareResult

# settings
logger = logging.getLogger("nccompare")

PASS = "PASSED"
FAIL = "FAILED"


def all_match_are_satisfied(matching_strings: tuple, file2: str):
    if len(matching_strings) == 0:
        raise ValueError("Matching string list is empty")
    for match in matching_strings:
        if match not in file2:
            return False
        else:
            logger.debug(f"Found {match} in {file2}")

    return True


def get_match(pattern, string):
    if pattern[0] != "(" or pattern[-1] != ")":
        pattern = f"({pattern})"  # force to use group refex as search
    match_object = re.search(pattern, string)
    if match_object is not None:
        return match_object.groups()
    else:
        return None


def common_pattern_exists(first_str: str, second_str: str, pattern: str) -> bool:
    """
    Check if pattern exists in first string and second string and if they match exactly the same string

    Example:
        first_str = "mfs-eas8_20150101_grid_T.nc"
        second_str = "my-simu_20150101_grid_T.nc"
        pattern = "\d{8}"

    Args:
        first_str: First string
        second_str: Second string
        pattern: regex pattern

    Returns:
        True if pattern match both first_str and second_str and the matched string is the same.
    """
    if pattern is None:
        return False

    regex = re.compile(pattern)

    match1 = regex.findall(first_str)
    match2 = regex.findall(second_str)

    if match1 and match2:
        return sorted(match1) == sorted(match2)

    return False


def safe_open_dataset(input_file: str) -> (Optional[xr.Dataset], str):
    try:
        return xr.open_dataset(input_file), None
    except FileNotFoundError:
        err_msg = "File not found or path is incorrect."
    except OSError as e:
        err_msg = f"An OS error occurred: {e}"
    except ValueError as e:
        err_msg = f"Value error occurred: {e}"
    except RuntimeError as e:
        err_msg = f"Runtime error occurred: {e}"
    except KeyError as e:
        err_msg = f"Key error occurred: {e}"
    except IndexError as e:
        err_msg = f"Index error occurred: {e}"
    except Exception as e:
        err_msg = f"An unexpected error occurred: {e}"

    # in case of exception
    return None, err_msg


def find_time_dims_name(dims: Iterable) -> str:
    time_dims_name = [dim for dim in dims if "time" in dim]
    if len(time_dims_name) == 0:
        return None
    if len(time_dims_name) > 1:
        raise ValueError(
            f"Found more than 1 time dimension: {', '.join(time_dims_name)}"
        )
    return time_dims_name.pop()


def compare_datasets(
    file1, file2, variables_to_compare: list, last_time_step: bool
) -> List:
    logger.info(f"Comparing {file1} with {file2}")
    dataset1, err_msg = safe_open_dataset(file1)
    if err_msg is not None:
        return [CompareResult(description=err_msg)]
    dataset2, err_msg = safe_open_dataset(file2)
    if err_msg is not None:
        return [CompareResult(description=err_msg)]

    # keep only float vars
    if variables_to_compare:
        dataset1_vars_list = variables_to_compare
    else:
        dataset1_vars_list, err_msg = get_dataset_variables(dataset1)

    if err_msg is not None:
        # an error message has been already printed at this point
        return [CompareResult(description=err_msg)]

    logger.debug(f"Variables to check: {dataset1_vars_list}")

    result = []
    for var in dataset1_vars_list:
        logger.info(f"Checking {var}")

        field1 = dataset1[var]

        # missing variables in comparison file
        try:
            field2 = dataset2[var]
        except Exception as e:
            result.append(
                CompareResult(description=f"cannot read {var} from {file2}: {e}")
            )
            continue

        # if last_time_step option is used:
        # - drop all time steps except last one
        # - do not compare time variables
        if last_time_step:
            time_dims_name = find_time_dims_name(field1.dims)
            if time_dims_name and field1.shape[0] > 1:
                field1 = field1.drop_isel(
                    {time_dims_name: [t for t in range(field1.shape[0] - 1)]}
                )
            if time_dims_name and field2.shape[0] > 1:
                field2 = field2.drop_isel(
                    {time_dims_name: [t for t in range(field2.shape[0] - 1)]}
                )
            if "time" in var:
                continue

        # dimensions mismatch
        if field1.shape != field2.shape:
            result.append(
                CompareResult(
                    description=f"Can't compare {var} in {file1} and in {file2} with shapes {field1.shape} {field2.shape}"
                )
            )
            continue

        array1, mask_array1 = np.array(field1.values), field1.to_masked_array()
        array2, mask_array2 = np.array(field2.values), field2.to_masked_array()

        # try computing difference
        try:
            difference_field: np.ma.MaskedArray = mask_array1 - mask_array2
        except Exception as e:
            result.append(
                CompareResult(
                    description=f"an unknown error occurs while comparing {var}: {e}"
                )
            )
            continue

        # get statistics
        max_difference = float(difference_field.max())
        min_difference = float(difference_field.min())
        mask_is_equal = np.array_equal(mask_array1.mask, mask_array2.mask)

        if min_difference is np.nan and max_difference is np.nan:
            min_difference = 0
            max_difference = 0
            descr = "WARNING all nan values found - comparison is not possible"
            rel_err = 0
        else:
            rel_err = compute_relative_error(difference_field, field2)
            descr = "-"

        check_result = FAIL
        if (
            min_difference == 0
            and max_difference == 0
            and mask_is_equal
            and rel_err == 0
        ):
            check_result = PASS

        result.append(
            CompareResult(
                result=check_result,
                relative_error=f"{rel_err:.2e}",
                min_diff=f"{min_difference:.2e}",
                max_diff=f"{max_difference:.2e}",
                mask_equal=f"{mask_is_equal}",
                file1=f"{os.path.basename(file1)}",
                file2=f"{os.path.basename(file2)}",
                variable=f"{var}",
                description=descr,
            )
        )
    return result


def compute_relative_error(diff: np.ma.MaskedArray, field2: xr.DataArray):
    diff_no_nan = np.nan_to_num(diff, nan=0)
    if np.all(diff_no_nan == 00):
        return 0.0

    if field2.dtype in settings.TIME_DTYPE:
        field2_values = field2.values.view("int64")
    else:
        field2_values = field2.values

    array2_no_nan = np.nan_to_num(field2_values, nan=0)
    try:
        # Suppress division by zero and invalid value warnings
        with np.errstate(divide="ignore", invalid="ignore"):
            array2_abs = np.abs(array2_no_nan)
            rel_err = np.max(diff_no_nan / array2_abs)
    except Exception as e:
        logger.warning(f"An error occurred when computing relative error: {e}")
        rel_err = np.nan

    if field2.dtype in settings.TIME_DTYPE:
        return rel_err / np.timedelta64(1, "s")
    return rel_err


def get_dataset_variables(dataset: xr.Dataset):
    """Extract all non char/str variables included dimension from a dataset"""
    variables = []
    try:
        variables.extend(
            [
                var_name
                for var_name in dataset.data_vars
                if dataset[var_name].dtype not in settings.DTYPE_NOT_CHECKED
            ]
        )
    except Exception as e:
        return None, f"Cannot extract variables from {dataset}: {e}"

    try:
        variables.extend(
            [
                var_name
                for var_name in dataset.dims
                if dataset[var_name].dtype not in settings.DTYPE_NOT_CHECKED
            ]
        )
    except Exception as e:
        return None, f"Cannot extract dimensions from {dataset}: {e}"

    return variables, None


def find_not_common_files(reference_files, comparison_files) -> (List[str], List[str]):
    """

    Args:
        reference_files: Reference files
        comparison_files: Files to compare with reference files

    Returns:
        A list of files in reference_files but not in comparison_files, \
        a list if files in comparison_files but not in reference_files
    """

    reference_filenames = {f.name for f in reference_files}
    comparison_filenames = {f.name for f in comparison_files}

    missing_filenames = reference_filenames - comparison_filenames
    not_expected_filenames = comparison_filenames - reference_filenames

    missing_files = [f for f in reference_files if f.name in missing_filenames]
    not_expected_files = [
        f for f in comparison_files if f.name in not_expected_filenames
    ]
    return missing_files, not_expected_files


def find_file_matches(
    reference_input_files: List[Path],
    comparison_input_files: List[Path],
    common_pattern: str = None,
) -> Dict[Path, List[Path]]:
    """
    For each file in reference_input_files,
    return a list of file with the same filename or with the same substring matching common_pattern.
    If no match is found, an empty list is associated to that file.

    Args:
        reference_input_files: List of reference input files
        comparison_input_files: List of files to compare with reference input files
        common_pattern: regex expression to identify a common substring between two files

    Returns:

    """
    to_compare = dict()
    for ref in reference_input_files:
        to_compare[ref] = []
        for cmp in comparison_input_files:
            if ref.name == cmp.name or common_pattern_exists(
                ref.name, cmp.name, common_pattern
            ):
                to_compare[ref].append(cmp)

    return to_compare


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
                    "Result",
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
            result = compare_datasets(reference, cmp, variables, last_time_step)
            for row_data in result:
                df.loc[len(df)] = list(row_data)
            df_to_print = df.drop(["Comparison File", "Reference File"], axis=1)
            print(f"\n- Reference file: {reference}")
            print(f"- Comparison file: {cmp}")
            print(df_to_print.to_string(index=False))
            if (df["Result"] == "FAILED").any():
                errors_found += 1
            results.append(df)

        if errors_found > 0:
            exit(1)


def load_files(directory: Path, filter_name: str) -> List[Path]:
    """Load all files within a directory if they match the filter name"""
    return [f for f in directory.glob(filter_name) if f.is_file()]
