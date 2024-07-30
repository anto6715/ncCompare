#!/usr/bin/env python

__author__ = "Antonio Mariani"
__email__ = "antonio.mariani@cmcc.it"
__version__ = "0.1.0"

import logging
import os
import re
from collections import namedtuple
from collections.abc import Sequence, Iterable
from typing import Optional, List

import numpy as np
import pandas as pd
import xarray as xr

# settings
logging.basicConfig(
    level=logging.WARNING,
    format="[%(asctime)s] - %(levelname)s: %(message)s",
    # format='[%(asctime)s] - %(levelname)s - %(name)s.%(funcName)s: %(message)s',
    handlers=[
        # logging.FileHandler('example.log')    # to create a log files
        logging.StreamHandler()
    ],
)
logger = logging.getLogger("compare")

PASS = "PASSED"
FAIL = "FAILED"

DTYPE_NOT_CHECKED = ["S8", "S1", "O"]  # S8|S1:char, O: string
TIME_DTYPE = ["datetime64[ns]", "<M8[ns]"]

COMPARE_MESSAGE = namedtuple(
    "CompareMessage",
    [
        "result",
        "relative_error",
        "min_diff",
        "max_diff",
        "mask_equal",
        "file1",
        "file2",
        "variable",
        "description",
    ],
    defaults=["-" for _ in range(9)],
)


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


def get_file_list_to_compare_with_match(
    sequence1: Sequence, sequence2, match_pattern: str
):
    """
    Given a regex match pattern, it finds a fileX in sequence1 that match the regex expression,
        then try to find in sequence2 a fileY that match the regex pattern with the same value of fileX.
        Example:
            match_pattern = \d{7}_\d{8}
            fileX = MO_PR_PF_*3901977_20221201*_p01_20221201_20221201.nc
            fileY = GL_LATEST_PR_PF_*3901977_20221201*.nc
        Between ** character has been highlighted the pattern that match with regex match_patten
    Args:
        sequence1: An iterable where find the files that match with match_pattern
        sequence2: An iterable where find the files that match match_pattern with the same values of the files
            from sequence1
        match_pattern: A regex expression

    Returns: A list of tuple. Each tuple contains two files that have the same value in correspondence of match_pattern
    """
    not_found = 0
    match_list = list()
    for file in sequence1:
        matching_strings = get_match(match_pattern, file)
        if matching_strings:
            match_found = False
            for file2 in sequence2:
                filename2 = os.path.basename(file2)
                if all_match_are_satisfied(matching_strings, filename2):
                    match_found = True
                    logger.debug("Found two files that match")
                    logger.debug(f"\t- {file}")
                    logger.debug(f"\t- {file2}")
                    match_list.append((file, file2))
            if not match_found:
                print(f"{FAIL} No matching found for {file}")
                not_found += 1
        else:
            logger.debug(f"The file {file} doesn't match with {match_pattern}")

    return match_list, not_found


def walklevel(input_dir, level=1):
    """Generator function, usage: for root, dirs, files in walklevel: # do something"""
    input_dir = input_dir.rstrip(os.path.sep)
    assert os.path.isdir(input_dir)
    num_sep = input_dir.count(os.path.sep)
    max_depth_level = (
        num_sep + level - 1
    )  # -1 used to have the same behavior of find bash command
    for root, dirs, files in os.walk(input_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        # level < 0 is a special case used to explore all the subdirs
        if num_sep_this >= max_depth_level and level > 0:
            del dirs[:]


def get_file_list_to_compare(nc_file_list1: list, nc_file_list2: list):
    not_found = 0
    file_list_to_compare = list()
    for file1 in nc_file_list1:
        filename1 = os.path.basename(file1)
        match_file2 = [file2 for file2 in nc_file_list2 if filename1 in file2]
        if len(match_file2) == 0:
            print(f"{FAIL} No matching found for {file1}")
            not_found += 1
        else:
            file_list_to_compare += [(file1, file2) for file2 in match_file2]

    return file_list_to_compare, not_found


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
        return [COMPARE_MESSAGE(description=err_msg)]
    dataset2, err_msg = safe_open_dataset(file2)
    if err_msg is not None:
        return [COMPARE_MESSAGE(description=err_msg)]

    # keep only float vars
    if variables_to_compare:
        dataset1_vars_list = variables_to_compare
    else:
        dataset1_vars_list, err_msg = get_dataset_variables(dataset1)

    if err_msg is not None:
        # an error message has been already printed at this point
        return [COMPARE_MESSAGE(description=err_msg)]

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
                COMPARE_MESSAGE(description=f"cannot read {var} from {file2}: {e}")
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
                COMPARE_MESSAGE(
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
                COMPARE_MESSAGE(
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
            COMPARE_MESSAGE(
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

    if field2.dtype in TIME_DTYPE:
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

    if field2.dtype in TIME_DTYPE:
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
                if dataset[var_name].dtype not in DTYPE_NOT_CHECKED
            ]
        )
    except Exception as e:
        return None, f"Cannot extract variables from {dataset}: {e}"

    try:
        variables.extend(
            [
                var_name
                for var_name in dataset.dims
                if dataset[var_name].dtype not in DTYPE_NOT_CHECKED
            ]
        )
    except Exception as e:
        return None, f"Cannot extract dimensions from {dataset}: {e}"

    return variables, None


def execute(
    folder1: str,
    folder2: str,
    filter_name: str,
    common_pattern: str,
    maxdepth: int,
    variables: list,
    last_time_step: bool,
):
    # read input file list
    nc_file_list1 = load_file_list(filter_name, folder1, maxdepth)
    nc_file_list2 = load_file_list(filter_name, folder2, maxdepth)

    # filter file list to compare
    if common_pattern is None:
        files_to_compare, not_found = get_file_list_to_compare(
            nc_file_list1, nc_file_list2
        )
    else:
        files_to_compare, not_found = get_file_list_to_compare_with_match(
            nc_file_list1, nc_file_list2, common_pattern
        )

    if len(files_to_compare) == 0:
        logger.error("No files to compare")
        exit(1)

    # start comparison
    results = []
    errors_found = not_found
    for file1, file2 in files_to_compare:
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
        result = compare_datasets(file1, file2, variables, last_time_step)
        for row_data in result:
            df.loc[len(df)] = list(row_data)
        df_to_print = df.drop(["Comparison File", "Reference File"], axis=1)
        print(f"\n- Reference file: {file1}")
        print(f"- Comparison file: {file2}")
        print(df_to_print.to_string(index=False))
        if (df["Result"] == "FAILED").any():
            errors_found += 1
        results.append(df)

    if errors_found > 0:
        exit(1)


def load_file_list(filter_name, folder1, maxdepth):
    nc_file_list1 = [
        os.path.join(root, f)
        for root, dirs, files in walklevel(folder1, maxdepth)
        for f in files
        if get_match(filter_name, os.path.join(root, f))
    ]
    return nc_file_list1
