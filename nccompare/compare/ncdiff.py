import logging
from pathlib import Path
from typing import List, Optional, Iterable

import numpy as np
import xarray as xr

import nccompare.conf as settings
from nccompare.model import CompareResult

logger = logging.getLogger("nccompare")
PASS = "PASSED"
FAIL = "FAILED"


def compare_datasets(
    dataset1: xr.Dataset,
    dataset2: xr.Dataset,
    variables,
    last_time_step: bool,
):
    # keep only float vars
    if variables:
        dataset1_vars_list = variables
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
                CompareResult(description=f"cannot read {var} from file2: {e}")
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
                    description=f"Can't compare {var} with shapes {field1.shape} {field2.shape}"
                    # description = f"Can't compare {var} in {file1} and in {file2} with shapes {field1.shape} {field2.shape}"
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
                relative_error=f"{rel_err:.2e}",
                min_diff=f"{min_difference:.2e}",
                max_diff=f"{max_difference:.2e}",
                mask_equal=f"{mask_is_equal}",
                file1="file1",
                # file1=f"{os.path.basename(file1)}",
                file2="file2",
                # file2=f"{os.path.basename(file2)}",
                variable=f"{var}",
                description=descr,
            )
        )
    return result


def compare_files(file1: Path, file2: Path, **kwargs) -> List[CompareResult]:
    logger.info(f"Comparing {file1} with {file2}")
    dataset1, err_msg = safe_open_dataset(file1)
    if err_msg is not None:
        return [CompareResult(description=err_msg)]
    dataset2, err_msg = safe_open_dataset(file2)
    if err_msg is not None:
        return [CompareResult(description=err_msg)]

    return compare_datasets(dataset1, dataset2, **kwargs)


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
