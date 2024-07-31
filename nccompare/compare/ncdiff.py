import logging
from pathlib import Path
from typing import List, Iterable

import numpy as np
import xarray as xr

import nccompare.conf as settings
from nccompare.model import CompareResult
from nccompare.model.comparison import Comparison

logger = logging.getLogger("nccompare")
PASS = "PASSED"
FAIL = "FAILED"


def compare_files(
    file1: Path, file2: Path, variables: List[str], **kwargs
) -> Comparison:
    logger.info(f"Comparing {file1} with {file2}")
    comparison = Comparison(file1, file2)

    try:
        dataset1 = xr.open_dataset(file1)
        dataset2 = xr.open_dataset(file2)
        variables_to_compare = get_dataset_variables(dataset1, variables)
        comparison.extend(
            compare_datasets(dataset1, dataset2, variables_to_compare, **kwargs)
        )

    except Exception as e:
        comparison.set_exception(e)

    return comparison


def compare_datasets(
    reference: xr.Dataset,
    comparison: xr.Dataset,
    variables: List[str],
    last_time_step: bool,
) -> List[CompareResult]:
    results = []
    for var in variables:
        logger.info(f"Comparing {var}")
        try:
            field1 = reference[var]
            field2 = comparison[var]
            results.append(compare_variables(field1, field2, var, last_time_step))
        except Exception as e:
            logger.error(e)
            results.append(CompareResult(variable=var, description=str(e)))
    return results


def compare_variables(
    field1: xr.DataArray,
    field2: xr.DataArray,
    var,
    last_time_step: bool,
):
    # - drop all time steps except last one
    # - do not compare time variables
    if last_time_step:
        if "time" in var:
            return CompareResult(description="Skipping variable")
        time_dims_name = find_time_dims_name(field1.dims)
        if time_dims_name and field1.shape[0] > 1:
            field1 = field1.drop_isel(
                {time_dims_name: [t for t in range(field1.shape[0] - 1)]}
            )
        if time_dims_name and field2.shape[0] > 1:
            field2 = field2.drop_isel(
                {time_dims_name: [t for t in range(field2.shape[0] - 1)]}
            )

    # dimensions mismatch
    if field1.shape != field2.shape:
        return CompareResult(
            description=f"Can't compare {var} with different shapes: '{field1.shape}' vs '{field2.shape}'"
        )

    array1, mask_array1 = np.array(field1.values), field1.to_masked_array()
    array2, mask_array2 = np.array(field2.values), field2.to_masked_array()

    # try computing difference
    difference_field: np.ma.MaskedArray = mask_array1 - mask_array2

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

    return CompareResult(
        relative_error=rel_err,
        min_diff=min_difference,
        max_diff=max_difference,
        mask_equal=mask_is_equal,
        file1="file1",
        # file1=os.path.basename(file1),
        file2="file2",
        # file2=os.path.basename(file2),
        variable=var,
        description=descr,
    )


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


def get_dataset_variables(dataset: xr.Dataset, variables: List[str]) -> List[str]:
    """Extract all non char/str variables included dimension from a dataset"""
    ds_variables = []

    if variables is settings.DEFAULT_VARIABLES_TO_CHECK:
        variables_to_check = list(dataset.data_vars) + list(dataset.dims)
    else:
        variables_to_check = variables

    for v in variables_to_check:
        if dataset[v].dtype not in settings.DTYPE_NOT_CHECKED:
            ds_variables.append(v)

    return ds_variables
