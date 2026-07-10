# -*- python-fmt -*-

## Copyright (c) 2024, 2026  University of Washington.
##
## Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are met:
##
## 1. Redistributions of source code must retain the above copyright notice, this
##    list of conditions and the following disclaimer.
##
## 2. Redistributions in binary form must reproduce the above copyright notice,
##    this list of conditions and the following disclaimer in the documentation
##    and/or other materials provided with the distribution.
##
## 3. Neither the name of the University of Washington nor the names of its
##    contributors may be used to endorse or promote products derived from this
##    software without specific prior written permission.
##
## THIS SOFTWARE IS PROVIDED BY THE UNIVERSITY OF WASHINGTON AND CONTRIBUTORS “AS
## IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
## IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
## DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY OF WASHINGTON OR CONTRIBUTORS BE
## LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
## CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
## GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
## HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
## LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
## OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Tests for seaglider_utils.py targeting 100% statement and branch coverage.

Notes on coverage:
    seaglider_utils.py has no `elif` branches of its own except the two
    documented below, and two genuinely unreachable dead-code blocks (flagged
    with `# pragma: no cover` in the source, matching the same pattern already
    applied to sg_l123_utils.py):

    - collect_dive_ncfiles's final `else: return []` (after the
      `if dive_files: ... elif ncdf_files: ...` chain) can never execute: if
      both dive_files and ncdf_files are empty, max_dive_num is -1 and the
      function already returns via the earlier `if max_dive_num <= 0:` check.
    - load_var's `except Exception as exception:` block immediately does
      `raise exception`, so everything after that line in the except block is
      unreachable. test_load_var_reraises_exception below exercises the
      re-raise itself.

    decode_qc's `nc_qc_type` is a module-level "Final" constant, but Final is
    not runtime-enforced, so it genuinely can be monkeypatched to "i" to
    exercise its second branch for real (unlike the dead code above).
"""

import logging
import pathlib
from typing import Any

import netCDF4
import numpy as np
import pytest
from numpy.typing import NDArray

import seaglider_utils as su

# ---------------------------------------------------------------------------
# open_netcdf_file
# ---------------------------------------------------------------------------


def test_open_netcdf_file_success(tmp_path: pathlib.Path) -> None:
    """Covers the `else: return ds` arm (no exception)."""
    path = tmp_path.joinpath("test.nc")
    ds = netCDF4.Dataset(path, "w")
    ds.close()

    result = su.open_netcdf_file(path, "r")

    assert result is not None
    result.close()


def test_open_netcdf_file_failure_with_logger(tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture) -> None:
    """Covers the except block's True arm of `if logger:`."""
    logger = logging.getLogger("test_open_netcdf_file_failure_with_logger")

    result = su.open_netcdf_file(tmp_path.joinpath("does_not_exist.nc"), "r", logger=logger)

    assert result is None
    assert any(r.levelname == "ERROR" for r in caplog.records)


def test_open_netcdf_file_failure_without_logger(tmp_path: pathlib.Path) -> None:
    """Covers the except block's False arm of `if logger:`."""
    result = su.open_netcdf_file(tmp_path.joinpath("does_not_exist.nc"), "r", logger=None)

    assert result is None


# ---------------------------------------------------------------------------
# collect_dive_ncfiles
# ---------------------------------------------------------------------------


def test_collect_dive_ncfiles_none_dir() -> None:
    """Covers the True arm of `if not mission_dir:`."""
    assert su.collect_dive_ncfiles(None) == []  # ty: ignore[invalid-argument-type]


def test_collect_dive_ncfiles_empty_dir(tmp_path: pathlib.Path) -> None:
    """Covers the True arm of `if max_dive_num <= 0:` (no matching files at all)."""
    assert su.collect_dive_ncfiles(tmp_path) == []


def test_collect_dive_ncfiles_nc_only(tmp_path: pathlib.Path) -> None:
    """Covers: dive_files ternary True, ncdf_files ternary False, `if dive_files:` True."""
    tmp_path.joinpath("p0010001.nc").touch()
    tmp_path.joinpath("p0010002.nc").touch()

    result = su.collect_dive_ncfiles(tmp_path)

    assert result == [tmp_path.joinpath("p0010001.nc"), tmp_path.joinpath("p0010002.nc")]


def test_collect_dive_ncfiles_ncdf_only(tmp_path: pathlib.Path) -> None:
    """Covers: dive_files ternary False, ncdf_files ternary True, `elif ncdf_files:` True."""
    tmp_path.joinpath("p0010001.ncdf").touch()
    tmp_path.joinpath("p0010002.ncdf").touch()

    result = su.collect_dive_ncfiles(tmp_path)

    assert result == [tmp_path.joinpath("p0010001.ncdf"), tmp_path.joinpath("p0010002.ncdf")]


def test_collect_dive_ncfiles_mixed_with_gap(tmp_path: pathlib.Path) -> None:
    """Covers the merge loop's three outcomes in one pass.

    .nc found (dive 1), .ncdf found (dive 2, since .nc is absent there), and
    neither found (dive 3, a gap).
    """
    tmp_path.joinpath("p0010001.nc").touch()
    tmp_path.joinpath("p0010002.ncdf").touch()
    tmp_path.joinpath("p0010004.nc").touch()

    result = su.collect_dive_ncfiles(tmp_path)

    assert result == [
        tmp_path.joinpath("p0010001.nc"),
        tmp_path.joinpath("p0010002.ncdf"),
        tmp_path.joinpath("p0010004.nc"),
    ]


# ---------------------------------------------------------------------------
# dive_number
# ---------------------------------------------------------------------------


def test_dive_number_success() -> None:
    """Covers the try body succeeding."""
    assert su.dive_number(pathlib.Path("p0010042.nc")) == 42


def test_dive_number_failure() -> None:
    """Covers the except block (a filename too short to contain a dive number)."""
    assert su.dive_number(pathlib.Path("bad.nc")) == -1


# ---------------------------------------------------------------------------
# decode_qc
# ---------------------------------------------------------------------------


def test_decode_qc_q_numeric() -> None:
    """Covers `nc_qc_type == "Q"` True, then `type_qc is int` True (early return)."""
    result = su.decode_qc(np.array([1, 4, 1]))
    np.testing.assert_array_equal(result, [1, 4, 1])


def test_decode_qc_q_str() -> None:
    """Covers `nc_qc_type == "Q"` True, then `elif type_qc is str:` True (decode below)."""
    result = su.decode_qc(np.array(["1", "0", "4"]))
    np.testing.assert_array_equal(result, [1.0, 0.0, 4.0])


def test_decode_qc_q_neither_raises() -> None:
    """Covers the False arm of both `type_qc is int/float` and `elif type_qc is str:`."""
    with pytest.raises(TypeError):
        su.decode_qc(np.array([True, False]))


def test_decode_qc_i_numeric(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers the False (else, 'i') arm of `nc_qc_type == "Q"`, then the int early return."""
    monkeypatch.setattr(su, "nc_qc_type", "i")

    result = su.decode_qc(np.array([1, 4, 1]))

    np.testing.assert_array_equal(result, [1, 4, 1])


def test_decode_qc_i_str(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers the False (else, 'i') arm of `nc_qc_type == "Q"`, then the str decode-below arm."""
    monkeypatch.setattr(su, "nc_qc_type", "i")

    result = su.decode_qc(np.array(["1", "0", "4"]))

    np.testing.assert_array_equal(result, [1.0, 0.0, 4.0])


def test_decode_qc_i_neither_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers the False arm of both `type_qc is int/float` and `elif type_qc is str:` under 'i'."""
    monkeypatch.setattr(su, "nc_qc_type", "i")

    with pytest.raises(TypeError):
        su.decode_qc(np.array([True, False]))


# ---------------------------------------------------------------------------
# load_var
# ---------------------------------------------------------------------------


class _FakeNcf:
    """Minimal stand-in for a netCDF4.Dataset, exposing only `.variables`."""

    def __init__(self, variables: dict[str, NDArray[Any]]) -> None:
        self.variables = variables


def test_load_var_missing_variable_no_logger() -> None:
    """Covers `if var_n not in ncf.variables:` True, then `if logger:` False."""
    ncf = _FakeNcf({})

    result = su.load_var(ncf, "temp", None, None, None, "depth", "ctd_time", "ctd_depth", logger=None)  # ty: ignore[invalid-argument-type]

    assert result == (None, None)


def test_load_var_missing_variable_with_logger(caplog: pytest.LogCaptureFixture) -> None:
    """Covers `if var_n not in ncf.variables:` True, then `if logger:` True."""
    logger = logging.getLogger("test_load_var_missing_variable_with_logger")
    ncf = _FakeNcf({"trajectory": np.array([42])})

    result = su.load_var(ncf, "temp", None, None, None, "depth", "ctd_time", "ctd_depth", logger=logger)  # ty: ignore[invalid-argument-type]

    assert result == (None, None)
    assert any(r.levelname == "WARNING" for r in caplog.records)


def test_load_var_no_qc_direct_depth() -> None:
    """Covers `var_qc_n and ...` False (var_qc_n is None) and `if var_depth_n is not None:` True."""
    ncf = _FakeNcf({"temp": np.array([1.0, 2.0, 3.0]), "depth": np.array([10.0, 20.0, 30.0])})

    var, depth = su.load_var(ncf, "temp", None, None, None, "depth", "ctd_time", "ctd_depth", logger=None)  # ty: ignore[invalid-argument-type]

    np.testing.assert_array_equal(var, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(depth, [10.0, 20.0, 30.0])


def test_load_var_qc_present_but_absent_from_variables() -> None:
    """Covers `var_qc_n and var_qc_n in ncf.variables` False via the second operand."""
    ncf = _FakeNcf({"temp": np.array([1.0, 2.0, 3.0]), "depth": np.array([10.0, 20.0, 30.0])})

    var, depth = su.load_var(ncf, "temp", "missing_qc", None, None, "depth", "ctd_time", "ctd_depth", logger=None)  # ty: ignore[invalid-argument-type]

    np.testing.assert_array_equal(var, [1.0, 2.0, 3.0])


def test_load_var_qc_valid_filters_bad_values() -> None:
    """Covers the qc block's True arm and the inner try's `else:` (decode succeeds)."""
    ncf = _FakeNcf(
        {
            "temp": np.array([1.0, 2.0, 3.0]),
            "temp_qc": np.array([1, 4, 1]),  # QC_GOOD == 1
            "depth": np.array([10.0, 20.0, 30.0]),
        }
    )

    var, depth = su.load_var(ncf, "temp", "temp_qc", None, None, "depth", "ctd_time", "ctd_depth", logger=None)  # ty: ignore[invalid-argument-type]

    np.testing.assert_array_equal(var, [1.0, np.nan, 3.0])


def test_load_var_qc_decode_failure_with_logger(caplog: pytest.LogCaptureFixture) -> None:
    """Covers the inner try's `except Exception:` and its `if logger:` True arm."""
    logger = logging.getLogger("test_load_var_qc_decode_failure_with_logger")
    ncf = _FakeNcf(
        {
            "temp": np.array([1.0, 2.0, 3.0]),
            "temp_qc": np.array([True, False, True]),
            "depth": np.array([10.0, 20.0, 30.0]),
        }
    )

    var, depth = su.load_var(ncf, "temp", "temp_qc", None, None, "depth", "ctd_time", "ctd_depth", logger=logger)  # ty: ignore[invalid-argument-type]

    # decode failed, so QC filtering was never applied
    np.testing.assert_array_equal(var, [1.0, 2.0, 3.0])
    assert any(r.levelname == "WARNING" for r in caplog.records)


def test_load_var_qc_decode_failure_without_logger() -> None:
    """Covers the inner try's `except Exception:` and its `if logger:` False arm."""
    ncf = _FakeNcf(
        {
            "temp": np.array([1.0, 2.0, 3.0]),
            "temp_qc": np.array([True, False, True]),
            "depth": np.array([10.0, 20.0, 30.0]),
        }
    )

    var, _depth = su.load_var(ncf, "temp", "temp_qc", None, None, "depth", "ctd_time", "ctd_depth", logger=None)  # ty: ignore[invalid-argument-type]

    np.testing.assert_array_equal(var, [1.0, 2.0, 3.0])


def test_load_var_depth_via_var_time_n() -> None:
    """Covers `if var_depth_n is not None:` False, then `if var_time_n in ncf.variables:` True."""
    ncf = _FakeNcf(
        {
            "temp": np.array([1.0, 2.0, 3.0]),
            "temp_time": np.array([0.0, 1.0, 2.0]),
            "ctd_time": np.array([0.0, 1.0, 2.0, 3.0]),
            "ctd_depth": np.array([10.0, 20.0, 30.0, 40.0]),
        }
    )

    var, depth = su.load_var(ncf, "temp", None, "temp_time", "truck_time", None, "ctd_time", "ctd_depth", logger=None)  # ty: ignore[invalid-argument-type]

    np.testing.assert_array_equal(var, [1.0, 2.0, 3.0])
    assert depth is not None


def test_load_var_depth_via_truck_time_n() -> None:
    """Covers `if var_time_n in ncf.variables:` False, then `elif truck_time_n in ...:` True."""
    ncf = _FakeNcf(
        {
            "temp": np.array([1.0, 2.0, 3.0]),
            "truck_time": np.array([0.0, 1.0, 2.0]),
            "ctd_time": np.array([0.0, 1.0, 2.0, 3.0]),
            "ctd_depth": np.array([10.0, 20.0, 30.0, 40.0]),
        }
    )

    var, depth = su.load_var(ncf, "temp", None, "temp_time", "truck_time", None, "ctd_time", "ctd_depth", logger=None)  # ty: ignore[invalid-argument-type]

    np.testing.assert_array_equal(var, [1.0, 2.0, 3.0])
    assert depth is not None


def test_load_var_no_time_variable_with_logger(caplog: pytest.LogCaptureFixture) -> None:
    """Covers the final `else:` (neither time variable found), with `if logger:` True."""
    logger = logging.getLogger("test_load_var_no_time_variable_with_logger")
    ncf = _FakeNcf({"temp": np.array([1.0, 2.0, 3.0])})

    var, depth = su.load_var(
        ncf,  # ty: ignore[invalid-argument-type]
        "temp",
        None,
        "temp_time",
        "truck_time",
        None,
        "ctd_time",
        "ctd_depth",
        logger=logger,
    )

    np.testing.assert_array_equal(var, [1.0, 2.0, 3.0])
    assert depth is None
    assert any(r.levelname == "ERROR" for r in caplog.records)


def test_load_var_no_time_variable_without_logger() -> None:
    """Covers the final `else:` (neither time variable found), with `if logger:` False."""
    ncf = _FakeNcf({"temp": np.array([1.0, 2.0, 3.0])})

    var, depth = su.load_var(ncf, "temp", None, "temp_time", "truck_time", None, "ctd_time", "ctd_depth", logger=None)  # ty: ignore[invalid-argument-type]

    np.testing.assert_array_equal(var, [1.0, 2.0, 3.0])
    assert depth is None


def test_load_var_reraises_exception() -> None:
    """Covers `except Exception as exception: raise exception`.

    The outer exception is re-raised unchanged (a missing depth variable key
    raises KeyError from the dict lookup).
    """
    ncf = _FakeNcf({"temp": np.array([1.0, 2.0, 3.0])})

    with pytest.raises(KeyError):
        su.load_var(ncf, "temp", None, None, None, "missing_depth", "ctd_time", "ctd_depth", logger=None)  # ty: ignore[invalid-argument-type]


# ---------------------------------------------------------------------------
# interp1_extend
# ---------------------------------------------------------------------------


def test_interp1_extend_extends_both_ends() -> None:
    """Covers the True arms of both `if t2[0] < t1[0]:` and `if t2[-1] > t1[-1]:`."""
    t1 = np.array([1.0, 2.0, 3.0])
    data = np.array([10.0, 20.0, 30.0])
    t2 = np.array([0.0, 1.5, 4.0])

    result = su.interp1_extend(t1, data, t2)

    np.testing.assert_allclose(result, [10.0, 15.0, 30.0])


def test_interp1_extend_within_range() -> None:
    """Covers the False arms of both `if t2[0] < t1[0]:` and `if t2[-1] > t1[-1]:`."""
    t1 = np.array([1.0, 2.0, 3.0])
    data = np.array([10.0, 20.0, 30.0])
    t2 = np.array([1.5, 2.5])

    result = su.interp1_extend(t1, data, t2)

    np.testing.assert_allclose(result, [15.0, 25.0])
