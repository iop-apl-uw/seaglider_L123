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

"""Tests for sg_l123_utils.py targeting 100% statement and branch coverage.

Notes on coverage:
    sg_l123_utils.py contains a single except block, inside the
    `if __name__ == "__main__":` guard at the bottom of the file. It has no
    `elif` branches; all conditionals are plain `if`/`if-else` statements, each
    of which is exercised for both its True and False outcomes below.

    One branch is unreachable dead code under the current source and cannot be
    triggered by any test: `DEBUG_PDB` in the `__main__` guard is a local
    variable hardcoded to `False` immediately before the `try/except`, so the
    `if DEBUG_PDB:` body (pdb.post_mortem on exceptions) can never execute.
    test_dunder_main_exception below exercises the `except Exception:` block
    itself (and the False side of `if DEBUG_PDB:`), which is the only part of
    that branch that is actually reachable.
"""

import argparse
import pathlib
import runpy
import sys
from unittest.mock import Mock

import numpy as np
import pytest

import sg_l123_utils
import utils

# ---------------------------------------------------------------------------
# interp1
# ---------------------------------------------------------------------------


def test_interp1_no_extrapolate_fills_nan() -> None:
    """Covers the False arm of the `"extrapolate" if extrapolate else np.nan` ternary."""
    x = np.array([0.0, 1.0, 2.0, 3.0])
    v = np.array([0.0, 10.0, 20.0, 30.0])
    xq = np.array([0.5, 1.5, 5.0])  # 5.0 is outside the range of x

    result = sg_l123_utils.interp1(x, v, xq, extrapolate=False)

    assert result[0] == pytest.approx(5.0)
    assert result[1] == pytest.approx(15.0)
    assert np.isnan(result[2])


def test_interp1_extrapolate_fills_value() -> None:
    """Covers the True arm of the `"extrapolate" if extrapolate else np.nan` ternary."""
    x = np.array([0.0, 1.0, 2.0, 3.0])
    v = np.array([0.0, 10.0, 20.0, 30.0])
    xq = np.array([0.5, 5.0])  # 5.0 is outside the range of x

    result = sg_l123_utils.interp1(x, v, xq, extrapolate=True)

    assert result[0] == pytest.approx(5.0)
    assert not np.isnan(result[1])


# ---------------------------------------------------------------------------
# running_average_non_uniform
# ---------------------------------------------------------------------------


def test_running_average_non_uniform_1d_x_returns_none() -> None:
    """A 1-d x array is not supported and hits the early `return (None, None, None)`."""
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0])
    data = np.array([1.0, 2.0, 3.0])

    result = sg_l123_utils.running_average_non_uniform(x, y, data, DX=1.0, DY=1.0, FF=0.9)

    assert result == (None, None, None)


def test_running_average_non_uniform_tiles_1d_y_no_plots() -> None:
    """A 1-d y is tiled to match x (the `if len(np.shape(y)) == 1:` True arm).

    args is left as None here, covering the False arm of every `if args is not None:`
    check in the function.
    """
    n_cols = 30
    dense_row = np.arange(n_cols, dtype=np.float64)
    sparse_row = np.full(n_cols, np.nan)
    sparse_row[[0, 1]] = [0.0, 1.0]

    x = np.vstack([dense_row, sparse_row, dense_row, np.full(n_cols, np.nan), dense_row])
    y_1d = np.linspace(0, 300, n_cols)  # 1-d, forces tiling
    data = np.sin(x / 5.0) * 10 + 50
    data[3, :] = np.nan

    data_avg, data_std, grid = sg_l123_utils.running_average_non_uniform(
        x, y_1d, data, DX=5.0, DY=30.0, FF=0.9, args=None
    )

    assert data_avg is not None
    assert data_std is not None
    assert grid is not None
    assert data_avg.shape == (5, n_cols)
    assert data_std.shape == (5, n_cols)
    assert len(grid) == 5


def test_running_average_non_uniform_full_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercises every remaining branch in a single, deliberately heterogeneous dataset.

    Five profile rows are constructed so that, across the run:
        - `if len(i1) > 10:` (x gap-fill) is hit True (dense rows) and False (sparse/empty rows)
        - `if len(i1) > 2:` / `if len(ii1) > 2:` (x_tmp2 construction) hit True and False
        - `if len(iii) > 2:` (data_avg1 assembly) hits True (3 overlapping dense rows in the
          first half of the depth range) and False (only 2 rows overlap in the second half,
          since one dense row's y only spans half the depth range)
        - `if len(jjj) > 2:` (final data_avg assembly) hits True (rows with fully finite y)
          and False (a row with almost entirely NaN y)
        - `if tmp_tmp.size != 0:` hits True (populated bins) and False (empty bins)
        - every `if args is not None:` check hits True, via a real argparse.Namespace
          with plot_heatmap mocked out to avoid writing real html/webp files
    """
    n_cols = 30
    dense_row = np.arange(n_cols, dtype=np.float64)
    sparse_row = np.full(n_cols, np.nan)
    sparse_row[[0, 1]] = [0.0, 1.0]
    empty_row = np.full(n_cols, np.nan)

    x = np.vstack([dense_row, sparse_row, dense_row, empty_row, dense_row])

    y_full = np.linspace(0, 300, n_cols)
    y_sparse = np.full(n_cols, np.nan)
    y_sparse[0] = 5.0
    y_partial = np.linspace(0, 150, n_cols)  # only covers half the depth range
    y = np.vstack([y_full, y_sparse, y_full, y_full, y_partial])

    data = np.sin(x / 5.0) * 10 + 50
    data[3, :] = np.nan

    mock_plot_heatmap = Mock()
    monkeypatch.setattr(sg_l123_utils, "plot_heatmap", mock_plot_heatmap)

    data_avg, data_std, grid = sg_l123_utils.running_average_non_uniform(
        x, y, data, DX=5.0, DY=30.0, FF=0.9, args=argparse.Namespace(interactive=False)
    )

    # 5 distinct diagnostic heatmaps are generated when args is not None
    assert mock_plot_heatmap.call_count == 5

    assert data_avg is not None
    assert data_std is not None
    assert grid is not None
    x_ref, y_ref, data_avg0, data_std0, data_N0 = grid
    assert np.isfinite(data_avg0).any()
    assert np.isnan(data_avg0).any()
    assert np.isfinite(data_avg).any()
    assert np.isnan(data_avg).any()


# ---------------------------------------------------------------------------
# bindata
# ---------------------------------------------------------------------------


def test_bindata_all_nan_sigma_true() -> None:
    """Covers `if not idx.any():` True, and the nested `if sigma:` True."""
    bins = np.arange(0.0, 10.1, 1.0).astype(np.float64)
    x = np.array([1.0, 2.0, 3.0])
    y = np.full(3, np.nan)

    b, n, s = sg_l123_utils.bindata(x, y, bins, sigma=True)

    assert np.isnan(b).all()
    assert np.isnan(n).all()
    assert s is not None
    assert np.isnan(s).all()


def test_bindata_all_nan_sigma_false() -> None:
    """Covers `if not idx.any():` True, and the nested `if sigma:` False (else)."""
    bins = np.arange(0.0, 10.1, 1.0).astype(np.float64)
    x = np.array([1.0, 2.0, 3.0])
    y = np.full(3, np.nan)

    b, n, s = sg_l123_utils.bindata(x, y, bins, sigma=False)

    assert np.isnan(b).all()
    assert np.isnan(n).all()
    assert s is None


def test_bindata_valid_data_sigma_true() -> None:
    """Covers `if not idx.any():` False (main path), and `if sigma:` True."""
    bins = np.arange(0.0, 10.1, 1.0).astype(np.float64)
    x = np.array([1.0, 1.5, 2.5, 3.5, 9.9])
    y = np.array([10.0, 12.0, 20.0, 30.0, 99.0])

    b, n, s = sg_l123_utils.bindata(x, y, bins, sigma=True)

    assert b[1] == pytest.approx(11.0)  # mean of the two points in bin 1
    assert n[1] == pytest.approx(2.0)
    assert s is not None
    assert s[1] == pytest.approx(1.0)


def test_bindata_valid_data_sigma_false() -> None:
    """Covers `if not idx.any():` False (main path), and `if sigma:` False (else)."""
    bins = np.arange(0.0, 10.1, 1.0).astype(np.float64)
    x = np.array([1.0, 1.5, 2.5, 3.5, 9.9])
    y = np.array([10.0, 12.0, 20.0, 30.0, 99.0])

    b, n, s = sg_l123_utils.bindata(x, y, bins, sigma=False)

    assert b[1] == pytest.approx(11.0)
    assert n[1] == pytest.approx(2.0)
    assert s is None


# ---------------------------------------------------------------------------
# find_gaps
# ---------------------------------------------------------------------------


def test_find_gaps() -> None:
    """Covers the (branch-free) statements in find_gaps."""
    gap_vector = np.array([1.0, 1.0, 1.0])
    data = np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0, 0.0])

    result = sg_l123_utils.find_gaps(gap_vector, data)

    assert len(result) == len(data)
    assert result[2] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_default_verbosity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture
) -> None:
    """Covers the False (else) arm of `"debug" if args.verbose else "info"`."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["sg_l123_utils.py"])

    retval = sg_l123_utils.main()

    assert retval == 0
    capsys.readouterr()


def test_main_verbose(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture) -> None:
    """Covers the True arm of `"debug" if args.verbose else "info"`."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["sg_l123_utils.py", "--verbose"])

    retval = sg_l123_utils.main()

    assert retval == 0
    capsys.readouterr()


# ---------------------------------------------------------------------------
# `if __name__ == "__main__":` guard, including the except block
# ---------------------------------------------------------------------------


def test_dunder_main_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture
) -> None:
    """Runs the module as __main__ with no error: covers the try body and `sys.exit(0)`."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["sg_l123_utils.py"])

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(sg_l123_utils.__file__, run_name="__main__")

    assert excinfo.value.code == 0
    capsys.readouterr()


def test_dunder_main_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture
) -> None:
    """Forces main() to raise, covering `except Exception:` and the False arm of `if DEBUG_PDB:`.

    `utils.init_logger` is patched to raise. Since sg_l123_utils imports it with
    `from utils import init_logger`, re-running the module via runpy under
    run_name="__main__" re-binds that name from the (already-patched) cached
    `utils` module, so the freshly executed `main()` calls the raising stub.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["sg_l123_utils.py"])

    def raise_runtime_error(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(utils, "init_logger", raise_runtime_error)

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(sg_l123_utils.__file__, run_name="__main__")

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "Exception in main" in captured.err
    assert "RuntimeError: boom" in captured.err
