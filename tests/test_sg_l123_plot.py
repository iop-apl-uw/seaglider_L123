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

"""Tests for sg_l123_plot.py targeting 100% statement and branch coverage.

Notes on coverage:
    sg_l123_plot.py has no `elif` branches; all conditionals are plain
    `if`/`if-else` statements. It has three except clauses, all inside the
    `if __name__ == "__main__":` guard at the bottom of the file:
    `except SystemExit: pass` and `except Exception:` in the guard itself, and
    the `if DEBUG_PDB:` inside DEBUG_PDB_F (which the `except Exception:`
    clause calls). All three are exercised below.

    Unlike sg_l123_utils.py, DEBUG_PDB here is a real module-level global that
    main() mutates via `global DEBUG_PDB`, so (unlike the hardcoded-False dead
    code in sg_l123_utils.py) both arms of `if DEBUG_PDB:` are genuinely
    reachable and are both tested directly against DEBUG_PDB_F() below.

    `sys.exit(0)` at the end of the `__main__` guard is unconditional (outside
    the try/except), so all three exception scenarios below expect SystemExit
    with code 0 -- that return code does not, by itself, distinguish success
    from failure in this script.
"""

import pathlib
import runpy
import sys
from unittest.mock import Mock

import numpy as np
import pytest
import xarray as xr

import sg_l123_plot
import utils

# ---------------------------------------------------------------------------
# cmocean_to_plotly
# ---------------------------------------------------------------------------


def test_cmocean_to_plotly() -> None:
    """Converts a cmocean colormap into a plotly colorscale of [position, rgb-string] pairs."""
    import cmocean

    colorscale = sg_l123_plot.cmocean_to_plotly(cmocean.cm.thermal, 4)  # ty: ignore[unresolved-attribute]

    assert len(colorscale) == 4
    assert colorscale[0][0] == pytest.approx(0.0)
    assert colorscale[-1][0] == pytest.approx(1.0)
    for _position, rgb in colorscale:
        assert isinstance(rgb, str)
        assert rgb.startswith("rgb(")


# ---------------------------------------------------------------------------
# DEBUG_PDB_F
# ---------------------------------------------------------------------------


def test_debug_pdb_f_true_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers the True arm of `if DEBUG_PDB:`, with pdb.post_mortem stubbed out."""
    monkeypatch.setattr(sg_l123_plot, "DEBUG_PDB", True)
    mock_post_mortem = Mock()
    monkeypatch.setattr("pdb.post_mortem", mock_post_mortem)

    try:
        raise ValueError("boom")
    except ValueError:
        sg_l123_plot.DEBUG_PDB_F()

    mock_post_mortem.assert_called_once()


def test_debug_pdb_f_false_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers the False arm of `if DEBUG_PDB:` (the debugger is never entered)."""
    monkeypatch.setattr(sg_l123_plot, "DEBUG_PDB", False)
    mock_post_mortem = Mock()
    monkeypatch.setattr("pdb.post_mortem", mock_post_mortem)

    try:
        raise ValueError("boom")
    except ValueError:
        sg_l123_plot.DEBUG_PDB_F()

    mock_post_mortem.assert_not_called()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _write_l2_l3_files(l123_dir: pathlib.Path, base_name: str = "test") -> None:
    """Writes a full-featured L2 dataset and a reduced L3 dataset under l123_dir.

    The L2 dataset has a "dive" variable and GPS position variables (covering
    the True arms of `if "dive" in dsi.variables and plot_dives:` and
    `if "lon_gps" in dsi.variables:`), and two plotted variables -- one with a
    non-empty "description" attribute and one with an empty one (covering both
    arms of `if descr:`). The L3 dataset omits "dive" and the GPS variables
    (covering the False arms of those same two checks), and only one variable
    is present in each dataset out of all of `plot_vars`, so most of
    `plot_vars` misses on `if var_n not in dsi.variables:` (True arm) while
    "T" hits (False arm).
    """
    n_z = 5
    n_prof = 4
    z = np.linspace(0.0, 100.0, n_z)
    rng = np.random.default_rng(0)

    l2_ds = xr.Dataset(
        {
            "z": ("z", z),
            "dive": ("profile", np.array([1, 1, 2, 2], dtype=np.int32)),
            "T": (("profile", "z"), rng.normal(10, 1, (n_prof, n_z)), {"description": "Temperature"}),
            "S": (("profile", "z"), rng.normal(35, 1, (n_prof, n_z)), {"description": ""}),
            "lon_gps": ("profile", np.array([-122.1, -122.2, -122.3, -122.4])),
            "lat_gps": ("profile", np.array([47.1, 47.2, 47.3, 47.4])),
            "lon_profile": ("profile", np.array([-122.1, -122.2, -122.3, -122.4])),
            "lat_profile": ("profile", np.array([47.1, 47.2, 47.3, 47.4])),
            "lon_dive": ("profile", np.array([-122.1, -122.2, -122.3, -122.4])),
            "lat_dive": ("profile", np.array([47.1, 47.2, 47.3, 47.4])),
        }
    )
    l2_ds.to_netcdf(l123_dir.joinpath(f"{base_name}_level2.nc"), format="NETCDF4")
    l2_ds.close()

    l3_ds = xr.Dataset(
        {
            "z": ("z", z),
            "T": (("profile", "z"), rng.normal(10, 1, (2, n_z)), {"description": "Temperature L3"}),
        }
    )
    l3_ds.to_netcdf(l123_dir.joinpath(f"{base_name}_level3.nc"), format="NETCDF4")
    l3_ds.close()


def test_main_verbose(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """Covers the True arm of `"debug" if args.verbose else "info"`.

    Also covers nearly every other branch in main(): the plots-dir mkdir (True on
    the L2 file, False on the L3 file since they share a parent dir), dive
    presence/absence, variable presence/absence, the description truthy/falsy
    check, and lon_gps presence/absence.
    """
    _write_l2_l3_files(tmp_path)
    mock_plot_heatmap = Mock()
    monkeypatch.setattr(sg_l123_plot, "plot_heatmap", mock_plot_heatmap)
    monkeypatch.setattr(
        sys,
        "argv",
        ["sg_l123_plot.py", "--L123_dir", str(tmp_path), "--base_name", "test", "--verbose"],
    )

    sg_l123_plot.main()

    # T and S plotted for L2, T only for L3 (S is absent there)
    assert mock_plot_heatmap.call_count == 3
    # position plot only written for L2, which has lon_gps
    assert tmp_path.joinpath("plots", "test_level2.nc_positions.html").exists()
    assert not tmp_path.joinpath("plots", "test_level3.nc_positions.html").exists()


def test_main_default_verbosity(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """Covers the False (else) arm of `"debug" if args.verbose else "info"`."""
    _write_l2_l3_files(tmp_path)
    monkeypatch.setattr(sg_l123_plot, "plot_heatmap", Mock())
    monkeypatch.setattr(sys, "argv", ["sg_l123_plot.py", "--L123_dir", str(tmp_path), "--base_name", "test"])

    sg_l123_plot.main()


# ---------------------------------------------------------------------------
# `if __name__ == "__main__":` guard, including all three except clauses
# ---------------------------------------------------------------------------


def test_dunder_main_success(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """Runs the module as __main__ with no error: covers the try body with no exception."""
    _write_l2_l3_files(tmp_path)
    monkeypatch.setattr(utils, "plot_heatmap", Mock())
    monkeypatch.setattr(sys, "argv", ["sg_l123_plot.py", "--L123_dir", str(tmp_path), "--base_name", "test"])

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(sg_l123_plot.__file__, run_name="__main__")

    assert excinfo.value.code == 0


def test_dunder_main_argparse_system_exit(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """Missing required args make argparse raise SystemExit(2): covers `except SystemExit: pass`.

    The unconditional `sys.exit(0)` after the try/except still fires afterwards,
    so the process ultimately exits 0 despite the argparse failure.
    """
    monkeypatch.setattr(sys, "argv", ["sg_l123_plot.py"])

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(sg_l123_plot.__file__, run_name="__main__")

    assert excinfo.value.code == 0
    capsys.readouterr()


def test_dunder_main_generic_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture
) -> None:
    """Forces main() to raise: covers `except Exception:` (and, via DEBUG_PDB_F, its call site).

    `utils.init_logger` is patched to raise. Since sg_l123_plot imports it with
    `from utils import init_logger`, re-running the module via runpy under
    run_name="__main__" re-binds that name from the (already-patched) cached
    `utils` module, so the freshly executed `main()` calls the raising stub.
    """
    monkeypatch.setattr(sys, "argv", ["sg_l123_plot.py", "--L123_dir", str(tmp_path), "--base_name", "test"])

    def raise_runtime_error(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(utils, "init_logger", raise_runtime_error)

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(sg_l123_plot.__file__, run_name="__main__")

    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "Exception in main" in captured.err
    assert "RuntimeError: boom" in captured.err
