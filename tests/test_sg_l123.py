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

"""Tests for sg_l123.py.

test_downward is the pre-existing full-pipeline integration test, run against the
real testdata/simple mission. Everything below it targets branches that full
pipeline run doesn't reach: the standalone helper functions, error paths in
main()'s config/argument handling, and the `if __name__ == "__main__":` guard.

Notes on coverage -- known gaps not addressed here:
    - The per-dive-variable ".ncdf" (legacy) branch inside main()'s main
      processing loop (~line 660-691) is not covered. remap_ncfd_vars and
      inventory_vars's own .ncdf handling ARE covered directly below, but
      main()'s loop needs a full legacy-format mission dataset to reach.
    - ocr504i-hack's inner `"ocr504i" in var_n` branch isn't reachable with the
      current testdata/simple fixture, which has no ocr504i-named variable; only
      the outer warning (L2_L3_conf.ocr504i_hack) is covered.
    - Several deep add_variable/type_mapper branches (string-typed data,
      per-level l1/l2/l3-specific descriptions, missing long_name warning)
      depend on specific var_meta.yml entries; test_downward's real config
      already exercises a good portion of these but not necessarily all.
"""

import logging
import pathlib
import runpy
import sys
from typing import Any
from unittest.mock import Mock

import netCDF4
import numpy as np
import pytest

import sg_l123
import utils
from utils import AttributeDict

simple_dir = "testdata/simple"
cmd_lines = [
    (
        "--verbose",
        "--profile_dir",
        simple_dir,
        "--L123_dir",
        "testdata/l123",
        "--base_name",
        "NANNOOS_Apr24",
        "--mission_meta",
        "testdata/simple/NANOOS_mission.yml",
    )
]


@pytest.mark.parametrize("cmd_line", cmd_lines)
def test_downward(caplog: pytest.LogCaptureFixture, cmd_line: list[str]) -> None:
    """Runs the full L1/L2/L3 pipeline against the real testdata/simple mission."""
    result = sg_l123.main(cmd_line)
    assert result == 0
    for record in caplog.records:
        if record.levelname == "WARNING" and "Dives(s) [1, 2, 3, 4, 5, 6, 7, 8, 9] not present" in record.msg:
            continue
        assert record.levelname not in ["CRITICAL", "ERROR", "WARNING"]


# ---------------------------------------------------------------------------
# fix_attr_type
# ---------------------------------------------------------------------------


def test_fix_attr_type_int_value() -> None:
    """Covers the `if isinstance(v, int):` True arm."""
    result = sg_l123.fix_attr_type(np.int32, {"scale": 5})

    assert result["scale"] == np.int32(5)


def test_fix_attr_type_flag_values_and_default() -> None:
    """Covers the `elif k == "flag_values":` True arm and the `else:` arm."""
    result = sg_l123.fix_attr_type(np.int32, {"flag_values": [1, 2, 3], "units": "m"})

    assert result["flag_values"] == [np.int32(1), np.int32(2), np.int32(3)]
    assert result["units"] == "m"


# ---------------------------------------------------------------------------
# average_position
# ---------------------------------------------------------------------------


def test_average_position() -> None:
    """Computes the great-circle midpoint between two nearby gps positions."""
    lat, lon = sg_l123.average_position(47.0, -122.0, 47.1, -122.1)

    assert lat == pytest.approx(47.05, abs=0.01)
    assert lon == pytest.approx(-122.05, abs=0.01)


# ---------------------------------------------------------------------------
# DEBUG_PDB_F
# ---------------------------------------------------------------------------


def test_debug_pdb_f_true_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers the True arm of `if DEBUG_PDB:`, with pdb.post_mortem stubbed out."""
    monkeypatch.setattr(sg_l123, "DEBUG_PDB", True)
    mock_post_mortem = Mock()
    monkeypatch.setattr("pdb.post_mortem", mock_post_mortem)

    try:
        raise ValueError("boom")
    except ValueError:
        sg_l123.DEBUG_PDB_F()

    mock_post_mortem.assert_called_once()


def test_debug_pdb_f_false_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers the False arm of `if DEBUG_PDB:`."""
    monkeypatch.setattr(sg_l123, "DEBUG_PDB", False)
    mock_post_mortem = Mock()
    monkeypatch.setattr("pdb.post_mortem", mock_post_mortem)

    try:
        raise ValueError("boom")
    except ValueError:
        sg_l123.DEBUG_PDB_F()

    mock_post_mortem.assert_not_called()


# ---------------------------------------------------------------------------
# remap_ncfd_vars
# ---------------------------------------------------------------------------


class _FakeNcf:
    """Minimal stand-in for a netCDF4.Dataset, exposing only `.variables`."""

    def __init__(self, variables: dict[str, Any], attrs: dict[str, Any] | None = None) -> None:
        self.variables = variables
        self._attrs = attrs or {}

    def __getattr__(self, name: str) -> object:
        if name in self._attrs:
            return self._attrs[name]
        raise AttributeError(name)

    def close(self) -> None:
        pass


def test_remap_ncfd_vars_no_time_var() -> None:
    """Covers `if "time" not in ncf.variables: return None`."""
    ncf = _FakeNcf({})

    assert sg_l123.remap_ncfd_vars(ncf, pathlib.Path("f.ncdf"), logging.getLogger("t")) is None  # ty: ignore[invalid-argument-type]


def test_remap_ncfd_vars_time_zero(caplog: pytest.LogCaptureFixture) -> None:
    """Covers `if ncf.variables["time"][0][0] == 0:` True."""
    ncf = _FakeNcf({"time": np.array([[0.0]])})

    result = sg_l123.remap_ncfd_vars(ncf, pathlib.Path("f.ncdf"), logging.getLogger("t"))  # ty: ignore[invalid-argument-type]

    assert result is None
    assert any(r.levelname == "ERROR" for r in caplog.records)


def test_remap_ncfd_vars_missing_depth_keyerror(caplog: pytest.LogCaptureFixture) -> None:
    """Covers the `except KeyError as exception:` block."""
    ncf = _FakeNcf({"time": np.array([[1.0]])})  # "depth" is missing

    result = sg_l123.remap_ncfd_vars(ncf, pathlib.Path("f.ncdf"), logging.getLogger("t"))  # ty: ignore[invalid-argument-type]

    assert result is None
    assert any(r.levelname == "ERROR" for r in caplog.records)


def test_remap_ncfd_vars_success_with_log_gps() -> None:
    """Covers the try body succeeding, and `if "log_GPS" in ncf.variables:` True."""
    ncf = _FakeNcf(
        {
            "time": np.array([[1.0, 2.0]]),
            "depth": np.array([[10.0, 20.0]]),
            "log_GPS": [0, 47.5, -122.5],
        }
    )

    result = sg_l123.remap_ncfd_vars(ncf, pathlib.Path("f.ncdf"), logging.getLogger("t"))  # ty: ignore[invalid-argument-type]

    assert result is ncf
    assert "ctd_time" in ncf.variables
    assert "ctd_depth" in ncf.variables
    np.testing.assert_array_equal(ncf.variables["latitude"], [[47.5, 47.5]])


def test_remap_ncfd_vars_success_without_log_gps() -> None:
    """Covers the try body succeeding, and `if "log_GPS" in ncf.variables:` False."""
    ncf = _FakeNcf({"time": np.array([[1.0, 2.0]]), "depth": np.array([[10.0, 20.0]])})

    result = sg_l123.remap_ncfd_vars(ncf, pathlib.Path("f.ncdf"), logging.getLogger("t"))  # ty: ignore[invalid-argument-type]

    assert result is ncf

    assert "ctd_time" in ncf.variables
    assert np.isnan(ncf.variables["latitude"]).all()


# ---------------------------------------------------------------------------
# inventory_vars
# ---------------------------------------------------------------------------


def test_inventory_vars_empty_dive_list() -> None:
    """Covers the `for dive_ncf in dive_ncfs:` loop's zero-iteration case."""
    result = sg_l123.inventory_vars([], {}, (), logging.getLogger("t"))

    assert result == ([], [], [], {})


def test_inventory_vars_dimension_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers the dive_vars/profile_vars/l1_profile_vars branches and the missing-var skip.

    Also covers the open_netcdf_file-returns-None skip, and both arms of the
    platform-attribs hasattr/bytes handling (one attrib present-as-bytes, one
    attrib entirely absent).
    """
    var_dict = {
        "var_a": AttributeDict({"nc_dimensions": ["dim1"], "nc_L1_dimensions": None}),
        "var_b": AttributeDict({"nc_dimensions": ["dim1", "dim2"], "nc_L1_dimensions": None, "include_in_L23": True}),
        "var_c": AttributeDict({"nc_dimensions": ["dim1", "dim2"], "nc_L1_dimensions": None, "include_in_L23": False}),
        "var_e": AttributeDict({"nc_dimensions": ["dim1", "dim2"], "nc_L1_dimensions": None}),
        "var_d": AttributeDict({"nc_dimensions": None, "nc_L1_dimensions": ["dim1"]}),
        "var_missing": AttributeDict({"nc_dimensions": ["dim1"], "nc_L1_dimensions": None}),
    }
    fake_vars = {name: np.array([1.0]) for name in ("var_a", "var_b", "var_c", "var_e", "var_d")}
    ncf = _FakeNcf(fake_vars, attrs={"source_bytes": b"hello"})  # "platform_id" is absent

    calls = {"n": 0}

    def fake_open(path: pathlib.Path, logger: logging.Logger | None = None) -> _FakeNcf | None:
        calls["n"] += 1
        return None if calls["n"] == 1 else ncf

    monkeypatch.setattr(sg_l123, "open_netcdf_file", fake_open)

    dive_vars, profile_vars, l1_profile_vars, platform_attribs = sg_l123.inventory_vars(
        [pathlib.Path("p0010001.nc"), pathlib.Path("p0010002.nc")],
        var_dict,
        ("platform_id", "source_bytes"),
        logging.getLogger("t"),
    )

    assert dive_vars == ["var_a"]
    assert profile_vars == ["var_b", "var_e"]
    assert l1_profile_vars == ["var_d"]
    assert platform_attribs == {"source_bytes": "hello"}


def test_inventory_vars_processing_error_skip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers `if "processing_error" in ncf.variables: ncf.close(); continue`."""
    var_dict = {"var_a": AttributeDict({"nc_dimensions": ["dim1"], "nc_L1_dimensions": None})}
    ncf_err = _FakeNcf({"processing_error": np.array([1]), "var_a": np.array([1.0])})
    ncf_good = _FakeNcf({"var_a": np.array([1.0])}, attrs={"platform_id": "SG1"})

    calls = {"n": 0}

    def fake_open(path: pathlib.Path, logger: logging.Logger | None = None) -> _FakeNcf:
        calls["n"] += 1
        return ncf_err if calls["n"] == 1 else ncf_good

    monkeypatch.setattr(sg_l123, "open_netcdf_file", fake_open)

    dive_vars, _profile_vars, _l1_vars, platform_attribs = sg_l123.inventory_vars(
        [pathlib.Path("p0010001.nc"), pathlib.Path("p0010002.nc")],
        var_dict,
        ("platform_id",),
        logging.getLogger("t"),
    )

    assert dive_vars == ["var_a"]
    assert platform_attribs == {"platform_id": "SG1"}


def test_inventory_vars_ncdf_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers `if dive_ncf.suffix == ".ncdf": remap_ncfd_vars(...)`."""
    var_dict = {"var_a": AttributeDict({"nc_dimensions": ["dim1"], "nc_L1_dimensions": None})}
    ncf = _FakeNcf(
        {"time": np.array([[1.0, 2.0]]), "depth": np.array([[10.0, 20.0]]), "var_a": np.array([1.0])},
        attrs={"platform_id": "SG2"},
    )
    monkeypatch.setattr(sg_l123, "open_netcdf_file", lambda path, logger=None: ncf)

    dive_vars, _profile_vars, _l1_vars, platform_attribs = sg_l123.inventory_vars(
        [pathlib.Path("p0010001.ncdf")], var_dict, ("platform_id",), logging.getLogger("t")
    )

    assert dive_vars == ["var_a"]
    assert platform_attribs == {"platform_id": "SG2"}


def test_inventory_vars_ncdf_remap_failure_skips_file(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Covers `ncf = remap_ncfd_vars(...); if ncf is None: logger.warning(...); continue`.

    The first file is a .ncdf missing "time", so remap_ncfd_vars returns None and it's
    skipped (with a warning logged); the second (good) file is then inventoried instead.
    """
    var_dict = {"var_a": AttributeDict({"nc_dimensions": ["dim1"], "nc_L1_dimensions": None})}
    bad_ncf = _FakeNcf({"var_a": np.array([1.0])})  # no "time" -> remap_ncfd_vars returns None
    good_ncf = _FakeNcf(
        {"time": np.array([[1.0, 2.0]]), "depth": np.array([[10.0, 20.0]]), "var_a": np.array([1.0])},
        attrs={"platform_id": "SG3"},
    )

    calls = {"n": 0}

    def fake_open(path: pathlib.Path, logger: logging.Logger | None = None) -> _FakeNcf:
        calls["n"] += 1
        return bad_ncf if calls["n"] == 1 else good_ncf

    monkeypatch.setattr(sg_l123, "open_netcdf_file", fake_open)

    dive_vars, _profile_vars, _l1_vars, platform_attribs = sg_l123.inventory_vars(
        [pathlib.Path("p0010001.ncdf"), pathlib.Path("p0010002.ncdf")],
        var_dict,
        ("platform_id",),
        logging.getLogger("t"),
    )

    assert dive_vars == ["var_a"]
    assert platform_attribs == {"platform_id": "SG3"}
    assert any("Skipping" in r.message and "p0010001.ncdf" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# main -- config/argument error paths
# ---------------------------------------------------------------------------


def test_main_no_dives_found(tmp_path: pathlib.Path) -> None:
    """Covers `if not dive_ncfs: ... return 1` (an empty profile_dir)."""
    empty_profile_dir = tmp_path.joinpath("profiles")
    empty_profile_dir.mkdir()

    result = sg_l123.main(
        [
            "--profile_dir",
            str(empty_profile_dir),
            "--L123_dir",
            str(tmp_path),
            "--base_name",
            "test",
            "--mission_meta",
            f"{simple_dir}/NANOOS_mission.yml",
        ]
    )

    assert result == 1


def test_main_mission_meta_file_not_found(tmp_path: pathlib.Path) -> None:
    """Covers `except Exception: logger.exception(...); return 1` around load_mission_meta."""
    result = sg_l123.main(
        [
            "--profile_dir",
            simple_dir,
            "--L123_dir",
            str(tmp_path),
            "--base_name",
            "test",
            "--mission_meta",
            str(tmp_path.joinpath("does_not_exist.yml")),
        ]
    )

    assert result == 1


def test_main_mission_meta_fails_validation(tmp_path: pathlib.Path) -> None:
    """Covers `if L2_L3_conf is None or ncf_global_attribs is None: return 1`."""
    bad_yml = tmp_path.joinpath("bad_mission.yml")
    bad_yml.write_text("processing_config:\n  despike_running_mean_dx: 'not_a_number'\nglobal_attributes: {}\n")

    result = sg_l123.main(
        [
            "--profile_dir",
            simple_dir,
            "--L123_dir",
            str(tmp_path),
            "--base_name",
            "test",
            "--mission_meta",
            str(bad_yml),
        ]
    )

    assert result == 1


def test_main_load_instrument_metadata_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """Covers `except Exception: logger.exception(...); return 1` around load_instrument_metadata."""

    def raise_runtime_error(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(sg_l123, "load_instrument_metadata", raise_runtime_error)

    result = sg_l123.main(
        [
            "--profile_dir",
            simple_dir,
            "--L123_dir",
            str(tmp_path),
            "--base_name",
            "test",
            "--mission_meta",
            f"{simple_dir}/NANOOS_mission.yml",
        ]
    )

    assert result == 1


def test_main_load_instrument_metadata_returns_none(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """Covers `if L2_L3_var_meta is None or additional_variables is None: return 1`.

    load_instrument_metadata's real implementation never returns None (see module
    docstring), so this is exercised by monkeypatching it directly.
    """
    monkeypatch.setattr(sg_l123, "load_instrument_metadata", lambda *a, **kw: (None, None))

    result = sg_l123.main(
        [
            "--profile_dir",
            simple_dir,
            "--L123_dir",
            str(tmp_path),
            "--base_name",
            "test",
            "--mission_meta",
            f"{simple_dir}/NANOOS_mission.yml",
        ]
    )

    assert result == 1


def test_main_ocr504i_hack_warning(tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture) -> None:
    """Covers `if L2_L3_conf.ocr504i_hack: logger.warning(...)`."""
    mission_text = pathlib.Path(f"{simple_dir}/NANOOS_mission.yml").read_text()
    variant_yml = tmp_path.joinpath("ocr504i_mission.yml")
    variant_yml.write_text(mission_text.replace("ocr504i_hack: false", "ocr504i_hack: true"))

    result = sg_l123.main(
        [
            "--profile_dir",
            simple_dir,
            "--L123_dir",
            str(tmp_path),
            "--base_name",
            "test",
            "--mission_meta",
            str(variant_yml),
        ]
    )

    assert result == 0
    assert any("OCR504i HACK" in r.message for r in caplog.records)


def test_main_remove_missing_dives_false(tmp_path: pathlib.Path) -> None:
    """Covers the False (else) arm of `if L2_L3_conf.remove_missing_dives:`.

    This currently crashes downstream with a dimension-mismatch error when building
    the output xr.Dataset, rather than returning cleanly -- testdata/simple has missing
    dive numbers, and this configuration keeps NaN-filled placeholder profiles for them
    instead of dropping them, which appears to desync array lengths somewhere in the
    L1/L2 assembly. Documented here as current (likely buggy) behavior, not fixed.
    """
    mission_text = pathlib.Path(f"{simple_dir}/NANOOS_mission.yml").read_text()
    variant_yml = tmp_path.joinpath("keep_missing_mission.yml")
    variant_yml.write_text(mission_text.replace("remove_missing_dives: true", "remove_missing_dives: false"))

    with pytest.raises(Exception):  # noqa: B017
        sg_l123.main(
            [
                "--profile_dir",
                simple_dir,
                "--L123_dir",
                str(tmp_path),
                "--base_name",
                "test",
                "--mission_meta",
                str(variant_yml),
            ]
        )


def test_main_dive_open_fails_mid_loop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Covers `if ncf is None: logger.error(...); continue` in main()'s per-dive loop.

    Distinct from collect_dive_ncfiles, which already found the file on disk.
    """
    real_open = sg_l123.open_netcdf_file
    target = pathlib.Path(simple_dir).joinpath("p2490015.nc").resolve()

    def fake_open(path: pathlib.Path, logger: logging.Logger | None = None) -> netCDF4.Dataset | None:
        if pathlib.Path(path).resolve() == target:
            return None
        return real_open(path, logger=logger)

    monkeypatch.setattr(sg_l123, "open_netcdf_file", fake_open)

    result = sg_l123.main(
        [
            "--profile_dir",
            simple_dir,
            "--L123_dir",
            str(tmp_path),
            "--base_name",
            "test",
            "--mission_meta",
            f"{simple_dir}/NANOOS_mission.yml",
        ]
    )

    assert result == 0
    assert any("Unable to open" in r.message for r in caplog.records)


def test_main_ncdf_remap_failure_skips_dive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Covers `ncf = remap_ncfd_vars(ncf, dive_nc, logger); if ncf is None: continue`.

    An extra fake ".ncdf" dive (no real file on disk) with no "time" variable is
    appended to the real dive list; remap_ncfd_vars returns None for it and it's
    skipped (with a warning logged), while the real dives still process normally.
    """

    class _NoTimeNcf:
        def __init__(self) -> None:
            self.variables: dict[str, object] = {}

        def close(self) -> None:
            pass

    real_collect = sg_l123.collect_dive_ncfiles
    real_open = sg_l123.open_netcdf_file
    fake_path = pathlib.Path(simple_dir).joinpath("p2490100.ncdf").resolve()

    def fake_collect(mission_dir: pathlib.Path) -> list[pathlib.Path]:
        return [*real_collect(mission_dir), fake_path]

    def fake_open(path: pathlib.Path, logger: logging.Logger | None = None) -> netCDF4.Dataset | _NoTimeNcf | None:
        if pathlib.Path(path).resolve() == fake_path:
            return _NoTimeNcf()
        return real_open(path, logger=logger)

    monkeypatch.setattr(sg_l123, "collect_dive_ncfiles", fake_collect)
    monkeypatch.setattr(sg_l123, "open_netcdf_file", fake_open)

    result = sg_l123.main(
        [
            "--profile_dir",
            simple_dir,
            "--L123_dir",
            str(tmp_path),
            "--base_name",
            "test",
            "--mission_meta",
            f"{simple_dir}/NANOOS_mission.yml",
        ]
    )

    assert result == 0
    assert any("Skipping" in r.message and "p2490100.ncdf" in r.message for r in caplog.records)


def test_main_no_gps_variables(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers the False arm of `if all(x in ncf.variables for x in [...gps vars...]):`.

    Also strips "latitude"/"longitude" so inventory_vars never inventories them as
    variables to process, covering the False arm of `if "latitude" in sg_L3:` too
    (sg_L3 otherwise always pre-allocates a "latitude" key -- with NaN placeholder
    data -- for any variable inventory_vars found, regardless of whether real data
    ever loads for it).
    """
    stripped_vars = ("log_gps_time", "log_gps_lat", "log_gps_lon", "latitude", "longitude")

    class _NoGpsNcf:
        def __init__(self, real: netCDF4.Dataset) -> None:
            self._real = real
            self.variables = {k: v for k, v in real.variables.items() if k not in stripped_vars}

        def close(self) -> None:
            self._real.close()

        def __getattr__(self, name: str) -> object:
            return getattr(self._real, name)

    real_open = sg_l123.open_netcdf_file

    def fake_open(path: pathlib.Path, logger: logging.Logger | None = None) -> _NoGpsNcf | None:
        real = real_open(path, logger=logger)
        return None if real is None else _NoGpsNcf(real)

    monkeypatch.setattr(sg_l123, "open_netcdf_file", fake_open)

    result = sg_l123.main(
        [
            "--profile_dir",
            simple_dir,
            "--L123_dir",
            str(tmp_path),
            "--base_name",
            "test",
            "--mission_meta",
            f"{simple_dir}/NANOOS_mission.yml",
        ]
    )

    assert result == 0


# ---------------------------------------------------------------------------
# `if __name__ == "__main__":` guard
# ---------------------------------------------------------------------------


def test_dunder_main_success(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """Covers the try body succeeding, falling through to `sys.exit(retval)` with retval=0."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sg_l123.py",
            "--profile_dir",
            str(pathlib.Path(simple_dir).resolve()),
            "--L123_dir",
            str(tmp_path),
            "--base_name",
            "test",
            "--mission_meta",
            str(pathlib.Path(simple_dir).joinpath("NANOOS_mission.yml").resolve()),
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(sg_l123.__file__, run_name="__main__")

    assert excinfo.value.code == 0


def test_dunder_main_argparse_system_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers `except SystemExit: pass`, falling through to `sys.exit(retval)`.

    retval keeps its initial value of 1, since main() never returned.
    """
    monkeypatch.setattr(sys, "argv", ["sg_l123.py"])

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(sg_l123.__file__, run_name="__main__")

    assert excinfo.value.code == 1


def test_dunder_main_generic_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture
) -> None:
    """Covers `except Exception: DEBUG_PDB_F(); sys.stderr.write(...)`."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sg_l123.py",
            "--profile_dir",
            str(pathlib.Path(simple_dir).resolve()),
            "--L123_dir",
            str(tmp_path),
            "--base_name",
            "test",
            "--mission_meta",
            str(pathlib.Path(simple_dir).joinpath("NANOOS_mission.yml").resolve()),
        ],
    )

    def raise_runtime_error(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(utils, "init_logger", raise_runtime_error)

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(sg_l123.__file__, run_name="__main__")

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "Exception in main" in captured.err


def test_dunder_main_profile_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """Covers `if "--profile" in sys.argv:` True, including the cProfile.run/stats path.

    main()'s `cmdline_args` default is bound to `sys.argv[1:]` at `def main(...)` time --
    a *copy* made before `sys.argv.remove("--profile")` runs -- so the profiled `main()`
    call still sees "--profile" as an unrecognized argument and argparse raises SystemExit
    internally. cProfile's own run() helper catches SystemExit before printing stats, so
    this doesn't propagate: retval is never reassigned from its initial value of 1.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sg_l123.py",
            "--profile",
            "--profile_dir",
            str(pathlib.Path(simple_dir).resolve()),
            "--L123_dir",
            str(tmp_path),
            "--base_name",
            "test",
            "--mission_meta",
            str(pathlib.Path(simple_dir).joinpath("NANOOS_mission.yml").resolve()),
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(sg_l123.__file__, run_name="__main__")

    assert excinfo.value.code == 1
