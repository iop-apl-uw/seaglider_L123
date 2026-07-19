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

"""Tests for utils.py targeting 100% statement and branch coverage.

Notes on coverage:
    utils.py has no `except` blocks and no `elif` branches; all conditionals
    are plain `if`/`if-else` statements (FullPathAction.__call__ has an
    `if`/`elif`/`else` chain), each exercised for every arm below.

    plot_heatmap's `f_webp=True` path calls `Figure.write_image(..., engine="kaleido")`;
    `Figure.write_image` is patched out for that test to keep it fast and avoid an
    actual image render, rather than exercised for real.
"""

import argparse
import logging
import pathlib
from unittest.mock import Mock

import numpy as np
import plotly.graph_objects
import pytest

import utils

# ---------------------------------------------------------------------------
# init_logger
# ---------------------------------------------------------------------------


def test_init_logger_no_log_dir() -> None:
    """Covers the False arm of `if log_dir is not None:` (no FileHandler added)."""
    logger = utils.init_logger(logger_name="test_init_logger_no_log_dir", log_dir=None)

    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    assert not any(isinstance(h, logging.FileHandler) for h in logger.handlers)


def test_init_logger_timestamped(tmp_path: pathlib.Path) -> None:
    """Covers the True arm of `if log_dir is not None:` and of `if time_stamped_logfile:`."""
    logger = utils.init_logger(logger_name="test_init_logger_timestamped", log_dir=tmp_path, time_stamped_logfile=True)

    file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 1
    log_files = list(tmp_path.glob("test_init_logger_timestamped_*.log"))
    assert len(log_files) == 1


def test_init_logger_no_timestamp(tmp_path: pathlib.Path) -> None:
    """Covers the False (else) arm of `if time_stamped_logfile:`."""
    logger = utils.init_logger(
        logger_name="test_init_logger_no_timestamp", log_dir=tmp_path, time_stamped_logfile=False
    )

    assert len(logger.handlers) == 2
    assert tmp_path.joinpath("test_init_logger_no_timestamp.log").exists()


# ---------------------------------------------------------------------------
# plot_heatmap
# ---------------------------------------------------------------------------


def test_plot_heatmap_contour_with_annotation_and_layout(tmp_path: pathlib.Path) -> None:
    """Covers the True arms of `if rot90:`, `if f_contour:`, `if annotation:`, `if layout:`.

    Also covers the False arm of `if not output_name:` (an explicit output_name is supplied).
    """
    data = np.random.default_rng(0).normal(0, 1, (4, 5))
    args = argparse.Namespace(interactive=False)
    output_name = str(tmp_path.joinpath("contour.html"))

    utils.plot_heatmap(
        data,
        "contour title",
        args,
        rot90=True,
        annotation="a note",
        layout={"xaxis": {"title": "x"}},
        f_contour=True,
        output_name=output_name,
    )

    assert pathlib.Path(output_name).exists()


def test_plot_heatmap_default_output_name_and_webp(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """Covers the False arms of `if rot90:`, `if f_contour:`, `if annotation:`, `if layout:`.

    Also covers the True arm of `if not output_name:` (the default name is derived from the
    title) and the True arm of `if f_webp:`. `Figure.write_image` is mocked to keep the test
    fast and deterministic.
    """
    monkeypatch.chdir(tmp_path)
    mock_write_image = Mock()
    monkeypatch.setattr(plotly.graph_objects.Figure, "write_image", mock_write_image)

    data = np.random.default_rng(1).normal(0, 1, (4, 5))
    args = argparse.Namespace(interactive=False)

    utils.plot_heatmap(
        data,
        "heatmap title",
        args,
        rot90=False,
        annotation=None,
        layout=None,
        f_contour=False,
        f_webp=True,
    )

    assert tmp_path.joinpath("heatmap_title.html").exists()
    mock_write_image.assert_called_once()
    assert mock_write_image.call_args.args[0] == pathlib.Path("heatmap_title.webp")


# ---------------------------------------------------------------------------
# FullPathAction
# ---------------------------------------------------------------------------


def test_full_path_action_empty_string() -> None:
    """Covers the True arm of `if values == "" or values is None:` (empty-string case)."""
    action = utils.FullPathAction(option_strings=["--foo"], dest="foo")
    namespace = argparse.Namespace()

    action(argparse.ArgumentParser(), namespace, "")

    assert namespace.foo == ""


def test_full_path_action_none() -> None:
    """Covers the True arm of `if values == "" or values is None:` (None case)."""
    action = utils.FullPathAction(option_strings=["--foo"], dest="foo")
    namespace = argparse.Namespace()

    action(argparse.ArgumentParser(), namespace, None)

    assert namespace.foo == ""


def test_full_path_action_single_string() -> None:
    """Covers the `elif isinstance(values, str):` arm."""
    action = utils.FullPathAction(option_strings=["--foo"], dest="foo")
    namespace = argparse.Namespace()

    action(argparse.ArgumentParser(), namespace, "some/rel/path")

    assert namespace.foo == pathlib.Path("some/rel/path").expanduser().resolve()


def test_full_path_action_sequence() -> None:
    """Covers the final `else:` arm (a sequence of path strings, e.g. from nargs="+")."""
    action = utils.FullPathAction(option_strings=["--foo"], dest="foo", nargs="+")
    namespace = argparse.Namespace()

    action(argparse.ArgumentParser(), namespace, ["a", "b"])

    assert namespace.foo == [
        pathlib.Path("a").expanduser().resolve(),
        pathlib.Path("b").expanduser().resolve(),
    ]


# ---------------------------------------------------------------------------
# ensure_basename
# ---------------------------------------------------------------------------


def test_ensure_basename_replaces_all_problem_characters() -> None:
    """Replaces spaces, commas, slashes, and ampersands in a single pass."""
    assert utils.ensure_basename("a b,c/d&e") == "a_b_c_d_e"


# ---------------------------------------------------------------------------
# AttributeDict
# ---------------------------------------------------------------------------


def test_attribute_dict_dot_access() -> None:
    """Reads, writes, and deletes dict entries via attribute-style dot access."""
    d = utils.AttributeDict({"a": 1})

    assert d.a == 1

    d.b = 2
    assert d["b"] == 2

    del d.a
    assert "a" not in d
