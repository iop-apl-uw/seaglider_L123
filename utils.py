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

"""General utility routines."""

import argparse
import logging
import pathlib
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import plotly.graph_objects

import utils

# Logging


def init_logger(
    log_level_for_console: str = "info",
    log_level_for_file: str = "debug",
    log_dir: pathlib.Path | None = None,
    logger_name: str = "default_logger",
    time_stamped_logfile: bool = True,
) -> logging.Logger:
    """Sets up console logging and, if requested, file logging.

    Args:
        log_level_for_console: minimum log level to emit to the console
        log_level_for_file: minimum log level to emit to the log file
        log_dir: directory to write the log file to, or None to skip file logging
        logger_name: name of the logger, and base name for the log file
        time_stamped_logfile: True to append a timestamp to the log file name

    Returns:
        Configured logger object.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level=logging.DEBUG)
    # Prevents propagation to parent logger - not clear this was ever correct
    # Prevented pytest from capturing log messages
    # logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s: %(levelname)s: %(filename)s(%(lineno)d): %(message)s",
        "%Y-%m-%d %H:%M:%S %Z",
    )

    ch = logging.StreamHandler()
    ch.setLevel(log_level_for_console.upper())
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_dir is not None:
        # Add time stamp to log name
        log_dir = log_dir.expanduser().resolve()
        if time_stamped_logfile:
            ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
            ts = f"_{ts}"
        else:
            ts = ""
        fh = logging.FileHandler(log_dir.joinpath(f"{logger_name}{ts}.log"))
        fh.setLevel(log_level_for_file.upper())
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


#
# Plotting
#


def plot_heatmap(
    data: npt.NDArray,
    title: str,
    args: argparse.Namespace,
    colorscale: str | list[list[int | float | str]] = "Viridis",
    x: npt.NDArray | None = None,
    y: npt.NDArray | None = None,
    rot90: bool = True,
    annotation: str | None = None,
    layout: dict | None = None,
    hovertemplate: str | None = None,
    output_name: str | pathlib.Path | None = None,
    trim_zrange: float = 1.0,
    f_contour: bool = False,
    f_webp: bool = False,
) -> None:
    """Renders data as a plotly heatmap (or contour plot) and writes it to an html file.

    Args:
        data: 2-d array of values to plot
        title: plot title, also used to derive the output filename when output_name is None
        args: parsed command line arguments; args.interactive controls whether the plot
            is opened in a browser
        colorscale: plotly colorscale name, or an explicit colorscale list
        x: values for the x axis; defaults to array indices when None
        y: values for the y axis; defaults to array indices when None
        rot90: True to rotate data 90 degrees before plotting
        annotation: Optional text annotation to add below the plot
        layout: Optional dict of plotly layout overrides
        hovertemplate: Optional plotly hover template string
        output_name: Output html file path; defaults to a sanitized version of title.
            The webp path (when f_webp is True) is derived by replacing the suffix.
        trim_zrange: fraction of the z data range to display, trimming outliers symmetrically
        f_contour: True to plot a contour plot instead of a heatmap
        f_webp: True to additionally write a webp image of the plot

    Returns:
        None. Writes the plot to output_name (and a matching .webp file if f_webp is True).
    """
    std_config_dict = {
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        "scrollZoom": True,
    }

    if rot90:
        data = np.rot90(data)

    z_max = np.nanmax(data)
    z_min = np.nanmin(data)
    delta = np.abs((z_max - z_min) * (1.0 - trim_zrange)) / 2.0
    z_max -= delta
    z_min += delta

    fig = plotly.graph_objects.Figure()
    if f_contour:
        fig.add_trace(
            {
                "type": "contour",
                "x": x,
                "y": y,
                "z": data,
                "colorscale": colorscale,
                "hovertemplate": hovertemplate,
                "zmax": z_max,
                "zmin": z_min,
                "contours_coloring": "heatmap",
                "connectgaps": False,
                "contours": {
                    "coloring": "heatmap",
                    "showlabels": True,
                    "labelfont": {"family": "Raleway", "size": 12, "color": "white"},
                },
            }
        )
    else:
        fig.add_trace(
            {
                "type": "heatmap",
                "x": x,
                "y": y,
                "z": data,
                "colorscale": colorscale,
                "hovertemplate": hovertemplate,
                "zmax": z_max,
                "zmin": z_min,
                "colorbar": {
                    "title": {
                        "side": "top",
                    },
                },
            }
        )

    fig.update_layout(title=title)

    if annotation:
        l_annotations = [
            {
                "text": annotation,
                "showarrow": False,
                "xref": "paper",
                "yref": "paper",
                "x": 0.0,
                "y": -0.08,
            }
        ]
        fig.update_layout({"annotations": tuple(l_annotations)})

    if layout:
        fig.update_layout(layout)

    output_name = pathlib.Path(output_name) if output_name else pathlib.Path(f"{utils.ensure_basename(title)}.html")

    fig.write_html(
        file=output_name,
        # include_plotlyjs="cdn",
        include_plotlyjs=True,
        full_html=True,
        auto_open=args.interactive,
        validate=True,
        config=std_config_dict,
    )

    if f_webp:
        std_width = 1058
        std_height = 894
        std_scale = 1.0

        fig.write_image(
            output_name.with_suffix(".webp"),
            format="webp",
            width=std_width,
            height=std_height,
            scale=std_scale,
            validate=True,
            engine="kaleido",
        )


# Args

""" Helper routines for argument parsing
"""


class FullPathAction(argparse.Action):
    """Argparse action that expands and resolves path argument(s) to a fully qualified pathlib.Path."""

    def __init__(self, option_strings: Sequence[str], dest: str, nargs: str | int | None = None, **kwargs: Any) -> None:
        """Initializes the action.

        Args:
            option_strings: option strings associated with this action (e.g. ["--foo"])
            dest: attribute name to store the resolved path(s) under
            nargs: number of command-line arguments to consume
            **kwargs: additional keyword arguments passed through to argparse.Action

        Returns:
            None.
        """
        # if nargs is not None:
        #    raise ValueError("nargs not allowed")
        super().__init__(option_strings, dest, **kwargs)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ) -> None:
        """Resolves values to fully qualified pathlib.Path object(s) and stores them on namespace.

        Args:
            parser: the ArgumentParser invoking this action
            namespace: namespace to store the resolved path(s) on
            values: raw argument value(s) - a single path string, a sequence of path strings, or None
            option_string: the option string that was used to invoke this action

        Returns:
            None. Sets the dest attribute on namespace.
        """
        if values == "" or values is None:
            setattr(namespace, self.dest, "")
        elif isinstance(values, str):
            # setattr(namespace, self.dest, os.path.abspath(os.path.expanduser(values)))
            setattr(namespace, self.dest, pathlib.Path(values).expanduser().resolve())
        else:
            # setattr(namespace, self.dest, list(map(lambda y: os.path.abspath(os.path.expanduser(y)), values)))
            setattr(namespace, self.dest, list(map(lambda y: pathlib.Path(y).expanduser().resolve(), values)))


# Misc


def ensure_basename(basename: str) -> str:
    """Returns basename with problematic filename characters replaced.

    Args:
        basename: candidate filename

    Returns:
        basename, with spaces, commas, slashes, and ampersands replaced by underscores.
    """
    return basename.replace(" ", "_").replace(",", "_").replace("/", "_").replace("&", "_")


class AttributeDict(dict[Any, Any]):
    """Allow dot access for dictionaries."""

    __getattr__ = dict.__getitem__
    __setattr__: Any = dict.__setitem__
    __delattr__: Any = dict.__delitem__
