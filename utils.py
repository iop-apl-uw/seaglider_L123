# -*- python-fmt -*-
## Copyright (c) 2024  University of Washington.
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

"""General utility routines"""

import argparse
import logging
import os
import pathlib
import time
from typing import Any, Sequence

import numpy as np
import plotly.graph_objects

import utils

# Logging


def init_logger(
    log_level_for_console: str = "info",
    log_level_for_file: str = "debug",
    log_dir: str = None,
    logger_name: str = "default_logger",
    time_stamped_logfile: bool = True,
):
    """Sets up console logging and if requested file logging
    Returns: logger object
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level=logging.DEBUG)
    logger.propagate = False

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
        log_dir = os.path.abspath(os.path.expanduser(log_dir))
        if time_stamped_logfile:
            ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
            ts = f"_{ts}"
        else:
            ts = ""
        fh = logging.FileHandler(os.path.join(log_dir, f"{logger_name}{ts}.log"))
        fh.setLevel(log_level_for_file.upper())
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


#
# Plotting
#


def plot_heatmap(
    data,
    title,
    conf,
    colorscale="Viridis",
    x=None,
    y=None,
    rot90=True,
    annotation=None,
    layout=None,
    hovertemplate=None,
    output_name=None,
    trim_zrange=1.0,
    f_contour=False,
    f_webp=False,
):
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

    if not output_name:
        output_name = f"{utils.ensure_basename(title)}.html"

    fig.write_html(
        file=output_name,
        # include_plotlyjs="cdn",
        include_plotlyjs=True,
        full_html=True,
        auto_open=conf.interactive,
        validate=True,
        config=std_config_dict,
    )

    if f_webp:
        std_width = 1058
        std_height = 894
        std_scale = 1.0

        fig.write_image(
            output_name.replace(".html", ".webp"),
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
    def __init__(self, option_strings: Any, dest: Any, nargs: Any = None, **kwargs: Any) -> None:
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
        if values == "" or values is None:
            setattr(namespace, self.dest, "")
        elif isinstance(values, str):
            # setattr(namespace, self.dest, os.path.abspath(os.path.expanduser(values)))
            setattr(namespace, self.dest, pathlib.Path(values).expanduser().absolute())
        else:
            # setattr(namespace, self.dest, list(map(lambda y: os.path.abspath(os.path.expanduser(y)), values)))
            setattr(namespace, self.dest, list(map(lambda y: pathlib.Path(y).expanduser().absolute(), values)))


# Misc


def ensure_basename(basename):
    """Returns basename with problematic filename characters replaced

    Inputs:
    basename - string

    Returns:
    basename possibly modified

    Raises:
    None
    """
    return basename.replace(" ", "_").replace(",", "_").replace("/", "_").replace("&", "_")


class AttributeDict(dict):
    """Allow dot access for dictionaries"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
