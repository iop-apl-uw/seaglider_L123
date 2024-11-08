#! /usr/bin/env python
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

"""L2 and L3 utility routines"""

import argparse
import pathlib
import pdb
import sys
import textwrap
import time
import traceback

import cmocean

# from scipy.io import netcdf_file
import numpy as np
import plotly
import xarray as xr

from utils import FullPathAction, PlotConf, init_logger, plot_heatmap

DEBUG_PDB = False  # Set to True to enter debugger on exceptions

plot_vars = {
    "T": cmocean.cm.thermal,
    "S": cmocean.cm.haline,
    "P": cmocean.cm.dense,
    "SA": cmocean.cm.haline,
    "CT": cmocean.cm.thermal,
    "PD": cmocean.cm.dense,
    ##    "ocr504i_chan1",
    ##    "ocr504i_chan2",
    ##    "ocr504i_chan3",
    ##    "ocr504i_chan4",
    "dissolved_oxygen": cmocean.cm.oxy,
    ##    # "oxygen_sat",
    ##    # "FL_baseline",
    ##    # "FL_counts",
    ##    # "FL_npqcorr",
    ##    # "baseline_470nm",
    ##    # "baseline_700nm",
    "wlbb2fl_sig470nm_adjusted": cmocean.cm.matter,
    "wlbb2fl_sig700nm_adjusted": cmocean.cm.matter,
    "wlbb2fl_sig695nm_adjusted": cmocean.cm.algae,
}


def cmocean_to_plotly(cmap, pl_entries):
    """Convert cmocean to plotly colorscale"""
    h = 1.0 / (pl_entries - 1)
    pl_colorscale = []

    for k in range(pl_entries):
        C = list(map(np.uint8, np.array(cmap(k * h)[:3]) * 255))
        pl_colorscale.append([k * h, "rgb" + str((C[0], C[1], C[2]))])

    return pl_colorscale


# @dataclass
# class PlotConf:
#    """Configuration params for plotting"""

#    do_plots: bool  # Geenerate main plots
#    do_plots_detailed: bool  # Generate detailed plots
#    interactive: bool  # display the plot in the browser


plot_conf = PlotConf(True, False, False)

# Set to True if you want to see the missing dives.  Sometimes, the heatmap will screw up in the
# display if this is true
plot_dives = True


def main():
    """Main cmdline entry point"""
    ap = argparse.ArgumentParser(description=__doc__)
    # Add verbosity arguments

    # ap.add_argument("-c", "--conf", required=True,
    #        help="path to the JSON configuration file")
    ap.add_argument("--verbose", default=False, action="store_true", help="enable verbose output")

    ap.add_argument(
        "--L123_dir",
        help="Directory where L1/2/3 are located",
        action=FullPathAction,
        required=True,
    )

    ap.add_argument(
        "--base_name",
        help="Base name for the L2 and L3 netcdf files",
        required=True,
    )

    ap.add_argument(
        "--plot_contour",
        help="Plot contours instead of heatmaps",
        action="store_true",
        default=False,
    )

    ap.add_argument(
        "--plot_webp",
        help="Generate webp output in additon to html",
        action="store_true",
        default=False,
    )

    args = ap.parse_args()

    # load our configuration settings
    # conf = Conf(args.conf)

    logger = init_logger(
        log_dir=args.L123_dir,
        logger_name=pathlib.Path(__file__).name,
        log_level_for_console="debug" if args.verbose else "info",
    )

    l2_ncf_name = f"{args.base_name}_level2.nc"
    l3_ncf_name = f"{args.base_name}_level3.nc"

    logger.info("Starting")

    cmdline = ""
    for i in sys.argv:
        cmdline += f"{i} "

    logger.info(f"Invoked with command line [{cmdline}]")

    t0 = time.time()

    for nc_file, level in (
        (
            args.L123_dir.joinpath(l2_ncf_name),
            "L2",
        ),
        (
            args.L123_dir.joinpath(l3_ncf_name),
            "L3",
        ),
    ):
        plot_dir = nc_file.parent.joinpath("plots")
        if not plot_dir.exists():
            plot_dir.mkdir()
        dsi = xr.open_dataset(nc_file)

        depth = dsi.variables["z"].to_numpy()
        dive = None

        if "dive" in dsi.variables and plot_dives:
            dive = dsi.variables["dive"].to_numpy()
            dive_tag = "Dive"
            x_title = dive_tag
        else:
            dive = None
            dive_tag = "Half Profile"
            x_title = dive_tag

        layout = {
            "xaxis": {
                "title": x_title,
                "showgrid": False,
                #'range': [min_salinity, max_salinity],
            },
            "yaxis": {
                "title": "Depth (m)",
                "autorange": "reversed",
                "range": [np.max(depth), np.min(depth)],
            },
        }

        for var_n, cmap in plot_vars.items():
            if var_n not in dsi.variables:
                continue
            vv = dsi.variables[var_n]
            # Not working - no sure why
            # with warnings.catch_warnings():
            #     # Runtime warnings are generated for the NaNs in various arrays
            #     warnings.simplefilter("ignore")

            #     vv[
            #         np.nonzero(np.logical_or.reduce((vv[:] > 400.0, vv[:] < 0.0)))[0]
            #     ] = np.nan
            descr = vv.attrs["description"]
            var = vv.to_numpy().transpose()
            if descr:
                # descr = descr.decode("utf-8")
                descr = f"{var_n} - {descr}"
            else:
                descr = var_n

            plot_heatmap(
                var,
                "<br>".join(textwrap.wrap(f"{level} {descr}", width=100)),
                plot_conf,
                rot90=False,
                layout=layout,
                x=dive,
                y=depth,
                hovertemplate=f"{dive_tag}: " + "%{x}<br>Depth: %{y}<br>" + f"{var_n}: " + "%{z}<extra></extra>",
                output_name=plot_dir.joinpath(f"{nc_file.name}_{var_n}.html"),
                colorscale=cmocean_to_plotly(cmap, 256),
                f_contour=args.plot_contour,
                f_webp=args.plot_webp,
            )

        # Create position plot

        if "lon_gps" in dsi.variables:
            std_config_dict = {
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                "scrollZoom": True,
            }

            fig = plotly.graph_objects.Figure()
            # lons = dsi.variables["lon_gps"].to_numpy()
            # lats = dsi.variables["lat_gps"].to_numpy()

            gps_meta = []
            profile_meta = []
            dive_meta = []
            for ii in range(len(dive) // 2):
                gps_meta.append(f"Dive {dive[ii * 2]} GPS2")
                gps_meta.append(f"Dive {dive[(ii * 2) + 1]} GPSE")
                profile_meta.append(f"Dive {dive[ii * 2]} Down Profile")
                profile_meta.append(f"Dive {dive[(ii * 2) + 1]} Up Profile")
                dive_meta.append(f"Dive {dive[ii * 2]} Mid Point")

            fig.add_trace(
                {
                    "name": "GPS Positions",
                    "type": "scattergeo",
                    "lon": dsi.variables["lon_gps"].to_numpy(),
                    "lat": dsi.variables["lat_gps"].to_numpy(),
                    "meta": gps_meta,
                    "marker": {"size": 3, "symbol": "cross"},
                    "hovertemplate": "%{lon:.4f} lon<br>%{lat:.4f} lat<br>%{meta}<extra></extra>",
                }
            )
            fig.add_trace(
                {
                    "name": "Profile positions",
                    "type": "scattergeo",
                    "lon": dsi.variables["lon_profile"].to_numpy(),
                    "lat": dsi.variables["lat_profile"].to_numpy(),
                    "meta": profile_meta,
                    "marker": {"size": 3, "symbol": "cross"},
                    "hovertemplate": "%{lon:.4f} lon<br>%{lat:.4f} lat<br>%{meta}<extra></extra>",
                }
            )
            fig.add_trace(
                {
                    "name": "Dive Positions",
                    "type": "scattergeo",
                    "lon": dsi.variables["lon_dive"].to_numpy(),
                    "lat": dsi.variables["lat_dive"].to_numpy(),
                    "meta": dive_meta,
                    "marker": {"size": 3, "symbol": "cross"},
                    "hovertemplate": "%{lon:.4f} lon<br>%{lat:.4f} lat<br>%{meta}<extra></extra>",
                }
            )

            fig.update_geos(
                {
                    "projection_type": "orthographic",
                    # "center" : {"lat" : np.mean(lats), "lon" :np.mean(lons)},
                    "fitbounds": "locations",
                    "lataxis": {"showgrid": True},
                    "lonaxis": {"showgrid": True},
                    "resolution": 50,
                }
            )
            fig.update_layout({"title": "<br>".join(textwrap.wrap(f"{level} positions", width=100))})
            # if plot_opts full_html
            output_name = plot_dir.joinpath(f"{nc_file.name}_positions.html")
            fig.write_html(
                file=output_name,
                include_plotlyjs="cdn",
                full_html=True,
                auto_open=plot_conf.interactive,
                validate=True,
                config=std_config_dict,
                include_mathjax="cdn",
            )

        dsi.close()

    logger.info(f"Finished - Run Time {time.time() - t0}")


if __name__ == "__main__":
    retval = 1

    try:
        main()
    except SystemExit:
        pass
    except Exception:
        if DEBUG_PDB:
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)
        sys.stderr.write(f"Exception in main ({traceback.format_exc()})")

    sys.exit(0)
