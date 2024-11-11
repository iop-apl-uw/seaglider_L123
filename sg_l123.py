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

"""Performs L2 and L3 processing on seaglider data and outputs L1/L2/L3 data files"""

import argparse
import cProfile
import enum
import itertools
import logging
import math
import pathlib
import pdb
import pstats
import sys
import time
import traceback
import uuid
import warnings
from typing import Any, Literal

import gsw
import isodate
import numpy as np
import xarray as xr
from scipy import signal

from seaglider_utils import collect_dive_ncfiles, dive_number, load_var, open_netcdf_file

# import ExtendedDataClass
from sg_l123_files import load_instrument_metadata, load_mission_meta
from sg_l123_utils import (
    bindata,
    interp1,
    running_average_non_uniform,
)
from utils import AttributeDict, FullPathAction, PlotConf, init_logger, plot_heatmap


class QualityFlags(enum.IntEnum):
    nodata = 0
    outside_despiker = 1
    good = 2


DEBUG_PDB = True

# nc dimension names
zdp_dim = "z_data_point"
hpdp_dim = "half_profile_data_point"
ddp_dim = "dive_data_point"

#
# These are pulled from the per-dive netcdf files first occurance
#
platform_specific_attribs: dict[str, str | int | float] = {}
platform_specific_attribs_list = [
    "platform_id",
    "source",
    "summary",
    "project",
    "glider",
    "mission",
    "seaglider_software_version",
    "base_station_version",
    "base_station_micro_version",
    "quality_control_version",
    # "processing_level",
    # "instrument",
    # "title",
    # "keywords",
]


def fix_ints(data_type: type, attrs: dict[str, Any]) -> dict[str, Any]:
    """Convert int values from LL (yaml format) to appropriate size per gliderdac specs"""
    new_attrs = {}
    for k, v in attrs.items():
        if isinstance(type(v), int):
            new_attrs[k] = data_type(v)
        elif k == "flag_values":
            new_attrs[k] = [data_type(li) for li in v]
        else:
            new_attrs[k] = v
    return new_attrs


def average_position(gps_a_lat: float, gps_a_lon: float, gps_b_lat: float, gps_b_lon: float) -> tuple[float, float]:
    """Given two gps positions in D.D format,
    calculate the mean position between them, based on the great cicle route
    """
    gps_a_lat_rad = math.radians(gps_a_lat)
    gps_a_lon_rad = math.radians(gps_a_lon)
    gps_b_lat_rad = math.radians(gps_b_lat)
    delta_lon_rad = math.radians(gps_b_lon - gps_a_lon)

    b_x = math.cos(gps_b_lat_rad) * math.cos(delta_lon_rad)
    b_y = math.cos(gps_b_lat_rad) * math.sin(delta_lon_rad)

    gps_mean_lat = math.atan2(
        math.sin(gps_a_lat_rad) + math.sin(gps_b_lat_rad),
        math.sqrt((math.cos(gps_a_lat_rad) + b_x) * (math.cos(gps_a_lat_rad) + b_x) + (b_y * b_y)),
    )
    gps_mean_lon = gps_a_lon_rad + math.atan2(b_y, math.cos(gps_a_lat_rad) + b_x)

    return (math.degrees(gps_mean_lat), math.degrees(gps_mean_lon))


class Seaglider_L1_L2_L3(AttributeDict):
    """Struct for holding all variables"""

    pass


def inventory_vars(
    dive_ncfs: list[pathlib.Path], var_dict: dict[str, AttributeDict], logger: logging.Logger
) -> tuple[list[str], list[str], list[str]]:
    """
    Input:
        dive_ncfs - sorted list of dive netcdf file names

    Return:
        profile_vars - list of profile variables to be processed
        l1_profile_vars - list of profile variables to be processed for l1 product
        dive_vars - list of dive variables to be propagated to the final product
    """

    dive_vars = []
    profile_vars = []
    l1_profile_vars = []
    # For now, just consider the first netcdf file that is not marked as haveing a processing error
    # We could do a complete search
    for dive_ncf in dive_ncfs:
        ncf = open_netcdf_file(dive_ncf, logger=logger)
        if ncf is None:
            continue
        if "processing_error" in ncf.variables:
            # logger.warning(
            #     f"{dive_ncf} is marked as having a processing error - using anyway"
            # )
            ncf.close()
            continue
        for vv in var_dict:
            if vv in ncf.variables:
                # if "nc_dimensions" in var_dict[vv]:
                if var_dict[vv].nc_dimensions is not None:
                    if len(var_dict[vv].nc_dimensions) == 1:
                        dive_vars.append(vv)
                    else:
                        if "include_in_L23" not in var_dict[vv] or var_dict[vv]["include_in_L23"]:
                            profile_vars.append(vv)

                if var_dict[vv].nc_L1_dimensions:
                    l1_profile_vars.append(vv)

        if not platform_specific_attribs:
            for attrib in platform_specific_attribs_list:
                # if attrib in ncf._attributes:
                # if isinstance(ncf._attributes[attrib], bytes):
                #    platform_specific_attribs[attrib] = ncf._attributes[attrib].decode()
                # else:
                #    platform_specific_attribs[attrib] = ncf._attributes[attrib]

                if hasattr(ncf, attrib):
                    if isinstance(getattr(ncf, attrib), bytes):
                        platform_specific_attribs[attrib] = getattr(ncf, attrib).decode()
                    else:
                        platform_specific_attribs[attrib] = getattr(ncf, attrib)

        break

    return (dive_vars, profile_vars, l1_profile_vars)


# def load_L1_data(
#     l1_ncf_name,
#     l1_dive_map,
#     l1_time,
#     var_n,
#     L2_L3_vars,
#     sg_L1,
#     num_dives,
#     dives,
#     logger,
#     glider_depth_var,
#     glider_time_var,
#     dive_adjust=0,
# ):
#     """Loads L1 data from a single timeseries file"""
#     l1_ncf = netcdf.netcdf_file(l1_ncf_name, "r", mmap=False)
#     l1_dive_map = l1_ncf.variables[l1_dive_map][:] + dive_adjust
#     l1_ts_time = l1_ncf.variables[l1_time][:]

#     if var_n not in l1_ncf.variables:
#         logger.error(f"{var_n} not in {l1_ncf_name} - skipping")
#         return

#     L2_L3_vars.append(var_n)
#     l1_ts_var = l1_ncf.variables[var_n][:]
#     l1_var_list = [None] * 2 * num_dives
#     l1_depth_list = [None] * 2 * num_dives
#     for ii in range(num_dives):
#         dive_i = dives[ii]
#         glider_depth = np.concatenate(
#             (
#                 getattr(sg_L1, glider_depth_var)[ii * 2],
#                 getattr(sg_L1, glider_depth_var)[ii * 2 + 1],
#             )
#         )
#         glider_time = np.concatenate(
#             (
#                 getattr(sg_L1, glider_time_var)[ii * 2],
#                 getattr(sg_L1, glider_time_var)[ii * 2 + 1],
#             )
#         )
#         dive_idx = np.nonzero(l1_dive_map == dive_i)[0]
#         # var_dive = l1_ts_var[dive_idx]
#         var_time = l1_ts_time[dive_idx]
#         var = l1_ts_var[dive_idx]
#         if len(var) < 2:
#             continue
#         var_depth = interp1_extend(glider_time, glider_depth, var_time)

#         max_depth_i = np.argmax(var_depth)
#         l1_var_list[ii * 2] = var[0:max_depth_i]
#         l1_depth_list[ii * 2] = var_depth[0:max_depth_i]

#         l1_var_list[ii * 2 + 1] = var[max_depth_i:]
#         l1_depth_list[ii * 2 + 1] = var_depth[max_depth_i:]
#     setattr(sg_L1, var_n, l1_var_list)
#     setattr(sg_L1, f"{var_n}_depth", l1_depth_list)


def main(cmdline_args: list[str] = sys.argv[1:]) -> int:
    """
    Main entry point
    """
    ap = argparse.ArgumentParser(description=__doc__)
    # Add verbosity arguments

    # ap.add_argument("-c", "--conf", required=True,
    #        help="path to the JSON configuration file")
    ap.add_argument("--verbose", default=False, action="store_true", help="enable verbose output")
    ap.add_argument(
        "--profile_dir",
        help="Directory where profile netcdfs are located",
        action=FullPathAction,
        required=True,
    )
    ap.add_argument(
        "--L123_dir",
        help="Directory where other L1 data live and L1/L2/3 output goes",
        action=FullPathAction,
        required=True,
    )

    ap.add_argument(
        "--base_name",
        help="Base name for the L1, L2 and L3 netcdf files",
        required=True,
    )
    ap.add_argument(
        "--var_meta",
        default=pathlib.Path(__file__).parent.joinpath("config/var_meta.yml"),
        help="Location for variable metadata configuration file",
        action=FullPathAction,
    )

    ap.add_argument(
        "--mission_meta",
        default=None,
        help="Location for mission metadata configuration file",
        action=FullPathAction,
        required=True,
    )

    ap.add_argument(
        "--instrument_meta",
        default=None,
        help="Location for instrument specific metadata configuration file",
        action=FullPathAction,
    )

    args = ap.parse_args(cmdline_args)

    logger = init_logger(
        log_dir=args.L123_dir,
        logger_name=pathlib.Path(__file__).name,
        log_level_for_console="debug" if args.verbose else "info",
    )

    start_time = time.time()
    logger.info("Starting")

    cmdline = ""
    for i in sys.argv:
        cmdline += f"{i} "

    logger.info(f"Invoked with command line [{cmdline}]")

    try:
        L2_L3_conf, ncf_global_attribs = load_mission_meta(args.mission_meta, logger)
    except Exception:
        logger.exception(f"Unable to load {args.mission_meta} - bailing out")
        return 1
    if L2_L3_conf is None or ncf_global_attribs is None:
        return 1

    try:
        L2_L3_var_meta, additional_variables = load_instrument_metadata(args.var_meta, args.instrument_meta, logger)
    except Exception:
        logger.exception("Unable to load variable metadata- bailing out")
        return 1
    if L2_L3_var_meta is None or additional_variables is None:
        return 1

    if L2_L3_conf.ocr504i_hack:
        logger.warning("OCR504i HACK - Marking downcast as bad.  Remove this code in the future?")

    # netcdf output
    l1_ncf_name = f"{args.base_name}_level1.nc"
    l2_ncf_name = f"{args.base_name}_level2.nc"
    l3_ncf_name = f"{args.base_name}_level3.nc"

    # Debugging - generate plots, show interactively
    # plot_conf = PlotConf(True, False, True)
    plot_conf = PlotConf(False, False, False)

    master_depth = "ctd_depth"
    master_time = "ctd_time"

    # Collect all files - returns sorted
    dive_ncfs = collect_dive_ncfiles(args.profile_dir)

    if not dive_ncfs:
        logger.error(f"No dives found in {args.profile_dir} - bailing")

    max_dive_n = dive_number(dive_ncfs[-1])

    # Check what dives are not present
    dives = list(np.arange(1, max_dive_n + 1))
    dives_present = []
    dive_num = []
    dive_num_L1 = []
    for d in dive_ncfs:
        dd = dive_number(d)
        dives.remove(dd)
        dives_present.append(dd)
        dive_num.append(dd)
        dive_num.append(dd)
        dive_num_L1.append(dd)
    if dives:
        logger.warning(f"Dives(s) {dives} not present")
    del dives

    if L2_L3_conf.remove_missing_dives:
        dives = dives_present
    else:
        dives = list(np.arange(1, max_dive_n + 1))

    num_dives = len(dives)

    # Inventories dives to find variables to process
    dive_vars, L2_L3_vars, L1_vars = inventory_vars(dive_ncfs, L2_L3_var_meta, logger)
    dive_vars += ["time_dive", "lat_dive", "lon_dive"]

    logger.info("Using following global_attributes from per-dive netcdfs")
    for k, v in platform_specific_attribs.items():
        logger.info(f"  {k}:{v}")

    half_profile_vars = [
        "time_gps",
        "lat_gps",
        "lon_gps",
        "time_profile",
        "lat_profile",
        "lon_profile",
    ]

    # Setup L2 and L3 data
    bin_centers = np.arange(0.0, 1000.1, 1.0)
    # This is actually bin edges, so one more point then actual bins
    bin_edges = np.arange(-0.5, 1000.51, 1.0)
    # Do this to ensure everything is caught in the binned statistic
    bin_edges[0] = -20.0
    bin_edges[-1] = 1050.0

    sg_L1 = Seaglider_L1_L2_L3()
    sg_L1.dive_num = None
    sg_L1.dive_num_single_L1 = None
    sg_L1.z = None
    sg_L1.bin_edges = None

    sg_L2 = Seaglider_L1_L2_L3()
    sg_L2.dive_num = np.array(dive_num, dtype=np.int16)
    sg_L2.dive_num_single_L1 = np.array(dive_num_L1, dtype=np.int16)
    sg_L2.z = np.array(bin_centers, dtype=np.int16)
    sg_L2.bin_edges = np.array(bin_edges, dtype=np.int16)

    sg_L3 = Seaglider_L1_L2_L3()
    sg_L3.dive_num = np.array(dive_num, dtype=np.int16)
    sg_L3.dive_num_single_L1 = None
    sg_L3.z = np.array(bin_centers, dtype=np.int16)
    sg_L3.bin_edges = np.array(bin_edges, dtype=np.int16)

    ncf_L2_vars = ["z", "dive_num"]
    ncf_L3_vars = ["z", "dive_num"]

    logger.info("L1 Processing")

    ncf_L1_vars = [f"{y}_gps_{x}" for y in ["time", "lat", "lon"] for x in ("start", "end")]

    for var_n in L1_vars:
        sg_L1[f"{var_n}_L1"] = [None] * num_dives

    for var_n in ncf_L1_vars:
        sg_L2[f"{var_n}_L1"] = [None] * num_dives

    ncf_L1_vars.append("dive_num_single")

    # Init the holding lists
    for var_n in L2_L3_vars:
        sg_L1[var_n] = [None] * 2 * num_dives
        sg_L1[f"{var_n}_depth"] = [None] * 2 * num_dives
        # L1 output

    for var_n in dive_vars:
        tmp = np.empty(num_dives)
        tmp[:] = np.nan
        sg_L2[var_n] = tmp
        sg_L3[var_n] = tmp.copy()

    for var_n in half_profile_vars:
        tmp = np.empty(2 * num_dives)
        tmp[:] = np.nan
        sg_L2[var_n] = tmp
        sg_L3[var_n] = tmp.copy()

    # Harvest the vars and associated depth columns, splitting into dive and climb
    # half profiles and applying QC
    for dive_nc in dive_ncfs:
        ncf = open_netcdf_file(dive_nc, logger=logger)
        dive_i = dives.index(dive_number(dive_nc))
        if ncf is None:
            logger.warning(f"Skipping {dive_nc}")
            continue

        # Alert to any processing issues
        if "processing_error" in ncf.variables:
            logger.warning(f"{dive_nc} is marked as having a processing error - skipping")
            continue

        # Variables with single value per dive
        for var_n in dive_vars:
            if var_n in ncf.variables:
                sg_L2[var_n][dive_i] = ncf.variables[var_n].getValue()
                sg_L3[var_n][dive_i] = ncf.variables[var_n].getValue()
            else:
                sg_L2[var_n][dive_i] = None
                sg_L3[var_n][dive_i] = None

        # Position related variables
        if all(x in ncf.variables for x in ["log_gps_time", "log_gps_lat", "log_gps_lon"]):
            gps_time = ncf.variables["log_gps_time"]
            gps_lat = ncf.variables["log_gps_lat"]
            gps_lon = ncf.variables["log_gps_lon"]

            for ii, tag in ((1, "start"), (2, "end")):
                sg_L2[f"time_gps_{tag}_L1"][dive_i] = gps_time[ii]
                sg_L2[f"lon_gps_{tag}_L1"][dive_i] = gps_lon[ii]
                sg_L2[f"lat_gps_{tag}_L1"][dive_i] = gps_lat[ii]

            for sg_vars in (sg_L2, sg_L3):
                for ii in range(2):
                    sg_vars["time_gps"][dive_i * 2 + ii] = gps_time[ii + 1]
                    sg_vars["lon_gps"][dive_i * 2 + ii] = gps_lon[ii + 1]
                    sg_vars["lat_gps"][dive_i * 2 + ii] = gps_lat[ii + 1]

                dive_lat, dive_lon = average_position(gps_lat[1], gps_lon[1], gps_lat[2], gps_lon[2])
                dive_time = (gps_time[1] + gps_time[2]) / 2.0

                sg_vars["time_dive"][dive_i] = dive_time
                sg_vars["lon_dive"][dive_i] = dive_lon
                sg_vars["lat_dive"][dive_i] = dive_lat

                profile_lat, profile_lon = average_position(gps_lat[1], gps_lon[1], dive_lat, dive_lon)
                profile_time = (gps_time[1] + dive_time) / 2.0

                sg_vars["time_profile"][dive_i * 2] = profile_time
                sg_vars["lon_profile"][dive_i * 2] = profile_lon
                sg_vars["lat_profile"][dive_i * 2] = profile_lat

                profile_lat, profile_lon = average_position(dive_lat, dive_lon, gps_lat[2], gps_lon[2])
                profile_time = (dive_time + gps_time[2]) / 2.0

                sg_vars["time_profile"][dive_i * 2 + 1] = profile_time
                sg_vars["lon_profile"][dive_i * 2 + 1] = profile_lon
                sg_vars["lat_profile"][dive_i * 2 + 1] = profile_lat

        # For each variable, split the profile, bin the two halves
        # and load into the L2 data
        for var_n in set(L1_vars + L2_L3_vars):
            var_met = L2_L3_var_meta[var_n]
            try:
                # TODO - need to update loadvar for the time variables for other instruments
                # to include the time from the truck as that variable, if present
                var, var_depth = load_var(
                    ncf,
                    var_n,
                    var_met.qc_name,
                    var_met.time_name,
                    var_met.truck_time_name,
                    var_met.depth_name,
                    master_time,
                    master_depth,
                    logger=logger,
                )
                if var is None:
                    logger.error(f"Failed to load {var_n} from {dive_nc} - skipping")
                    continue
                if var_depth is None:
                    logger.error(f"Failed to load depth varaible for {var_n} from {dive_nc} - skipping")
                    continue
            except KeyError:
                continue

            if var_n in L1_vars:
                L1_list = sg_L1[f"{var_n}_L1"]
                L1_list[dive_i] = var

            # TODO Range checking goes here

            if var_n in L2_L3_vars:
                max_depth_i = np.argmax(var_depth)
                # l1_var = getattr(sg_L1, var_n)
                # l1_var[ii * 2, :] = var[0:max_depth_i]
                # l1_var[ii * 2 + 1, :] = var[max_depth_i:]

                l1_var_list = sg_L1[var_n]
                l1_depth_list = sg_L1[f"{var_n}_depth"]

                l1_var_list[dive_i * 2] = var[0:max_depth_i]
                if L2_L3_conf.ocr504i_hack and "ocr504i" in var_n:
                    l1_var_list[dive_i * 2] = np.nan
                l1_depth_list[dive_i * 2] = var_depth[0:max_depth_i]
                l1_var_list[dive_i * 2 + 1] = var[max_depth_i:]
                l1_depth_list[dive_i * 2 + 1] = var_depth[max_depth_i:]
        ncf.close()

    # Run importers to get other L1 data
    logger.info("L1 Data importers")

    # glider_depth_var = "ctd_time_depth"
    # glider_time_var = "ctd_time"

    # TODO - break these out into separate files and configure based on conf settings

    # if False:
    #     logger.info("L1 O2 importer")

    #     # O2
    #     l1_ncf_name = os.path.join(
    #         args.L123_dir,
    #         "sg219_EXPORTS_Jul-18_timeseries_L1_aa4831_calibrated_despiked.ncf",
    #     )

    #     for var_n in (
    #         "aanderaa4831_dissolved_oxygen_calibrated",
    #         "aanderaa4831_oxygen_sat_calibrated",
    #     ):
    #         load_L1_data(
    #             l1_ncf_name,
    #             "aa4831_data_point_dive_number",
    #             "aa4831_time",
    #             var_n,
    #             L2_L3_vars,
    #             sg_L1,
    #             num_dives,
    #             dives,
    #             logger,
    #             glider_depth_var,
    #             glider_time_var,
    #         )

    #     logger.info("L1 FL importer")

    #     l1_ncf_name = os.path.join(args.L123_dir, "Seaglider_L1_chl.nc")

    #     for var_n in (
    #         "FL_baseline",
    #         "FL_counts",
    #     ):
    #         load_L1_data(
    #             l1_ncf_name,
    #             "Fl_data_point_dive_number",
    #             "time",
    #             var_n,
    #             L2_L3_vars,
    #             sg_L1,
    #             num_dives,
    #             dives,
    #             logger,
    #             glider_depth_var,
    #             glider_time_var,
    #         )

    #     logger.info("L1 Backscatter")

    #     for var_n, l1_ncf_name in (
    #         ("baseline_470nm", os.path.join(args.L123_dir, "SG219_470nm.v3.nc")),
    #         ("baseline_700nm", os.path.join(args.L123_dir, "SG219_700nm.v3.nc")),
    #     ):
    #         load_L1_data(
    #             l1_ncf_name,
    #             "dive",
    #             "time",
    #             var_n,
    #             L2_L3_vars,
    #             sg_L1,
    #             num_dives,
    #             dives,
    #             logger,
    #             glider_depth_var,
    #             glider_time_var,
    #             dive_adjust=1,
    #         )

    logger.info("L2 Processing")

    # Setup L1 variables
    for var_n in L1_vars:
        l1_name = f"{var_n}_L1"
        l1_list = sg_L1[l1_name]
        # TODO - this is clearly wrong - need init a numpy array, then append the data to it
        l1_var = np.empty(0)
        for dive_data in l1_list:
            l1_var = np.append(l1_var, dive_data)
        # Group the *_L1 variables in sg_L2, so later when adding both L1 and dive variables,
        # this single data class can be used
        sg_L2[l1_name] = l1_var

    # Set up L2 variables
    for var_n in L2_L3_vars:
        tmp = np.empty((num_dives * 2, np.shape(sg_L2.bin_edges)[0] - 1))
        tmp[:] = np.nan
        sg_L2[var_n] = tmp
        sg_L3[var_n] = tmp.copy()

        if f"{var_n}_np" in L2_L3_var_meta:
            sg_L2[f"{var_n}_np"] = tmp.copy()
            ncf_L2_vars.append(f"{var_n}_np")
            sg_L3[f"{var_n}_np"] = tmp.copy()
            ncf_L3_vars.append(f"{var_n}_np")

    # L2 Processing
    for var_n in L2_L3_vars:
        var = sg_L1.get(var_n, None)
        var_depth = sg_L1.get(f"{var_n}_depth", None)

        if var is None:
            logger.error(f"Could not find {var_n} in sg_L1 - skipping")
            continue
        if var_depth is None:
            logger.error(f"Could not find {var_n}_depth in sg_L1 - skipping")
            continue

        l2_var = sg_L2[var_n]
        l2_var_np = sg_L2.get(f"{var_n}_np", None)

        for ii in range(num_dives):
            if (
                var[ii * 2] is not None
                and not np.isnan(var[ii * 2]).all()
                and var_depth[ii * 2] is not None
                and not np.isnan(var_depth[ii * 2]).all()
            ):
                if var_depth[ii * 2 + 1].size != var[ii * 2 + 1].size:
                    logger.error(
                        f"len(depth) {var_depth[ii * 2].size} != len({var_n}) {var[ii * 2].size} "
                        f"for dive {dives[ii]} downcast - likely bad time definition for variable"
                    )
                else:
                    l2_var[ii * 2, :], num_pts, _ = bindata(var_depth[ii * 2], var[ii * 2], sg_L2.bin_edges)
                if l2_var_np is not None:
                    l2_var_np[ii * 2, :] = num_pts
            if (
                var[ii * 2 + 1] is not None
                and not np.isnan(var[ii * 2 + 1]).all()
                and var_depth[ii * 2 + 1] is not None
                and not np.isnan(var_depth[ii * 2 + 1]).all()
            ):
                if var_depth[ii * 2 + 1].size != var[ii * 2 + 1].size:
                    logger.error(
                        f"len(depth) {var_depth[ii * 2 + 1].size} != len({var_n}) {var[ii * 2 + 1].size} "
                        f"for dive {dives[ii]} upcast - likely bad time definition for variable"
                    )
                else:
                    l2_var[ii * 2 + 1, :], num_pts, _ = bindata(var_depth[ii * 2 + 1], var[ii * 2 + 1], sg_L2.bin_edges)
                    if l2_var_np is not None:
                        l2_var_np[ii * 2 + 1, :] = num_pts

    if plot_conf.do_plots:
        for var_n in L2_L3_vars:
            # plot_heatmap(np.rot90(getattr(sg_L1, var_n)), f"L1 {var_n}")
            plot_heatmap(sg_L2[var_n], f"L2 {var_n}", plot_conf)
            num_pts = sg_L2.get(f"{var_n}_np", None)
            if num_pts is not None:
                plot_heatmap(num_pts, f"L2 {var_n}_np", plot_conf)

    logger.info("L3 Processing")

    logger.info(
        f"L3 Calculate ref and rms for {L2_L3_conf.despike_running_mean_dx:0.1f} "
        f"days and {L2_L3_conf.despike_running_mean_dy:0.1f} m"
    )
    for var_n in L2_L3_vars:
        var_met = L2_L3_var_meta[var_n]
        if not var_met.despike:
            continue

        logger.info(f"    {var_n} ref/rms")
        ref, rms_ref, __ = running_average_non_uniform(
            sg_L2[master_time],
            sg_L2.z,
            sg_L2[var_n],
            L2_L3_conf.despike_running_mean_dx * 3600.0 * 24.0,
            L2_L3_conf.despike_running_mean_dy,
            L2_L3_conf.data_range,
        )

        sg_L3[f"{var_n}_ref"] = ref
        sg_L3[f"{var_n}_rms_ref"] = rms_ref
        ncf_L3_vars.append(f"{var_n}_ref")
        ncf_L3_vars.append(f"{var_n}_rms_ref")

        if plot_conf.do_plots_detailed:
            plot_heatmap(ref, f"{var_n} _ref", plot_conf)
            plot_heatmap(rms_ref, f"{var_n} rms ref", plot_conf)

    # Apply the interpolation to the points from the L1 data that are
    # outside the despiker, or propagate values on if the variable isn't being
    # despiked
    logger.info("L3 Filtering out despiked points")

    for var_n in L2_L3_vars:
        var_met = L2_L3_var_meta[var_n]
        if not var_met.despike:
            logger.info(f"    {var_n} not despiking")
            # Just copy the L2 data forward if not despiking
            sg_L3[var_n] = sg_L2[var_n].copy()
        else:
            logger.info(f"    {var_n} despiking")
            # Copy the L2 data forward to new variable
            sg_L3[f"{var_n}_L2"] = sg_L2[var_n].copy()
            ncf_L3_vars.append(f"{var_n}_L2")

            var = sg_L1.get(var_n, None)
            var_depth = sg_L1.get(f"{var_n}_depth", None)

            if var is None:
                logger.error(f"Could not find {var_n} in sg_L1 - skipping")
                continue
            if var_depth is None:
                logger.error(f"Could not find {var_n}_depth in sg_L1 - skipping")
                continue

            L3_var = sg_L3[var_n]

            for ii in range(num_dives):
                # Incorrect - need to check on the down and up cast separately
                # if var is None or var_depth is None:
                #    continue
                if (
                    var[ii * 2] is not None
                    and not np.isnan(var[ii * 2]).all()
                    and var_depth[ii * 2] is not None
                    and not np.isnan(var_depth[ii * 2]).all()
                ):
                    v0_ref_dn = interp1(
                        sg_L3.z,
                        sg_L3[f"{var_n}_ref"][ii * 2, :],
                        var_depth[ii * 2],
                    )
                    v0_rms_ref_dn = interp1(
                        sg_L3.z,
                        sg_L3[f"{var_n}_rms_ref"][ii * 2, :],
                        var_depth[ii * 2],
                    )
                    with warnings.catch_warnings():
                        # Runtime warnings are generated for the NaNs in various arrays
                        warnings.simplefilter("ignore")
                        qcflag_LF_v0_dn = (
                            np.abs(var[ii * 2] - v0_ref_dn) < v0_rms_ref_dn * L2_L3_conf.despike_deviations_for_mean
                        )
                    dn_idx = np.nonzero(qcflag_LF_v0_dn)[0]
                    L3_var[2 * ii, :], _, _ = bindata(
                        # var_depth[dn_idx], var[dn_idx], sg_L3.bin_edges
                        var_depth[ii * 2][dn_idx],
                        var[ii * 2][dn_idx],
                        sg_L3.bin_edges,
                    )

                if (
                    var[ii * 2 + 1] is not None
                    and not np.isnan(var[ii * 2 + 1]).all()
                    and var_depth[ii * 2 + 1] is not None
                    and not np.isnan(var_depth[ii * 2 + 1]).all()
                ):
                    v0_ref_up = interp1(
                        sg_L3.z,
                        sg_L3[f"{var_n}_ref"][ii * 2 + 1, :],
                        var_depth[ii * 2 + 1],
                    )
                    v0_rms_ref_up = interp1(
                        sg_L3.z,
                        sg_L3[f"{var_n}_rms_ref"][ii * 2 + 1, :],
                        var_depth[ii * 2 + 1],
                    )
                    with warnings.catch_warnings():
                        # Runtime warnings are generated for the NaNs in various arrays
                        warnings.simplefilter("ignore")
                        qcflag_LF_v0_up = (
                            np.abs(var[ii * 2 + 1] - v0_ref_up) < v0_rms_ref_up * L2_L3_conf.despike_deviations_for_mean
                        )
                    up_idx = np.nonzero(qcflag_LF_v0_up)[0]
                    L3_var[2 * ii + 1, :], _, _ = bindata(
                        var_depth[ii * 2 + 1][up_idx],
                        var[ii * 2 + 1][up_idx],
                        sg_L3.bin_edges,
                    )

            # setattr(sg_L3, var_n, L3_var)

    if plot_conf.do_plots:
        for var_n in L2_L3_vars:
            var_met = L2_L3_var_meta[var_n]
            if var_met.despike:
                plot_heatmap(sg_L3[var_n], f"L3_despiked_{var_n}", plot_conf)

    logger.info("L3 setting QC flags for depiked points")
    # flags:
    #   0 no data (interpolated)
    #   1 outside the despiker
    #   2 good

    for var_n in L2_L3_vars:
        var_met = L2_L3_var_meta[var_n]
        if var_met.despike:
            L2_var = sg_L2[var_n]
            L3_var = sg_L3[var_n]
            flags = np.zeros(np.shape(L3_var), np.int8)
            # if np.nonzero(np.logical_and(np.isfinite(L2_var), np.isnan(L3_var)))[
            #     0
            # ].size:
            #     pdb.set_trace()
            flags[np.isfinite(L2_var) & np.isnan(L3_var)] = QualityFlags.outside_despiker
            flags[np.isfinite(L2_var) & np.isfinite(L3_var)] = QualityFlags.good
            sg_L3[f"{var_n}_flags"] = flags
            ncf_L3_vars.append(f"{var_n}_flags")
            if plot_conf.do_plots_detailed:
                plot_heatmap(sg_L3[f"{var_n}_flags"], f"L3_{var_n}_flags", plot_conf)
        else:
            # if no despiking - all data considered good
            L3_var = sg_L3[var_n]
            sg_L3[f"{var_n}_flags"] = np.full(np.shape(L3_var), QualityFlags.good)

    # Here we interpolate the time to ensure a dense time grid

    # From the matlab code - I think this is the same as the binned L2 time variable - useusally ctd_time
    # % depth interpolation
    # for it=1:size(sg_data_L3.T,2)
    #   ii = find(isfinite(sg_data_L3.time(:,it)) );
    #   if length(ii)>2
    #     sg_data_L3.time(:,it)=interp1(sg_data_L3.z(ii), sg_data_L3.time(ii,it), sg_data_L3.z);
    #   end
    # end

    logger.info("Interpolating time for L3")
    L3_time = sg_L3[master_time].copy()
    for it in range(np.shape(L3_time)[0]):
        iii = np.nonzero(np.isfinite(L3_time[it, :]))[0]
        if len(iii) > 2:
            L3_time[it, :] = interp1(
                # sg_L3.z[ii], L3_time[it, ii], sg_L3.z, extrapolate=True
                sg_L3.z[iii],
                L3_time[it, iii],
                sg_L3.z,
            )
    sg_L3["time"] = L3_time

    if plot_conf.do_plots:
        plot_heatmap(sg_L3.time, "L3 time", plot_conf)

    if np.std(np.diff(sg_L3.z)) == 0.0:
        # flag_regular_z = True
        dz = np.mean(np.diff(sg_L3.z))
        logger.info(f"dz:{dz}")
    else:
        # Code below does not handle this case (yet)
        # f_regular_z = False
        logger.warning("Irregular z NYI")

    # Before final interpolation, generate a mask that finds larges gaps in the data - either
    # missing or below the accepatble RMS level from the despiker

    logger.info(f"L3 Look for missing data of {L2_L3_conf.max_depth_gap} m")

    gap_array = np.zeros((3, int(np.round(L2_L3_conf.max_depth_gap / dz))))
    gap_array[1, :] = 1.0

    for var_n in L2_L3_vars:
        # var_meta = L2_L3_var_meta[var_n]

        var_flags = sg_L3[f"{var_n}_flags"]
        tmp_flags = np.ones(np.shape(var_flags)) * (var_flags == QualityFlags.good)
        if plot_conf.do_plots:
            plot_heatmap(tmp_flags, f"{var_n} tmp_flags", plot_conf)
        # var_mask = np.empty(np.shape(tmp_flags))

        var_mask = signal.convolve(tmp_flags, gap_array, mode="same", method="direct")
        var_mask[var_mask > 0] = 1.0
        var_mask[var_mask <= 0.0] = np.nan
        if plot_conf.do_plots:
            plot_heatmap(var_mask, f"{var_n} var_mask", plot_conf)

        sg_L3[f"{var_n}_mask"] = var_mask

    # Interpolate to fill in the missing pieces
    logger.info("L3 final interpolation")
    for var_n in L2_L3_vars:
        # var_meta = L2_L3_var_meta[var_n]
        var = sg_L3[var_n]

        if var is None:
            logger.error(f"Could not find {var_n} in sg_L3 - skipping")
            continue

        # var_flags = getattr(sg_L3, f"{var_n}_flags", np.ones(np.shape(var)))
        # var_mask = getattr(sg_L3, f"{var_n}_mask", np.ones(np.shape(var)))
        var_flags = sg_L3[f"{var_n}_flags"]
        var_mask = sg_L3[f"{var_n}_mask"]
        var_interp = np.empty(np.shape(var))
        var_interp[:] = np.nan

        for ii in range(np.shape(sg_L3.time)[0]):
            jj = np.nonzero(np.isfinite(var[ii, :]) & (var_flags[ii, :] == QualityFlags.good))[0]
            if len(jj) > 2:
                var_interp[ii, :] = interp1(sg_L3.z[jj], var[ii, jj], sg_L3.z)

        if plot_conf.do_plots:
            plot_heatmap(var_interp, f"{var_n} L3 interpolation before mask", plot_conf)

        var_interp *= var_mask
        if plot_conf.do_plots:
            plot_heatmap(var_interp, f"{var_n} L3 interpolation", plot_conf)
        sg_L3[f"{var_n}"] = var_interp

    # Derived values
    logger.info("L3 derived variables")
    sg_L3.P = gsw.p_from_z(-sg_L3.z * np.ones(np.shape(sg_L3.temperature)), sg_L3.latitude)
    ncf_L3_vars.append("P")
    sg_L3.SA = gsw.SA_from_SP(sg_L3.salinity, sg_L3.P, sg_L3.longitude, sg_L3.latitude)
    ncf_L3_vars.append("SA")
    sg_L3.CT = gsw.CT_from_t(sg_L3.SA, sg_L3.temperature, sg_L3.P)
    ncf_L3_vars.append("CT")
    sg_L3.PD = gsw.pot_rho_t_exact(
        sg_L3.SA,
        sg_L3.temperature,
        sg_L3.P,
        np.zeros(np.shape(sg_L3.temperature)),
    )
    ncf_L3_vars.append("PD")

    #
    # Write out results
    #

    l1_dso = xr.Dataset()
    l2_dso = xr.Dataset()
    l3_dso = xr.Dataset()

    # return_T = typing.typevar("return_T", np.int8, np.int16, np.float32, np.float64)
    # return_T = TypeVar("return_T", type[np.int8], type[np.int16], type[np.float32], type[np.float64])
    # -> type[np.int8] | type[np.int16] | type[np.float32] | type[np.float64]:

    def type_mapper(
        nc_type: Literal["b", "s", "f", "d"],
    ) -> Any:
        mapping_dict = {
            "b": np.int8,
            "s": np.int16,
            "f": np.float32,
            "d": np.float64,
        }
        if isinstance(nc_type, str) and nc_type in mapping_dict:
            return mapping_dict[nc_type]
        else:
            return np.float64

    def add_variable(
        var_n: str,
        dso: xr.core.dataset.Dataset,
        sg_ll: Seaglider_L1_L2_L3,
        level_value: str,
        var_met_alt: AttributeDict | None = None,
        is_instrument: bool = False,
    ) -> None:
        # if var_n == "latitude":
        #    pdb.set_trace()
        if var_met_alt is None:
            var_met = L2_L3_var_meta[var_n]
        else:
            var_met = var_met_alt
        logger.info(f"Adding {var_n} to {level_value}")
        attribs = {}
        if not is_instrument:
            descr_str = None
            if dso is l1_dso and var_met.nc_attribs["l1"]:
                descr_str = var_met.nc_attribs["l1"]
            if dso is l2_dso and var_met.nc_attribs["l2"]:
                descr_str = var_met.nc_attribs["l2"]
            elif dso is l3_dso and var_met.nc_attribs["l3"]:
                descr_str = var_met.nc_attribs["l3"]
            elif var_met.nc_attribs["description"]:
                descr_str = var_met.nc_attribs["description"]

            if not descr_str:
                logger.warning(f"No valid description found for {var_n}")
            else:
                description = descr_str.format(
                    max_depth_gap=L2_L3_conf.max_depth_gap,
                    despike_deviations_for_mean=L2_L3_conf.despike_deviations_for_mean,
                    despike_running_mean_dx=L2_L3_conf.despike_running_mean_dx,
                    despike_running_mean_dy=L2_L3_conf.despike_running_mean_dy,
                )
                attribs["description"] = description
        for k, v in var_met.nc_attribs.items():
            if k == "FillValue":
                if v is None:
                    attribs["_FillValue"] = False
                else:
                    attribs["_FillValue"] = v
                continue
            if k in ("l1", "l2", "l3", "description"):
                continue
            if v is None:
                continue
            if k == "coverage_content_type":
                attribs[k] = v.name
                continue
            else:
                attribs[k] = v

        # is_str = False
        if isinstance(sg_ll, Seaglider_L1_L2_L3):
            if dso is l1_dso and var_met.nc_L1_dimensions:
                data = sg_ll[f"{var_n}_L1"]
            else:
                data = sg_ll[var_n]
        else:
            data = sg_ll

        if isinstance(data, str):
            inp_data = np.array(data, dtype=np.dtype(("S", len(data))))
            # is_str = True
        elif np.ndim(data) == 0:
            # Scalar data
            inp_data = np.dtype(type_mapper(var_met.nc_type)).type(data)
        elif isinstance(data, list):
            inp_data = np.array(data, dtype=type_mapper(var_met.nc_type))
        else:
            with warnings.catch_warnings():
                # May be nans in data - this will be dealt with below in _FillValue
                warnings.simplefilter("ignore")
                inp_data = data.astype(type_mapper(var_met.nc_type.name))

        if "decimal_pts" in var_met and isinstance(var_met.decimal_pts, int):
            inp_data = inp_data.round(decimals=var_met.decimal_pts)

        # Check for scalar variables
        if np.ndim(inp_data) == 0:
            if inp_data == np.nan and "FillValue" in var_met.nc_attribs and var_met.nc_attribs["FillValue"] is not None:
                inp_data = var_met.nc_attribs["FillValue"]
            if var_met.nc_dimensions == ["instrument"]:
                inp_data = np.array([inp_data])
        elif "FillValue" in var_met.nc_attribs and var_met.nc_attribs["FillValue"] is not None:
            inp_data[np.isnan(inp_data)] = var_met.nc_attribs["FillValue"]

        if dso is l1_dso and var_met.nc_L1_dimensions:
            dims = var_met["nc_L1_dimensions"]
        else:
            dims = var_met.nc_dimensions

        da = xr.DataArray(
            data=inp_data,
            dims=dims,
            # dims=template["variables"][var_name]["dimensions"] if not is_str else None,
            # attrs=attribs,
            attrs=fix_ints(np.int32, attribs),
            # coords=None,
        )
        dso[var_met.nc_varname] = da

    L1_vars = list(itertools.chain.from_iterable([L1_vars, ncf_L1_vars, dive_vars]))
    L2_vars = itertools.chain.from_iterable([L2_L3_vars, ncf_L2_vars, dive_vars, half_profile_vars])
    L3_vars = itertools.chain.from_iterable([L2_L3_vars, ncf_L3_vars, dive_vars, half_profile_vars])

    # Add in all varaibles
    for dso, ll, var_list, level_value in (
        (l1_dso, sg_L2, L1_vars, "L1"),
        (l2_dso, sg_L2, L2_vars, "L2"),
        (l3_dso, sg_L3, L3_vars, "L3"),
    ):
        for var_n in var_list:
            # Filter out variables not destine for L3
            if (
                dso is l3_dso
                and "include_in_L3" in L2_L3_var_meta[var_n]
                and not L2_L3_var_meta[var_n]["include_in_L3"]
            ):
                continue
            add_variable(var_n, dso, ll, level_value)

    # End add variables

    # Global Attributes

    for dso in (l1_dso, l2_dso, l3_dso):
        # Global trumps platform
        for attrib, value in platform_specific_attribs.items():
            dso.attrs[attrib] = value

        # Do this here to allow the processing level to be overridden in the global table
        # Yes - it sounds odd, but needed for PODACC submission for L1 and L2.
        if dso is l1_dso:
            dso.attrs["processing_level"] = "l1"
        elif dso is l2_dso:
            dso.attrs["processing_level"] = "l2"
        else:
            dso.attrs["processing_level"] = "l3"

        for attrib, value in ncf_global_attribs.items():
            dso.attrs[attrib] = value

        for var_n, attrs in additional_variables.items():
            add_variable(var_n, dso, attrs.data, level_value, var_met_alt=attrs, is_instrument=True)

        depth_name = L2_L3_var_meta["z"]["nc_varname"]
        lat_name = L2_L3_var_meta["latitude"]["nc_varname"]
        lon_name = L2_L3_var_meta["longitude"]["nc_varname"]

        dso.attrs["geospatial_lon_min"] = np.format_float_positional(
            np.nanmin(dso[lon_name] if dso is l1_dso else sg_L3.longitude),
            precision=4,
            unique=False,
        )

        dso.attrs["geospatial_lon_max"] = np.format_float_positional(
            np.nanmax(dso[lon_name] if dso is l1_dso else sg_L3.longitude),
            precision=4,
            unique=False,
        )

        dso.attrs["geospatial_lat_min"] = np.format_float_positional(
            np.nanmin(dso[lat_name] if dso is l1_dso else sg_L3.latitude),
            precision=4,
            unique=False,
        )

        dso.attrs["geospatial_lat_max"] = np.format_float_positional(
            np.nanmax(dso[lat_name] if dso is l1_dso else sg_L3.latitude),
            precision=4,
            unique=False,
        )

        dso.attrs["time_coverage_start"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(np.nanmin(dso["time"])),
        )

        dso.attrs["time_coverage_end"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(np.nanmax(dso["time"])),
        )

        # dso.attrs["time_coverage_duration"] = isodate.duration_isoformat(
        #    isodate.Duration(seconds=np.nanmax(sg_L3.time) - np.nanmin(sg_L3.time)),
        #    format=isodate.D_ALT_EXT,
        # )
        dso.attrs["time_coverage_duration"] = isodate.duration_isoformat(
            isodate.Duration(seconds=np.round(np.nanmax(dso["time"]) - np.nanmin(dso["time"])))
        )

        dso.attrs["geospatial_vertical_min"] = np.format_float_positional(
            np.nanmin(dso[depth_name]),
            precision=2,
            unique=False,
        )
        dso.attrs["geospatial_vertical_max"] = np.format_float_positional(
            np.nanmax(dso[depth_name]),
            precision=2,
            unique=False,
        )

        # File related
        t0 = time.time()
        dso.attrs["date_created"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0))

        dso.attrs["date_modified"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0))
        dso.attrs["uuid"] = str(uuid.uuid1())

    # End Global Attributes

    # Write out results
    for dso in (l1_dso, l2_dso, l3_dso):
        if dso is l1_dso:
            netcdf_out_filename = args.L123_dir.joinpath(l1_ncf_name)
        elif dso is l2_dso:
            netcdf_out_filename = args.L123_dir.joinpath(l2_ncf_name)
        else:
            netcdf_out_filename = args.L123_dir.joinpath(l3_ncf_name)

        comp = dict(zlib=True, complevel=9)
        # encoding = {var: comp for var in dso.data_vars}
        encoding = {}
        for nc_var in dso.data_vars:
            encoding[nc_var] = comp.copy()
            # if template["variables"][var]["type"] == "c":
            #    encoding[var]["char_dim_name"] = template["variables"][var][
            #        "dimensions"
            #    ][0]
        dso.to_netcdf(
            netcdf_out_filename,
            "w",
            encoding=encoding,
            # engine="netcdf4",
            format="NETCDF4",
        )

    logger.info(f"Finished - took {time.time() - start_time:.3f} secs")
    return 0


if __name__ == "__main__":
    retval = 1

    try:
        if "--profile" in sys.argv:
            sys.argv.remove("--profile")
            prof_file_name = str(
                pathlib.Path(sys.argv[0]).parent.joinpath(
                    "_"
                    + time.strftime("%H:%M:%S %d %b %Y %Z", time.gmtime(time.time()))
                    .replace(" ", "_")
                    .replace(",", "_")
                    .replace("/", "_")
                    .replace("&", "_")
                    + ".cprof"
                )
            )
            # Generate line timings
            cProfile.run("main()", filename=prof_file_name)
            stats = pstats.Stats(prof_file_name)
            stats.sort_stats("time", "calls")
            stats.print_stats()
        else:
            retval = main()
    except SystemExit:
        pass
    except Exception:
        if DEBUG_PDB:
            extype, exec_value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)
        sys.stderr.write(f"Exception in main ({traceback.format_exc()})\n")

    sys.exit(retval)
