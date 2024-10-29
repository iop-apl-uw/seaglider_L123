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

import os
import traceback

import netCDF4
import numpy as np
import scipy

# File handling


def open_netcdf_file(filename, mode="r", mmap=None, version=1, logger=None):
    """Common presets for opening netcdf files"""

    try:
        # ncf = netcdf.netcdf_file(filename, mode, mmap=False, version=version)
        ds = netCDF4.Dataset(filename, mode)
        ds.set_auto_mask(False)

    except Exception:
        if logger:
            logger.exception(f"Failed to open {filename}")
        return None
    else:
        return ds


def collect_dive_ncfiles(mission_dir):
    """Returns a sorted list of all per-dive netcdf files in the
    mission_dir
    """

    if not mission_dir:
        return None

    dive_files = []
    for m in mission_dir.glob("p???????.nc"):
        dive_files.append(m)

    return sorted(dive_files)


def dive_number(ncf_name):
    """
    Return the dive number, from the file per-dive filename
    """
    _, tail = os.path.split(ncf_name)
    return int(tail[4:8])


nc_qc_character_base = ord("0")
nc_qc_type = "Q"  # the alternative is 'i'

## For QC indications
# QC_NO_CHANGE = 0  # no QC performed
QC_GOOD = 1  # ok
# QC_PROBABLY_GOOD = 2  # ...
# QC_PROBABLY_BAD = 3  # potentially correctable
# QC_BAD = 4  # untrustworthy and irreperable
# QC_CHANGED = 5  # explicit manual change
# QC_UNSAMPLED = 6  # explicitly not sampled (vs. expected but missing)
# QC_INTERPOLATED = 8  # interpolated value
# QC_MISSING = 9  # value missing -- instrument timed out

# only_good_qc_values = [QC_GOOD, QC_PROBABLY_GOOD, QC_CHANGED]


def decode_qc(qc_v):
    """Ensure qc vector is a vector of floats"""
    type_qc = type(qc_v)
    if type_qc in np.ScalarType:
        scalar = True
    else:
        scalar = False
        type_qc = type(qc_v[0].item())  # get equivalent python scalar type
    if nc_qc_type == "Q":
        if type_qc is float or type_qc is int:
            return qc_v
        elif type_qc is str:
            pass  # decode below
    else:  # must be 'i'
        if type_qc is float or type_qc is int:
            return qc_v  # netcdf will coerce
        elif type_qc is str:
            pass  # must be from a previous setting of encode? decode below
    # if we get here, type_qv is str
    if scalar:
        qc_v = float(ord(qc_v[0])) - nc_qc_character_base
    else:  # array
        qc_v = np.array(list(map(ord, qc_v)), np.float64) - nc_qc_character_base
    return qc_v


def find_qc(qc_v, qc_values, mask=False):
    """Find the location (or provide a mask) of entries in qc_v with the given qc values
    Inputs:
    qc_v       - the qc array
    qc_values  - the qc values you are interested in
    mask       - whether you want a mask (True) or a set of indicies (False)

    Returns:
    indices_v  - location (or mask) of those qc values
    """
    # indices = (map if mask else filter )(lambda i: qc_v[i] in qc_values, list(range(len(qc_v))))
    if mask:
        print("Doesn't work - find fix.  Maybe wrap this with a list()")
        indices_v = map(lambda i: qc_v[i] in qc_values, list(range(len(qc_v))))
    else:
        # In Python3 filter is a generator - this will screw up downstream consumers
        # who expect the return of this function to look like an array
        indices_v = list(filter(lambda i: qc_v[i] in qc_values, list(range(len(qc_v)))))
    return indices_v


def load_var(
    ncf,
    var_n,
    var_qc_n,
    var_time_n,
    truck_time_n,
    var_depth_n,
    master_time_n,
    master_depth_n,
    logger=None,
):
    """
    Input:
        ncf - netcdf file object
        var_n - name of the variable
        var_qc_n - name of the matching QC variable
        var_time_n - name of the matching time variable
        var_depth_n - name of the matching depth variable, or None of there is none
        master_depth - name of the master depth variable (usually ctd_depth)
    Returns
        var - netcdf array, with QC applied (QC_GOOD only)
        depth - netcdf array for the matching depth (interpolated if need be)
    """
    var = ncf.variables[var_n][:]
    try:
        if var_qc_n and var_qc_n in ncf.variables:
            var_q = ncf.variables[var_qc_n][:]
            try:
                qc_vals = decode_qc(var_q)
            except Exception:
                if logger:
                    logger.warning(f"Could not decode QC for {var_n} - not applying")
            else:
                # This code doesn't work - find_qc for the mask needs to be fixed and propagated to
                # the basestation.

                # var[
                #    np.logical_not(find_qc(qc_vals, QC.only_good_qc_values, mask=True))
                # ] = nc_nan

                var[qc_vals != QC_GOOD] = np.nan

        if var_depth_n is not None:
            depth = ncf.variables[var_depth_n][:]
        else:
            if var_time_n in ncf.variables:
                tmp_time_n = var_time_n
            elif truck_time_n in ncf.variables:
                tmp_time_n = truck_time_n
            else:
                if logger:
                    logger.error(f"Could not find time variable {var_time_n} or {truck_time_n}")
                return (var, None)
            depth = interp1_extend(
                ncf.variables[master_time_n][:],
                ncf.variables[master_depth_n][:],
                ncf.variables[tmp_time_n][:],
            )
    except Exception:
        if logger:
            logger.error(f"Could not load {var_n} {traceback.format_exc()}")
        return (None, None)
    return (var, depth)


def interp1_extend(t1, data, t2):
    # add 'nearest' data item to the ends of data and t1
    if t2[0] < t1[0]:
        # Copy the first value below the interpolation range
        data = np.append(np.array([data[0]]), data)
        t1 = np.append(np.array([t2[0]]), t1)

    if t2[-1] > t1[-1]:
        # Copy the last value above the interpolation range
        data = np.append(data, np.array([data[-1]]))
        t1 = np.append(t1, np.array([t2[-1]]))

    return scipy.interpolate.interp1d(t1, data)(t2)
