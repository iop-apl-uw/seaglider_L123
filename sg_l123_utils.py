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

"""Utilities to support Seaglider L2 and L3 data processing"""

import argparse
import pathlib
import pdb
import sys
import traceback
import warnings

import numpy as np
import scipy
from scipy import signal
from scipy.stats import binned_statistic

from utils import init_logger, plot_heatmap


def interp1(X, V, Xq, assume_sorted=False, extrapolate=False):
    """
    Interpolates to find Vq, the values of the
    underlying function V=F(X) at the query points Xq.
    """
    # fill_value="extrapolate" - not appropriate
    # do like matlab here and fill with nan
    f = scipy.interpolate.interp1d(
        X,
        V,
        bounds_error=False,
        fill_value="extrapolate" if extrapolate else np.nan,
        assume_sorted=assume_sorted,
    )
    return f(Xq)


def running_average_non_uniform(x, y, data, DX, DY, FF, plot_conf=None):
    """Calculates the average and standard deviation over data, using only the
    middle FF percentage of the data set.

    Input:
        x: m x n (or one-d m) array describing the x-axis of the data (typically time)
        y: m x n (or one-d n) array describing the y-axis of the data (typically depth)
        data: m x n array of the data
        dx: length of the running average along the x axis in units of x
        dy: length of the running average along the x axis in units of y
        ff: middle percentage of data to be considered

    Returns:
        x_ref, y_ref: regular x and y axis
        data_avg0, data_std0: mean and standard deviation for the data on the x_ref/y_ref axis
        data_N0: number of points contributing to each point in the regularized data grid
    """
    if len(np.shape(x)) == 1:
        print("Generating mxn x")
        # NYI - this isn't it
        # x = np.tile(x, (np.shape(data)[1],1))
        return None

    if len(np.shape(y)) == 1:
        y = np.tile(y, (np.shape(data)[0], 1))

    # regular x
    dx = DX / 3.0
    x_ref = np.arange(np.floor(np.nanmin(x.flatten() - dx)), np.ceil(np.nanmax(x.flatten() + dx)), dx)

    # regular y
    dy = DY / 3.0
    y_ref = np.arange(np.nanmin(y.flatten() - dy), np.nanmax(y.flatten() + dy), dy)

    data_avg0 = np.empty((len(x_ref), len(y_ref)))
    data_avg0[:] = np.nan
    data_std0 = np.empty((len(x_ref), len(y_ref)))
    data_std0[:] = np.nan
    data_N0 = np.empty((len(x_ref), len(y_ref)))
    data_N0[:] = np.nan

    # For diagnostics
    #    pp = np.round(np.arange(0.1, 0.9, 0.1) * np.shape(data_avg0)[0])

    # Profiling indicated lots of re-calculating of these logical arrays in the original
    # version, so we pre-compute here
    data_inds = np.isfinite(data)

    y_inds = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for jj in range(np.shape(data_avg0)[1]):
            y_inds.append(
                np.logical_and.reduce(
                    (
                        y >= y_ref[jj] - DY / 2.0,
                        y < y_ref[jj] + DY / 2.0,
                    )
                )
            )

    for ii in range(np.shape(data_avg0)[0]):
        # Diagnostics
        # if np.sum(ii == pp) == 1:
        #     print(".", end="")
        # if ii == np.shape(data_avg0)[0]:
        #     print("")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x_inds = np.logical_and.reduce(
                (
                    x >= x_ref[ii] - DX / 2.0,
                    x < x_ref[ii] + DX / 2.0,
                )
            )

        for jj in range(np.shape(data_avg0)[1]):
            inds = np.logical_and.reduce(
                (
                    data_inds,
                    x_inds,
                    y_inds[jj],
                )
            )

            with warnings.catch_warnings():
                # Runtime warnings are generated for the NaNs in various arrays
                warnings.simplefilter("error")

                tmp = np.sort(data[inds])  # array is flattened before the sort
                n_excluded = int(np.round(len(tmp) * (1.0 - FF) / 2.0))  # one side
                # Select middle of range
                tmp_tmp = tmp[np.arange(n_excluded, len(tmp) - n_excluded)]
                if tmp_tmp.size != 0:
                    data_avg0[ii, jj] = np.nanmean(tmp_tmp)
                    data_std0[ii, jj] = np.nanstd(tmp_tmp)
                    data_N0[ii, jj] = len(tmp_tmp)

    if plot_conf:
        plot_heatmap(np.rot90(data_avg0), "data_avg0", conf=plot_conf)

    # temporary x, with all the values filled in
    # x_tmp = x;
    if plot_conf:
        plot_heatmap(np.rot90(x), "x (time)", colorscale="Inferno", conf=plot_conf)

    x_tmp = x.copy()

    # first interpolate the gaps

    for n in range(np.shape(x)[0]):
        i1 = np.nonzero(np.isfinite(x[n, :]))[0]
        if len(i1) > 10:
            x_tmp[n, :] = interp1(i1, x[n, i1], np.arange(np.shape(x)[1]))

    if plot_conf:
        plot_heatmap(np.rot90(x_tmp), "x_tmp", colorscale="Inferno", conf=plot_conf)

    x_tmp2 = np.empty((np.shape(x)[0], len(y_ref)))
    # print("shape x_tmp2", np.shape(x_tmp2))
    x_tmp2[:] = np.nan
    for n in range(np.shape(x_tmp2)[0]):
        i1 = np.nonzero(np.isfinite(x_tmp[n, :] + y[n, :]))[0]
        if len(i1) > 2:
            x_tmp2[n, :] = interp1(y[n, i1], x_tmp[n, i1], y_ref)

    for n in range(np.shape(x_tmp2)[0]):
        i1 = np.nonzero(np.isfinite(x_tmp2[n, :]))[0]
        if len(i1) > 2:
            x_tmp2[n, 0 : i1[0]] = x_tmp2[n, i1[0]]
            x_tmp2[n, i1[-1] : -1] = x_tmp2[n, i1[-1]]

    if plot_conf:
        plot_heatmap(np.rot90(x_tmp2), "x_tmp2", colorscale="Inferno", conf=plot_conf)

    # finally, re-interpolate onto the irregular time grid of itp_grid

    data_avg = np.empty(np.shape(data))
    data_avg[:] = np.nan
    data_std = np.empty(np.shape(data))
    data_std[:] = np.nan

    data_avg1 = np.empty((np.shape(data)[0], len(y_ref)))
    data_avg1[:] = np.nan
    data_std1 = np.empty((np.shape(data)[0], len(y_ref)))
    data_std1[:] = np.nan

    # Walk along depth, and interp avg0 to avg1
    for m in range(np.shape(data_avg1)[1]):
        ii = np.nonzero(np.isfinite(x_tmp2[:, m]))[0]
        if len(ii) > 2:
            data_avg1[ii, m] = interp1(x_ref, data_avg0[:, m], x_tmp2[ii, m])
            data_std1[ii, m] = interp1(x_ref, data_std0[:, m], x_tmp2[ii, m])

    if plot_conf:
        plot_heatmap(np.rot90(data_avg1), "data_avg1", conf=plot_conf)

    for n in range(np.shape(data_avg)[0]):
        jj = np.nonzero(np.isfinite(y[n, :]))[0]
        if len(jj) > 2:
            data_avg[n, jj] = interp1(y_ref, data_avg1[n, :], y[n, jj])
            data_std[n, jj] = interp1(y_ref, data_std1[n, :], y[n, jj])

    array_regular_grid = (x_ref, y_ref, data_avg0, data_std0, data_N0)
    return (data_avg, data_std, array_regular_grid)


def bindata(x, y, bins, sigma=False):
    """
    Bins y(x) onto bins by averaging, when bins define the right hand side of the bin
    NaNs are ignored.  Values less then bin[0] LHS are included in bin[0],
    values greater then bin[-1] RHS are included in bin[-1]

    Input:
        x: values to be binned
        y: data upon which the averaging will be calculated
        bins: right hand side of the bins
        sigma: boolean to indicated if the standard deviation should also be calculated

    Returns:
        b: binned data (averaged)
        n: number of points in each bin
        sigma: standard deviation of the data (if so requested)

    Notes:
        Current implimentation only handles the 1-D case
    """
    idx = np.logical_not(np.isnan(y))
    if not idx.any():
        nan_return = np.empty(bins.size - 1)
        nan_return[:] = np.nan
        if sigma:
            return (nan_return, nan_return.copy(), nan_return.copy())
        else:
            return (nan_return, nan_return.copy(), None)

    # Only consider the non-nan data
    x = x[idx]
    y = y[idx]

    # Note - this treats things to the left of the first bin edge as in "bin[0]",
    # but does not include it in the first bin statistics - that is avgs[0], which is considered
    # bin 1.  Same logic on the right.
    avgs, _, inds = binned_statistic(x, y, statistic="mean", bins=bins)

    bin_count = np.bincount(inds, minlength=bins.size)
    # Bin number zero number len(bins) are not in the stats, so remove them
    bin_count = bin_count[1 : bins.size]
    bin_count = bin_count * 1.0  # Convert to float
    bin_count[bin_count == 0] = np.nan

    if sigma:
        sigma, _, _ = binned_statistic(x, y, statistic="std", bins=bins)
        return (avgs, bin_count, sigma)
    else:
        return (avgs, bin_count, None)


def find_gaps(gap_vector, data):
    """Cheezy solution to locating gaps in data."""
    head = len(gap_vector) // 2

    c = signal.convolve(data, gap_vector)
    tail = len(c) - (head + len(data))
    # print(head, tail, len(c))
    return c[head:-tail]


def main():
    """Main entry point for testing"""
    ap = argparse.ArgumentParser(description=__doc__)
    # Add verbosity arguments

    # ap.add_argument("-c", "--conf", required=True,
    #        help="path to the JSON configuration file")
    ap.add_argument("--verbose", default=False, action="store_true", help="enable verbose output")

    # args = vars(ap.parse_args())
    args = ap.parse_args()

    # load our configuration settings
    # conf = Conf(args.conf)

    logger = init_logger(
        log_dir=".",
        logger_name=pathlib.Path(__file__).name,
        log_level_for_console="debug" if args.verbose else "info",
    )

    logger.info("Starting")

    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = x * 10.0
    bins = np.arange(1.0, 1000.1, 1.0)
    # print(bins[0], bins[-1])
    avgs, bin_count, sigma = bindata(x, y, bins, sigma=True)
    print(avgs, bin_count, sigma)
    # print(np.shape(avgs))

    # print(avgs)
    # print(bin_count)

    logger.info("Finished")
    return 0


if __name__ == "__main__":
    retval = 1

    DEBUG_PDB = False

    try:
        retval = main()
    except Exception:
        if DEBUG_PDB:
            extype, exec_value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)
        sys.stderr.write(f"Exception in main ({traceback.format_exc()})\n")

    sys.exit(retval)
