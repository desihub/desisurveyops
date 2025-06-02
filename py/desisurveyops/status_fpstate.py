#!/usr/bin/env python

import matplotlib

matplotlib.use("Agg")
# AR general
import os
from glob import glob
from datetime import datetime
from time import time
import warnings
import multiprocessing

# AR scientifical
import numpy as np
import fitsio

# AR astropy
from astropy.table import Table, vstack
from astropy.time import Time

# AR desisurveyops
from desisurveyops.status_utils import (
    get_filename,
    get_fns,
    get_obsdone_tiles,
    create_mp4,
)

# AR desitarget
from desitarget.geomask import match_to

# AR fiberassgin
from fiberassign.hardware import load_hardware
from fiberassign._internal import (
    FIBER_STATE_OK,
    FIBER_STATE_UNASSIGNED,
    FIBER_STATE_STUCK,
    FIBER_STATE_BROKEN,
    FIBER_STATE_RESTRICT,
)

# AR desiutil
from desiutil.log import get_logger

# AR matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MultipleLocator

log = get_logger()

_fpstate_ref_d = None


def process_fpstate(outdir, survey, specprod, numproc, recompute=False):
    """
    Wrapper function to generate the focal plane state plots
        (per-night files, plots, and overall movie).

    Args:
        outdir: output folder (str)
        survey: survey name (str)
        specprod: spectroscopic production (e.g. daily) (str)
        numproc: number of parallel processes to run (int)
        recompute (optional, defaults to False): if True recompute all maps;
            if False, only compute missing maps (bool)
    """

    log.info(
        "{}\tBEGIN process_fpstate".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    start = time()

    global _fpstate_ref_d

    # AR reference file with x, y for all fibers
    ref_fn = get_filename(outdir, survey, "fpstate", quant="fibxys", ext="ecsv")
    if not os.path.isfile(ref_fn):
        d = get_fpstate_ref()
        d.write(ref_fn)
    _fpstate_ref_d = Table.read(ref_fn)

    # AR settings
    nfiber = 5000
    bits = {
        "OK": FIBER_STATE_OK,
        "UNASSIGNED": FIBER_STATE_UNASSIGNED,
        "STUCK": FIBER_STATE_STUCK,
        "BROKEN": FIBER_STATE_BROKEN,
        "RESTRICT": FIBER_STATE_RESTRICT,
    }
    plotnames = [
        "OK",
        "BROKEN",
        "STUCK-notBROKEN",
        "RESTRICT-notSTUCK-notBROKEN",
        "UNASSIGNED-notRESTRICT-notSTUCK-notBROKEN",
    ]
    plotcolors = [
        "k",
        "b",
        "orange",
        "r",
        "g",
    ]

    # AR get avg x,y per fiber + locs
    if _fpstate_ref_d is None:
        _fpstate_ref_d = get_fpstate_ref()
    fibers, locs = _fpstate_ref_d["FIBER"], _fpstate_ref_d["LOCATION"]
    xs, ys = _fpstate_ref_d["FIBERASSIGN_X"], _fpstate_ref_d["FIBERASSIGN_Y"]

    fns = get_fns(survey=survey, specprod=specprod)

    # AR exposures (no cut on survey, just on the start date)
    e = Table.read(fns["spec"]["exps"])
    sel = e["NIGHT"] >= e["NIGHT"][e["SURVEY"] == survey][0]
    e = e[sel]

    # AR per-night table files
    nights = np.unique(e["NIGHT"])
    fns = np.array(
        [
            get_filename(outdir, survey, "fpstate", night=night, ext="ecsv")
            for night in nights
        ]
    )
    pngs = np.array(
        [
            get_filename(outdir, survey, "fpstate", night=night, ext="png")
            for night in nights
        ]
    )

    ii = []
    for i, (night, fn) in enumerate(zip(nights, fns)):
        if (not os.path.isfile(fn)) | (recompute):
            ii.append(i)
            log.info("(re)process {}".format(fn))

    # AR create ecsv files
    if len(ii) > 0:
        myargs = [(fn, night) for fn, night in zip(fns[ii], nights[ii])]
        pool = multiprocessing.Pool(processes=numproc)
        with pool:
            _ = pool.starmap(create_fpstate_ecsv, myargs)

    # AR read all per-night states into a dictionary
    log.info(
        "{}\tread {} fpstates: start".format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), len(fns)
        )
    )
    myargs = [[fn, locs, bits, plotnames] for fn in fns]
    pool = multiprocessing.Pool(processes=numproc)
    with pool:
        myds = pool.starmap(read_fpstate, myargs)
    log.info(
        "{}\tread {} fpstates: done".format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            len(fns),
        )
    )
    myd = {}
    for d in myds:
        myd.update(d)
    nights = np.array([int(_) for _ in myd])

    # AR create plots
    ii = []
    for i, (night, png) in enumerate(zip(nights, pngs)):
        if (not os.path.isfile(png)) | (recompute):
            ii.append(i)
            log.info("(re)process {}".format(png))
    if len(ii) > 0:
        myargs = [
            (png, night, myd, plotnames, plotcolors, fibers, xs, ys)
            for png, night in zip(pngs[ii], nights[ii])
        ]
        pool = multiprocessing.Pool(processes=numproc)
        with pool:
            _ = pool.starmap(plot_fpstate, myargs)

    # AR copy last image
    outpng = get_filename(outdir, survey, "fpstate", ext="png")
    cmd = "cp {} {}".format(
        pngs[-1],
        outpng,
    )
    log.info(cmd)
    os.system(cmd)

    # AR movie
    outmp4 = get_filename(outdir, survey, "fpstate", ext="mp4")
    log.info("create {}".format(outmp4))
    create_mp4(pngs, outmp4, duration=15)

    log.info(
        "{}\tEND process_fpstate (took {:.1f}s)".format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), time() - start
        )
    )


def get_night_rundate(night):
    """
    Returns the YYYY-MM-DDTT19:00:00+00:00 rundate.

    Args:
        night(int)

    Returns:
        rundate: YYYY-MM-DDTT19:00:00+00:00  (str)
    """
    year = night // 10000
    month = night // 100 % year
    day = night % (night // 100)
    return "{}-{:02d}-{:02d}T19:00:00+00:00".format(year, month, day)


def create_fpstate_ecsv(outfn, night):
    """
    For a given night, creates an .ecsv file recording the STATE of each LOCATION.

    Args:
        outdir: output folder (string)
        night: night (int)

    Notes:
        Takes the focal plane state at YYYY-MM-DDTT19:00:00+00:00.
    """

    # AR load hardware
    rundate = get_night_rundate(night)
    hw = load_hardware(rundate=rundate)

    d = Table()
    hdr = fitsio.FITSHDR()
    hdr["NIGHT"] = night
    hdr["RUNDATE"] = rundate
    d.meta = dict(hdr)
    d["LOCATION"] = hw.locations
    d["FIBER"] = [hw.loc_fiber[loc] for loc in d["LOCATION"]]
    d["STATE"] = [hw.state[loc] for loc in d["LOCATION"]]
    d.write(outfn, overwrite=True)


def get_fpstate_ref(ref_month=202401):
    """
    Get mean x, y, and location for the 5000 fibers.

    Args:
        ref_month (optional, defaults to 202401): we use all fp states from that month
            (YYYMM) (int)

    Returns:
        An astropy.table.Table structure with:
            FIBER, LOCATION, FIBERASSIGN_X, FIBERASSIGN_Y
    """

    # AR all loa 2021 Oct main tiles...
    fns = get_fns(survey="main", specprod="loa")
    t = Table.read(fns["spec"]["tiles"])
    sel = t["SURVEY"] == "main"
    sel &= (t["EFFTIME_SPEC"] > 0) & (t["LASTNIGHT"] // 100 == ref_month)
    t = t[sel]
    log.info("found {} tiles from {}".format(len(t), ref_month))

    # AR read fiberassign files
    fns = [
        os.path.join(
            os.getenv("DESI_TARGET"),
            "fiberassign",
            "tiles",
            "trunk",
            "{}".format("{:06d}".format(tileid)[:3]),
            "fiberassign-{:06d}.fits.gz".format(tileid),
        )
        for tileid in t["TILEID"]
    ]
    d = vstack(
        [
            Table(
                fitsio.read(
                    fn,
                    ext="FIBERASSIGN",
                    columns=["FIBER", "LOCATION", "FIBERASSIGN_X", "FIBERASSIGN_Y"],
                )
            )
            for fn in fns
        ],
        metadata_conflicts="silent",
    )
    ref_d = Table()
    ref_d.meta["REFMONTH"] = ref_month
    ref_d["FIBER"], ii = np.unique(d["FIBER"], return_index=True)
    assert np.all(ref_d["FIBER"] == np.arange(5000, dtype=int))
    ref_d["LOCATION"] = d["LOCATION"][ii]

    # AR x, y
    ref_d["FIBERASSIGN_X"], ref_d["FIBERASSIGN_Y"] = 0.0, 0.0
    for i in range(len(ref_d)):
        sel = d["FIBER"] == ref_d["FIBER"][i]
        ref_d["FIBERASSIGN_X"][i] = np.median(d["FIBERASSIGN_X"][sel])
        ref_d["FIBERASSIGN_Y"][i] = np.median(d["FIBERASSIGN_Y"][sel])

    return ref_d


def read_fpstate(fn, locs, bits, plotnames):
    """
    Read a focal plane file, and stores in dictionary the various infos.

    Args:
        fn: path to the desi-state*ecsv file (str)
        locs: the 5000 locations
        bits: a dictionary with the states to track (dict)
        plotnames: the states to plot (list of str)

    Returns:
        myd: a dictionary with the infos.

    Notes:
        For bits, plotnames, see desisurveyops.status_utils.process_fpstate().
        myd[night] will be a dictionary; handy for stacking results for all nights.
    """

    # AR read fp state file
    # AR match with locations, as there has been
    # AR some swapping with two fibers at some point...u
    d = Table.read(fn)
    night = d.meta["NIGHT"]
    # AR mute the ErfaWarning...
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mjd = Time(
            datetime.strptime(get_night_rundate(night), "%Y-%m-%dT%H:%M:%S%z")
        ).mjd
    ii = match_to(d["LOCATION"], locs)
    assert ii.size == locs.size
    states = d["STATE"][ii]

    # AR start dictionary
    myd = {night: {}}
    myd[night]["MJD"] = mjd
    myd[night]["STATE"] = states
    for plotname in plotnames:
        myd[night][plotname] = np.zeros(states.size, dtype=bool)
        if plotname == "OK":
            sel = states == 0
        else:
            sel = np.zeros(locs.size, dtype=bool)
            for name in plotname.split("-"):
                if name[:3] == "not":
                    name = name[3:]
                    sel &= (states & bits[name]) == 0
                else:
                    sel |= (states & bits[name]) > 0
        myd[night][plotname][sel] = True

    return myd


def plot_fpstate(
    outpng,
    night,
    myd,
    plotnames,
    plotcolors,
    fibers,
    xs,
    ys,
    plot_night_min=20210101,
    plot_night_max=20291231,
):
    """
    Make the survey status plot of the focal plane state for a given night.

    Args:
        outpng: the output file name (str)
        night: the night to consider (int)
        myd: dictionary with all the per-night infos
        plotnames: the states to plot (list of str)
        plotcolors: list of the colors to use for plotnames (str)
        fibers: list of the 5000 fibers (int)
        xs: list of the FIBER_X for the 5000 fibers (float)
        ys_ list of the FIBER_Y for the 5000 fibers (float)
        plot_night_min (optional, defaults to 20210101): lower value for the xlim (int)
        plot_night_max (optional, defaults to 20291231): higher value for the xlim (int)

    Notes:
        For myd, see the output description in read_fpstate().
    """

    # AR nights
    nights = np.array(list(myd.keys()))

    # AR previous states
    if night == nights.min():
        previous_states = -99 + np.zeros(myd[night]["STATE"].size)
    else:
        previous_night = nights[nights < night].max()
        previous_states = myd[previous_night]["STATE"]

    # AR mjd
    mjds = np.array([myd[night]["MJD"] for night in nights])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mjd_min = Time(datetime.strptime(str(plot_night_min), "%Y%m%d")).mjd
        mjd_max = Time(datetime.strptime(str(plot_night_max), "%Y%m%d")).mjd

    # AR plot
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(len(plotnames), 2, wspace=0.2)  # , width_ratios = [1, 3])
    ax = plt.subplot(gs[:, 0])

    # AR petals
    for petal in range(10):
        sel = (fibers >= 500 * petal) & (fibers < 500 * (petal + 1))
        color = np.array(["0.6", "0.8"])[petal % 2]
        ax.scatter(xs[sel], ys[sel], s=20, color=color, alpha=0.5, zorder=0)

    # AR per-status plot
    for plotname, plotcolor in zip(plotnames, plotcolors):
        sel = myd[night][plotname]
        # print(night, plotname, plotcolor, sel.sum())
        if (plotname != "OK") & (sel.sum() > 0):
            ax.scatter(
                xs[sel],
                ys[sel],
                s=20,
                color=plotcolor,
                zorder=1,
                label="{} ({})".format(plotname, sel.sum()),
            )
    diff = myd[night]["STATE"] != previous_states
    if previous_states.max() != -99:
        ax.scatter(
            xs[diff],
            ys[diff],
            s=25,
            facecolors="none",
            edgecolors="k",
            lw=1.0,
            alpha=1.0,
            zorder=2,
            label="New state ({})".format(diff.sum()),
        )

    ax.set_aspect("equal")
    ax.set_title(get_night_rundate(night))
    ax.set_xlabel("FIBERASSIGN_X")
    ax.set_ylabel("FIBERASSIGN_Y")
    ax.grid()
    ax.set_xlim(-425, 425)
    ax.set_ylim(-425, 425)
    ax.legend(loc=3, ncol=2)

    #
    for ix, (plotname, plotcolor) in enumerate(zip(plotnames, plotcolors)):
        axmjd = plt.subplot(gs[ix, 1])
        ys = np.array([myd[night][plotname].sum() for night in nights], dtype=float)
        sel = nights > night
        ys[sel] = np.nan
        axmjd.plot(
            mjds,
            ys - ys[0],
            color=plotcolor,
            label="N({}) - {:.0f}".format(plotname, ys[0]),
        )
        if ix == 0:
            axmjd.set_title(get_night_rundate(night))
        axmjd.set_ylabel("N(FIBER)")
        axmjd.grid()
        axmjd.set_xlim(mjd_min, mjd_max)
        xticks = axmjd.get_xticks()
        axmjd.set_xticks(xticks)
        axmjd.set_xlim(mjd_min, mjd_max)
        if ix == len(plotnames) - 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                xticklabels = [
                    Time(xtick, format="mjd").strftime("%Y%m%d") for xtick in xticks
                ]
        else:
            xticklabels = ["" for xtick in xticks]
        axmjd.set_xticklabels(xticklabels)
        axmjd.set_ylim(-250, 250)
        axmjd.legend(loc=2, ncol=1)

    plt.savefig(outpng, bbox_inches="tight")
    plt.close()
