#!/usr/bin/env python

import matplotlib

matplotlib.use("Agg")
# AR general
import os
from glob import glob
import tempfile
from datetime import datetime
from time import time
import multiprocessing

# AR scientifical
import numpy as np
import fitsio

# AR astropy
from astropy.table import Table
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u

# AR desisurveyops
from desisurveyops.status_utils import (
    get_filename,
    get_fns,
    get_mjd,
    get_moon_radecphase,
    create_pdf,
)

# AR desiutil
from desiutil.log import get_logger

# AR matplotlib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MultipleLocator

log = get_logger()


# AR per-night: intentionally no cut on survey, we just plot all observed science exposures
def process_obsconds(outdir, survey, specprod, numproc, recompute=False):
    """
    Wrapper function to generate the observing conditions plots
        (per-night pdf files and the summary png file).

    Args:
        outdir: output folder (str)
        survey: survey name (str)
        specprod: spectroscopic production (e.g. daily) (str)
        numproc: number of parallel processes to run (int)
        recompute (optional, defaults to False): if True recompute all maps;
            if False, only compute missing maps (bool)

    Notes:
        Usually use specprod=daily and specprod_ref=loa.
    """

    log.info(
        "{}\tBEGIN process_obsconds".format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    )
    start = time()

    # AR most of the time, we run 1-2 new nights, but try to be smart to handle the case
    # AR where we run all nights
    # AR a typical night has 20-30 exposures, so we run with numproc=32 per night
    # AR then we run numproc//32 nights in parallel
    # AR if numproc=256, that s eight nights in parallel
    # one_night_numproc = np.min([32, numproc])
    # parallel_nights_numproc = numproc // one_night_numproc
    # log.info("run with {} processes per night, and {} nights in parallel".format(one_night_numproc, parallel_nights_numproc))

    fns = get_fns(specprod=specprod)
    e = Table.read(fns["spec"]["exps"])
    sel = e["EFFTIME_SPEC"] > 0
    sel &= e["NIGHT"] >= 20210514  # AR after start of the main survey...
    e = e[sel]

    myargs = []
    for night in np.unique(e["NIGHT"]):
        fn = get_filename(outdir, None, "obsconds", night=night, ext="pdf")
        if (not os.path.isfile(fn)) | (recompute):
            log.info("generate {}".format(fn))
            myargs.append((fn, specprod, night))
    if len(myargs) > 0:
        log.info(
            "{}\tPlot per-night obsconds for {} nights".format(
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"), len(myargs)
            )
        )
        pool = multiprocessing.Pool(processes=numproc)
        with pool:
            _ = pool.starmap(create_obsconds_pdf, myargs)

    # AR cumulative
    fn = get_filename(outdir, None, "obsconds", case="cumulative", ext="png")
    create_obsconds_cumul(fn, survey, specprod)

    log.info(
        "{}\tEND process_obsconds (took {:.1f}s)".format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), time() - start
        )
    )


def create_obsconds_pdf(outfn, specprod, night):
    """
    For a given night, create a pdf file displaying the observing conditions.

    Args:
        outfn: output pdf file name (str)
        specprod: spectroscopic production (e.g. daily) (str)
        night: night (int)

    Notes:
        The pdf is one page per exposure.
    """

    # AR display settings
    tmpoutdir = tempfile.mkdtemp()
    prognames = np.array(["other", "backup", "bright", "dark"])
    progcols = np.array(["r", "y", "g", "k"])
    #
    keys = [
        "MOON_SEP_DEG",
        "AIRMASS",
        "TRANSPARENCY_GFA",
        "SEEING_GFA",
        "SKY_MAG_R_SPEC",
        "EFFTIME_SPEC / EXPTIME",
        "EFFTIME_SPEC * EBVFAC * AIRFAC / EXPTIME",
    ]
    mlocs = [25, 0.20, 0.20, 0.50, 1.0, 0.5, 0.5]
    ylims = [
        (0, 180),
        (0.9, 2),
        (0, 1.1),
        (0, 3),
        (17, 22),
        (0, 2.5),
        (0, 2.5),
    ]
    clim = (0, 0.03)  # AR for delta_xy

    # AR exposures
    fns = get_fns(specprod=specprod)
    e = Table.read(fns["spec"]["exps"])
    sel = (e["NIGHT"] == night) & (e["EFFTIME_SPEC"] > 0)
    e = e[sel]

    # AR grab mjd from raw data..
    e["MJD"] = [get_mjd(expid, night) for expid in e["EXPID"]]

    # AR add moon separation
    moonras, moondecs, e["MOON_PHASE"] = get_moon_radecphase(e["MJD"])
    cs = SkyCoord(ra=e["TILERA"] * u.deg, dec=e["TILEDEC"] * u.deg, frame="icrs")
    mooncs = SkyCoord(ra=moonras * u.deg, dec=moondecs * u.deg, frame="icrs")
    e["MOON_SEP_DEG"] = cs.separation(mooncs).to(u.deg).value

    # AR start building dictionary
    mydict = {}
    # AR other keys
    for key in ["EXPID", "MJD"] + keys:
        if key == "EFFTIME_SPEC / EXPTIME":
            mydict[key] = e["EFFTIME_SPEC"] / e["EXPTIME"]
        elif key == "EFFTIME_SPEC * EBVFAC * AIRFAC / EXPTIME":
            ebvfac = 10.0 ** (2 * 2.165 * e["EBV"] / 2.5)
            airfac = e["AIRMASS"] ** 1.75
            mydict[key] = e["EFFTIME_SPEC"] * ebvfac * airfac / e["EXPTIME"]
        elif key == "SKY_MAG_R_SPEC":
            mydict[key] = e["SKY_MAG_R_SPEC"]
        else:
            mydict[key] = e[key]

    # AR color
    mydict["COLOR"] = np.array(
        [progcols[prognames == "other"][0] for expid in mydict["EXPID"]]
    )
    for progname in ["backup", "bright", "dark"]:
        sel = np.array([_.lower().replace("1b", "") == progname for _ in e["PROGRAM"]])
        mydict["COLOR"][sel] = progcols[prognames == progname]

    # AR KPNO always is UTC-7, no day saving light
    date = "{}T17:00:00-07:00".format(night)
    mjdlo = Time(datetime.strptime(date, "%Y%m%dT%H:%M:%S%z")).mjd
    nextnight = int(
        Time(
            Time(datetime.strptime("{}".format(night), "%Y%m%d")).mjd + 1, format="mjd"
        ).strftime("%Y%m%d")
    )
    date = "{}T08:00:00-07:00".format(nextnight)
    mjdhi = Time(datetime.strptime(date, "%Y%m%dT%H:%M:%S%z")).mjd
    mjdlim = (mjdlo, mjdhi)
    # mjdlim = (mydict["MJD"].min() - 0.01, mydict["MJD"].max() + 0.01)
    mjdticks, mjdticklabels = [], []
    for n in [night, nextnight]:
        for h in range(24):
            date = "{}T{:02d}:00:00-07:00".format(n, h)
            mjd = Time(datetime.strptime(date, "%Y%m%dT%H:%M:%S%z")).mjd
            if (mjd > mjdlim[0]) & (mjd < mjdlim[1]):
                mjdticklabels.append("{}h".format(h))
                mjdticks.append(mjd)

    # AR positioners accuracy from exposure-qa-EXPID.fits
    nightdir = os.path.join(
        os.getenv("DESI_ROOT"),
        "spectro",
        "redux",
        specprod,
        "exposures",
        "{}".format(night),
    )
    outpngs = []
    for i, expid in enumerate(e["EXPID"]):
        fn = os.path.join(
            nightdir, "{:08d}".format(expid), "exposure-qa-{:08d}.fits".format(expid)
        )
        if not os.path.isfile(fn):
            continue

        outpng = os.path.join(tmpoutdir, "obsconds-{}-{:08d}.png".format(night, expid))
        outpngs.append(outpng)
        fig = plt.figure(figsize=(25, 10))
        gs = gridspec.GridSpec(
            len(keys), 2, hspace=0.2, wspace=0.15, width_ratios=[0.3, 1.0]
        )
        ax = plt.subplot(gs[:, 0])
        ax.text(
            0.5,
            1.5,
            "NIGHT = {}".format(night),
            color="k",
            ha="center",
            fontsize=20,
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            0.0,
            1.3,
            "Moon illumination: {:.2f}".format(e["MOON_PHASE"][i]),
            ha="left",
            fontsize=12,
            fontweight="normal",
            transform=ax.transAxes,
        )
        ax.text(
            0.0,
            1.2,
            "EBV: {:.2f}".format(e["EBV"][i]),
            ha="left",
            fontsize=12,
            fontweight="normal",
            transform=ax.transAxes,
        )
        # AR positioning accuracy
        d = fitsio.read(
            fn,
            ext="FIBERQA",
            columns=["FIBER_X", "FIBER_Y", "DELTA_X", "DELTA_Y"],
        )
        sc = ax.scatter(
            d["FIBER_X"],
            d["FIBER_Y"],
            c=np.sqrt(d["DELTA_X"] ** 2 + d["DELTA_Y"] ** 2),
            s=10,
            cmap=matplotlib.cm.coolwarm,
            vmin=clim[0],
            vmax=clim[1],
            rasterized=True,
        )
        cbar = plt.colorbar(sc, fraction=0.046, pad=0.04)
        cbar.set_label("sqrt( DELTA_X ** 2 + DELTA_Y ** 2) [mm]")
        cbar.mappable.set_clim(clim)
        ax.grid(True)
        ax.set_title(
            "EXPID={:08d} , TILEID={:06d}, EXPTIME={:.0f}s".format(
                expid, e["TILEID"][i], e["EXPTIME"][i]
            )
        )
        ax.set_xlabel("FIBER_X [mm]")
        ax.set_ylabel("FIBER_Y [mm]")
        ax.set_xlim(-450, 450)
        ax.set_ylim(-450, 450)
        ax.set_aspect("equal")
        ax.grid(True)
        # AR time-evolution of various quantities
        for ix, (key, ylim, mloc) in enumerate(zip(keys, ylims, mlocs)):
            ax = plt.subplot(gs[ix, 1])
            ax.plot(mydict["MJD"], mydict[key], color="b", lw=0.5)
            ax.scatter(
                mydict["MJD"],
                mydict[key],
                facecolors="none",
                edgecolors=mydict["COLOR"],
                s=100,
            )
            ax.scatter(mydict["MJD"][i], mydict[key][i], c=mydict["COLOR"][i], s=100)
            #
            xs = [
                e["MJD"][i],  # - e["EXPTIME"][i] / 2 / 3600.0 / 24.0,
                e["MJD"][i],  # + e["EXPTIME"][i] / 2 / 3600.0 / 24.0,
            ]
            ax.fill_between(
                xs,
                [ylim[0], ylim[0]],
                [ylim[1], ylim[1]],
                color=mydict["COLOR"][i],
                alpha=0.1,
                zorder=0,
            )

            # AR nominal / reference values
            yvals = None
            if key in ["AIRMASS", "SEEING_GFA"]:
                yvals = [1.1]
            if key == "SKY_MAG_R_SPEC":
                yvals = [21.07]
            if key in [
                "EFFTIME_SPEC / EXPTIME",
                "EFFTIME_SPEC * EBVFAC * AIRFAC / EXPTIME",
            ]:
                yvals = [0.4, 1.0]
            if yvals is not None:
                for yval in yvals:
                    ax.axhline(yval, color="y", ls="--", label="{}".format(yval))

            # AR program color legend
            for progname, progcol in zip(prognames, progcols):
                if (mydict["COLOR"] == progcol).sum() > 0:
                    ax.scatter(
                        np.nan,
                        np.nan,
                        facecolors="none",
                        edgecolors=progcol,
                        s=100,
                        label=progname,
                    )

            ax.legend(loc=1)
            ax.set_xticks(mjdticks)
            if key == keys[-1]:
                ax.set_xticklabels(mjdticklabels)
                ax.set_xlabel("KPNO Time")
            else:
                ax.set_xticklabels([])
            ax.set_xlim(mjdlim)
            ax.set_ylim(ylim)
            ax.yaxis.set_major_locator(MultipleLocator(mloc))
            ax.grid(True)
            ax.text(
                0.01,
                0.80,
                key,
                color="k",
                fontweight="bold",
                ha="left",
                transform=ax.transAxes,
            )

        plt.savefig(outpng, bbox_inches="tight")
        plt.close()

    # AR convert to pdf
    if len(outpngs) == 0:
        log.warning(
            "no processed exposures for {}; no {} created".format(
                night, os.path.basename(outfn)
            )
        )
    # else:
    create_pdf(outpngs, outfn, dpi=300)

    # AR clean
    for outpng in outpngs:
        os.remove(outpng)


def create_obsconds_cumul(outpng, survey, specprod):
    """
    Make the png summary file of the observing conditions for the survey.

    Args:
        outpng: output file name (str)
        survey: survey name (str)
        specprod: spectroscopic production (e.g. daily) (str)
    """
    fns = get_fns(survey=survey, specprod=specprod)

    # AR display settings
    prognames = np.array(["backup", "bright", "bright1b", "dark", "dark1b"])
    progcols = np.array(["y", "g", "m", "k", "0.5"])
    #
    keys = [
        "AIRMASS",
        "TRANSPARENCY_GFA",
        "SEEING_GFA",
        "FIBER_FRACFLUX_GFA",
        "SKY_MAG_R_SPEC",
        "EFFTIME_SPEC * EBVFAC * AIRFAC / EXPTIME",
    ]
    mlocs = [0.10, 0.10, 0.2, 0.1, 1.0, 0.5]
    xlims = [
        (1.0, 1.5),
        (0.5, 1.1),
        (0.5, 2.0),
        (0, 1),
        (17, 22),
        (0, 2.5),
    ]

    # AR exposures
    fns = get_fns(specprod=specprod)
    e = Table.read(fns["spec"]["exps"])
    sel = (e["SURVEY"] == survey) & (e["EFFTIME_SPEC"] > 0)
    e = e[sel]
    nexp = len(e)

    # AR grab mjd from raw data for MJD=0
    ii = np.where(e["MJD"] == 0)[0]
    for i, (expid, night) in enumerate(zip(e["EXPID"][ii], e["NIGHT"][ii])):
        fn = os.path.join(
            os.getenv("DESI_ROOT"),
            "spectro",
            "data",
            str(night),
            "{:08d}".format(expid),
            "desi-{:08d}.fits.fz".format(expid),
        )
        log.info("get MJD from {}".format(fn))
        e["MJD"][i] = fitsio.read_header(fn, "SPEC")["MJD-OBS"]

    # AR add moon separation
    moonras, moondecs, e["MOON_PHASE"] = get_moon_radecphase(e["MJD"])
    cs = SkyCoord(ra=e["TILERA"] * u.deg, dec=e["TILEDEC"] * u.deg, frame="icrs")
    mooncs = SkyCoord(ra=moonras * u.deg, dec=moondecs * u.deg, frame="icrs")
    e["MOON_SEP_DEG"] = cs.separation(mooncs).to(u.deg).value

    # AR start building dictionary
    mydict = {}
    # AR other keys
    for key in ["EXPID", "MJD"] + keys:
        if key == "EFFTIME_SPEC / EXPTIME":
            mydict[key] = e["EFFTIME_SPEC"] / e["EXPTIME"]
        elif key == "EFFTIME_SPEC * EBVFAC * AIRFAC / EXPTIME":
            ebvfac = 10.0 ** (2 * 2.165 * e["EBV"] / 2.5)
            airfac = e["AIRMASS"] ** 1.75
            mydict[key] = e["EFFTIME_SPEC"] * ebvfac * airfac / e["EXPTIME"]
        elif key == "SKY_MAG_R_SPEC":
            mydict[key] = e["SKY_MAG_R_SPEC"]
        else:
            mydict[key] = e[key]

    # AR color
    mydict["COLOR"] = np.zeros(len(e), dtype=object)
    for progname in prognames:
        sel = np.array([_.lower().replace("1b", "") == progname for _ in e["PROGRAM"]])
        mydict["COLOR"][sel] = progcols[prognames == progname]

    # AR plot
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.3, wspace=0.3)
    for i, (key, xlim, mloc) in enumerate(zip(keys, xlims, mlocs)):

        ax = plt.subplot(gs[i])

        bins = np.linspace(xlim[0] - 1, xlim[1] + 1, 1000)
        if key == "EFFTIME_SPEC * EBVFAC * AIRFAC / EXPTIME":
            weights = np.ones(nexp)
        else:
            weights = e["EXPTIME"]

        # AR all
        sel = np.isin(e["FAPRGRM"], prognames)
        _ = ax.hist(
            mydict[key][sel],
            bins=bins,
            weights=weights[sel],
            density=True,
            cumulative=True,
            color="b",
            histtype="stepfilled",
            alpha=0.3,
            zorder=0,
            label="All",
        )

        # AR per program
        for progname, progcol in zip(prognames, progcols):
            sel = e["FAPRGRM"] == progname
            if sel.sum() > 0:
                _ = ax.hist(
                    mydict[key][sel],
                    bins=bins,
                    weights=weights[sel],
                    density=True,
                    cumulative=True,
                    color=progcol,
                    histtype="step",
                    alpha=0.8,
                    lw=3,
                    zorder=1,
                    label=progname,
                )
                sel &= e["NIGHT"] == e["NIGHT"].max()
                _ = ax.hist(
                    mydict[key][sel],
                    bins=bins,
                    density=True,
                    cumulative=True,
                    color=progcol,
                    histtype="step",
                    alpha=0.5,
                    ls="--",
                    lw=1.5,
                    zorder=1,
                )

        if i == 1:
            ax.set_title(
                "{} {} exposures from {} to {} ({}) ; {} exposures from {} in dashed lines".format(
                    np.isin(e["FAPRGRM"], prognames).sum(),
                    survey,
                    e["NIGHT"].min(),
                    e["NIGHT"].max(),
                    ", ".join(
                        [
                            "{}={}".format(progname, (e["FAPRGRM"] == progname).sum())
                            for progname in prognames
                        ]
                    ),
                    (e["NIGHT"] == e["NIGHT"].max()).sum(),
                    e["NIGHT"].max(),
                )
            )
        ax.set_xlabel(key)
        ax.set_ylabel("Cumulative normalized distribution")
        ax.set_xlim(xlim)
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_locator(MultipleLocator(mloc))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.grid()
        if key in ["TRANSPARENCY_GFA", "SKY_MAG_R_SPEC", "FIBER_FRACFLUX_GFA"]:
            ax.legend(loc=2)
        else:
            ax.legend(loc=4)

    plt.savefig(outpng, bbox_inches="tight")
    plt.close()
