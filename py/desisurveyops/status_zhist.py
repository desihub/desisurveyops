#!/usr/bin/env python

import matplotlib

matplotlib.use("Agg")

# AR general
import os
from datetime import datetime
from time import time
import multiprocessing

# AR scientifical
import numpy as np
import fitsio

# AR astropy
from astropy.table import Table, vstack

# AR desisurveyops
from desisurveyops.status_utils import (
    get_filename,
    get_fns,
    get_obsdone_tiles,
    table_read_for_pool,
)

# AR desispec
from desispec.tile_qa_plot import (
    get_qa_config,
    get_zbins,
    get_tracer,
    get_zhists,
    get_qa_badmsks,
    get_quantz_cmap,
    make_tile_qa_plot,
)

# AR fiberassgin

# AR desiutil
from desiutil.log import get_logger

# AR matplotlib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec

log = get_logger()


def process_zhist(
    outdir,
    survey,
    specprod,
    programs,
    npassmaxs,
    program_strs,
    dchi2min,
    numproc,
    recompute=False,
):
    """
    Wrapper function to generate the per {tileid,lastnights} files with the n(z).

    Args:
        outdir: output folder (str)
        survey: survey name (str)
        specprod: spectroscopic production (e.g. daily) (str)
        programs: list of programs (str)
        npassmaxs: list of npassmaxs (str)
        program_strs: list of program_strs (str)
        dchi2_min: DELTACHI2 cut to select reliable zspecs (float)
        numproc: number of parallel processes to run (int)
        recompute (optional, defaults to False): if True recompute all maps;
            if False, only compute missing maps (bool)

    Notes:
        For (programs, npassmaxs, program_strs), see desisurveyops.sky_utils.get_programs_npassmaxs().
        Usually use specprod=daily.
    """

    log.info(
        "{}\tBEGIN process_zhist".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    start = time()

    # AR to get obs. tiles with efftime > 0
    fns = get_fns(survey=survey, specprod=specprod)
    e = Table.read(fns["spec"]["exps"])
    sel = (e["SURVEY"] == survey) & (e["EFFTIME_SPEC"] > 0)
    e = e[sel]

    # AR obs, donetiles
    obs_tiles, obs_nights, obs_progs, done_tiles = get_obsdone_tiles(survey, specprod)

    # AR loop on programs
    for program, npassmax, program_str in zip(programs, npassmaxs, program_strs):

        outfn = get_filename(
            outdir, survey, "zhist", program_str=program_str, ext="ecsv"
        )
        outpng = get_filename(
            outdir, survey, "zhist", program_str=program_str, ext="png"
        )

        # AR per-program setting
        if program == "BACKUP":
            continue
        if program[:4] == "DARK":
            rebin = 2
        else:
            rebin = 1

        # AR select the tiles
        sel = obs_progs == program
        sel &= np.in1d(obs_tiles, e["TILEID"])
        if npassmax is not None:
            t = Table.read(fns["ops"]["tiles"])
            t = t[t["PASS"] < npassmax]
            sel &= np.in1d(obs_tiles, t["TILEID"])

        if npassmax is not None:
            log.info(
                "found {} tiles with PROGRAM = {} and PASS < {}".format(
                    sel.sum(), program, npassmax
                )
            )
        else:
            log.info("found {} tiles with PROGRAM = {}".format(sel.sum(), program))

        # AR handle e.g. BRIGHT1B which does not exist yet
        if sel.sum() == 0:
            log.warning(
                "no tiles found for PROGRAM = {}, no file created".format(program)
            )
            continue

        tileids, lastnights = obs_tiles[sel], obs_nights[sel]
        fns = [
            get_filename(
                outdir, survey, "zhist", tileid=tileid, night=lastnight, ext="ecsv"
            )
            for tileid, lastnight in zip(tileids, lastnights)
        ]

        # AR compute?
        myargs = []
        for i, (tileid, lastnight, fn) in enumerate(zip(tileids, lastnights, fns)):
            if (not os.path.isfile(fns[i])) or (recompute):
                # if "{},{}".format(tileid, lastnight) in black_tileids_nights:
                if False:
                    log.info("zhist: ignore {},{}".format(tileid, lastnight))
                else:
                    if (recompute) & (os.path.isfile(fns[i])):
                        log.info("zhist: remove existing {}".format(fns[i]))
                        os.remove(fns[i])
                    myargs.append((fn, specprod, tileid, lastnight, program, dchi2min))
                    log.info("zhist: (re)compute {},{}".format(tileid, lastnight))
        if len(myargs) > 0:
            log.info(
                "{} launch computing for {} {} tiles".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    len(myargs),
                    program_str,
                )
            )
            pool = multiprocessing.Pool(numproc)
            with pool:
                _ = pool.starmap(compute_zhist, myargs)
            log.info(
                "{} computing done".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )

        # AR merge
        use_fitsio, columns = False, None
        myargs = [(fn, use_fitsio, columns) for fn in fns]
        log.info(
            "{}\tstart reading {} files for {}".format(
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"), len(myargs), program_str
            )
        )
        pool = multiprocessing.Pool(numproc)
        with pool:
            ds = pool.starmap(table_read_for_pool, myargs)
        log.info(
            "{}\treading done".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )

        # AR handle / null keywords
        keys = ["DCHI2MIN", "FAPRGRM", "GOALTIME", "MINTFRAC", "SURVEY", "TRACERS"]
        for key in ds[0].meta:
            if key not in keys:
                for d in ds:
                    d.meta[key] = None
            else:
                assert np.unique([d.meta[key] for d in ds]).size == 1

        # AR vstack + write
        d = vstack(ds)
        d.meta["SURVEY"], d.meta["SPECPROD"] = survey, specprod
        d.write(outfn, overwrite=True)

        # AR plot
        plot_zhist(
            outpng,
            outdir,
            survey,
            specprod,
            program,
            npassmax,
            program_str,
            lastnights.max(),
            rebin,
        )

    log.info(
        "{}\tEND process_zhist (took {:.1f}s)".format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), time() - start
        )
    )


def compute_zhist(outfn, specprod, tileid, night, program, dchi2_min):
    """
    Compute the n(z) from tile-qa-{tileid}-thru{night}.fits and write to a file.

    Args:
        outfn: output file name (str)
        survey: survey name (str)
        tileid: tileid (int)
        night: night (int)
        program: "BACKUP", "BRIGHT", or "DARK" (string)
        dchi2_min: DELTACHI2 cut to select reliable zspecs (float)

    Notes:
        Our fiducial choice is dchi2_min=25.
    """

    fns = get_fns(specprod=specprod)
    tracers, _ = get_prgrm_traccols(program)
    config = get_qa_config()

    tilesdir = os.path.join(
        os.getenv("DESI_ROOT"), "spectro", "redux", specprod, "tiles", "cumulative"
    )
    fiberqa_fn = os.path.join(
        fns["spec"]["cumuldir"],
        "{}".format(tileid),
        "{}".format(night),
        "tile-qa-{}-thru{}.fits".format(tileid, night),
    )
    # AR sometimes, mostly for backup, the pipeline is not run
    # AR instead of using a blacklist as before,
    # AR we just ignore those
    # AR they should be picked up later if they appear
    if not os.path.isfile(fiberqa_fn):
        log.warning("missing {}".format(fiberqa_fn))
        return None

    fiberqa_hdr = fitsio.read_header(fiberqa_fn, "FIBERQA")
    fiberqa = fitsio.read(fiberqa_fn, "FIBERQA")
    mydict = {"ZMIN": [], "ZMAX": [], "ZHIST": [], "TRACER": []}
    hdr = fitsio.FITSHDR()
    for key in [
        "TILEID",
        "LASTNITE",
        "SURVEY",
        "FAPRGRM",
        "EFFTIME",
        "GOALTIME",
        "MINTFRAC",
    ]:
        hdr[key] = fiberqa_hdr[key]
    hdr["TRACERS"] = ",".join(tracers)
    hdr["DCHI2MIN"] = dchi2_min
    for it, tracer in enumerate(tracers):
        zbins, zhist = get_zhists(tileid, tracer, dchi2_min, fiberqa)
        mydict["ZMIN"] += zbins[:-1].tolist()
        mydict["ZMAX"] += zbins[1:].tolist()
        mydict["ZHIST"] += zhist.tolist()
        mydict["TRACER"] += [tracer for x in zhist]
        hdr["NTRAC{}".format(it)] = get_tracer(tracer, fiberqa).sum()
    #
    d = Table()
    nrow = len(mydict["TRACER"])
    d["TILEID"] = np.full(nrow, tileid)
    d["LASTNIGHT"] = np.full(nrow, night)
    for key in ["TRACER", "ZMIN", "ZMAX", "ZHIST"]:
        d[key] = mydict[key]
    d.meta = dict(hdr)
    d.write(outfn)


def get_prgrm_traccols(program):
    """
    Get the plotted tracers and colors for the n(z) plots.

    Args:
        program: "BACKUP", "BRIGHT", or "DARK" (str)

    Returns:
        tracers: list of tracers (str)
        colors: list of colors (str)
    """
    config = get_qa_config()
    tracers = [
        tracer
        for tracer in list(config["tile_qa_plot"]["tracers"].keys())
        if program in config["tile_qa_plot"]["tracers"][tracer]["program"].split(",")
    ]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][: len(tracers)]
    return tracers, colors


def plot_zhist(
    outpng,
    outdir,
    survey,
    specprod,
    program,
    npassmax,
    program_str,
    cutoff_night,
    rebin,
):
    """
    Make the n(z) plot for the overview page.

    Args:
        outpng: output image file name (str)
        outdir: output folder name (equivalent to $DESI_ROOT/survey/observations/main/) (str)
        survey: survey name (str)
        specprod: spectroscopic production (e.g. daily) (str)
        programs: list of programs (str)
        npassmaxs: list of npassmaxs (str)
        program_strs: list of program_strs (str)
        cutoff_night: only keep tiles with lastnight<=cutoff_night (int)
        rebin: what re-binning of the zbins? (int)

    Notes:
        For (programs, npassmaxs, program_strs), see desisurveyops.sky_utils.get_programs_npassmaxs().
    """

    log.info("outpng: {}".format(outpng))

    tracers, colors = get_prgrm_traccols(program)

    fn = get_filename(outdir, survey, "zhist", program_str=program_str, ext="ecsv")
    d = Table.read(fn)
    dchi2_min = d.meta["DCHI2MIN"]
    sel = d["LASTNIGHT"] <= cutoff_night
    d = d[sel]
    ntile = np.unique(d["TILEID"]).size

    tilesfn = get_fns(survey=survey, specprod=specprod)["ops"]["tiles"]
    t = Table.read(tilesfn)
    sel = t["PROGRAM"] == program
    sel &= t["IN_DESI"]
    if npassmax is not None:
        sel &= t["PASS"] < npassmax
    t = t[sel]

    title = "{}/{}\n{}/{} (={:.0f}%) completed tiles up to {}".format(
        survey.capitalize(),
        program_str,
        ntile,
        len(t),
        100.0 * ntile / len(t),
        cutoff_night,
    )

    fig, ax = plt.subplots()
    zmins, zmaxs = np.unique(d["ZMIN"]), np.unique(d["ZMAX"])
    zcens = 0.5 * (zmins + zmaxs)
    nbin = len(zcens)
    remainder = nbin % rebin
    if remainder == 0:
        rebin_zcens = zcens.reshape((nbin // rebin, rebin)).mean(axis=1)
    else:
        rebin_zcens = zcens[:-remainder].reshape((nbin // rebin, rebin)).mean(axis=1)
    nrebin = len(rebin_zcens)
    tns = np.array(["{}-{}".format(t, n) for t, n in zip(d["TILEID"], d["LASTNIGHT"])])
    unq_tns = np.unique(tns)

    for tracer, color in zip(tracers, colors):

        # AR gather all zhists
        dd = d[d["TRACER"] == tracer]
        assert len(dd) == unq_tns.size * nbin
        hs = dd["ZHIST"].reshape((unq_tns.size, nbin))
        nights = dd["LASTNIGHT"].reshape((unq_tns.size, nbin))[:, 0]

        # AR rebin
        if remainder == 0:
            rebin_hs = hs.reshape((unq_tns.size, nbin // rebin, rebin)).sum(axis=-1)
        else:
            rebin_hs = (
                hs[:, :-remainder]
                .reshape((unq_tns.size, nbin // rebin, rebin))
                .sum(axis=-1)
            )

        # AR highlighting tiles observed last night
        ii = np.where(nights == cutoff_night)[0]
        for i in ii:
            ax.plot(
                rebin_zcens, rebin_hs[i, :], color="k", lw=0.25, alpha=1.0, zorder=1
            )

        # AR plot
        mean_rebin_hs = np.nanmean(rebin_hs, axis=0)
        ax.plot(rebin_zcens, mean_rebin_hs, color=color, lw=2, alpha=1.0, zorder=2)
        for lo_perc, hi_perc, alpha in zip(
            [2.1, 13.6, 34.1],
            [68.2, 95.5, 99.7],
            [0.50, 0.25, 0.05],
        ):
            lo_rebin_hs = np.nanpercentile(rebin_hs, lo_perc, axis=0)
            hi_rebin_hs = np.nanpercentile(rebin_hs, hi_perc, axis=0)
            ax.fill_between(
                rebin_zcens,
                lo_rebin_hs,
                hi_rebin_hs,
                color=color,
                alpha=alpha,
                zorder=0,
            )
        if tracer == "ELG_LOP":
            label = "ELG_LOP not QSO"
        else:
            label = tracer
        ax.plot(np.nan, np.nan, color=color, label=label)

    ax.plot(
        np.nan,
        np.nan,
        color="k",
        lw=0.25,
        alpha=1.0,
        zorder=1,
        label="Observed on {}".format(cutoff_night),
    )
    ax.set_title(title)
    ax.set_xlabel("Z")
    ax.set_ylabel("Per tile fractional count")
    if program == "BRIGHT":
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.4)
    if program == "DARK":
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 0.3)
    ax.grid()
    ax.set_axisbelow(True)
    ax.legend(loc=1)

    plt.savefig(outpng, bbox_inches="tight", dpi=300)
    plt.close()
