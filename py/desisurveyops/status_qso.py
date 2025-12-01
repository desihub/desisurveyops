#!/usr/bin/env python

import matplotlib

matplotlib.use("Agg")

# AR general
import os
from glob import glob
from datetime import datetime
from time import time
import multiprocessing

# AR scientifical
import numpy as np
import fitsio

# AR astropy
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
from astropy import units as u

# AR desisurveyops
from desisurveyops.status_utils import (
    get_filename,
    get_fns,
    get_obsdone_tiles,
)

# AR desispec
from desispec.io.util import get_tempfilename

# AR desimodel
from desimodel.focalplane.geometry import get_tile_radius_deg

# AR desitarget
from desitarget.geomask import match_to
from desitarget.targetmask import desi_mask
from desitarget.io import read_targets_in_cap

# AR desispec
from desispec.tile_qa_plot import (
    get_qa_badmsks,
)

# AR desiutil
from desiutil.log import get_logger

# AR matplotlib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MultipleLocator


log = get_logger()


def process_qso(
    outdir,
    survey,
    specprod,
    programs,
    skip_passes,
    program_strs,
    numproc,
    recompute=False,
):
    """
    Wrapper function to generate the qso files and plots.

    Args:
        outdir: output folder (str)
        survey: survey name (str)
        specprod: spectroscopic production (e.g. daily) (str)
        programs: list of programs (str)
        skip_passes: passes to skip in each program (np.ndarray of ints)
        program_strs: list of program_strs (str)
        numproc: number of parallel processes to run (int)
        recompute (optional, defaults to False): if True recompute all maps;
            if False, only compute missing maps (bool)

    Notes:
        For (programs, skip_passes, program_strs), see desisurveyops.sky_utils.get_programs_passparams().
        Usually use specprod=daily.
    """

    log.info(
        "{}\tBEGIN process_qso".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    start = time()

    # AR obs, donetiles
    obs_tiles, obs_nights, obs_progs, done_tiles = get_obsdone_tiles(survey, specprod)

    for program, _, program_str in zip(programs, skip_passes, program_strs):

        if program not in ["DARK", "DARK1B"]:
            continue

        # AR output files
        outecsv = get_filename(
            outdir, survey, "qso", program_str=program_str, ext="ecsv"
        )
        outpng = get_filename(outdir, survey, "qso", program_str=program_str, ext="png")
        log.info(outecsv)
        # lastnight = process_qso(outecsv, survey, specprod, program, args.numproc, recompute=args.recompute)
        # if lastnight is not None:
        #    plot_qso_stats(outpng, outecsv, lastnight)

        # AR compute
        log.info(
            "{}\tCompute qso stats".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        sel = obs_progs == program
        tileids, lastnights = obs_tiles[sel], obs_nights[sel]
        _ = np.char.add(tileids.astype(str), ",")
        tileids_lastnights = np.char.add(_, lastnights.astype(str))

        # AR if outecsv already exists and no recompute, only processing new files
        prev_d = None
        if (os.path.isfile(outecsv)) & (not recompute):
            prev_d = Table.read(outecsv, format="ascii")
            _ = np.char.add(prev_d["TILEID"].astype(str), ",")
            prev_tileids_lastnights = np.char.add(_, prev_d["LASTNIGHT"].astype(str))
            sel = ~np.isin(tileids_lastnights, prev_tileids_lastnights)
            tileids, lastnights = tileids[sel], lastnights[sel]
            tileids_lastnights = tileids_lastnights[sel]

        if len(tileids_lastnights) > 0:

            log.info(
                "{}\tprocessing {} (tileid, lastnight) : {}".format(
                    program, len(tileids_lastnights), tileids_lastnights
                )
            )
            myargs = [
                (survey, specprod, tileid, lastnight)
                for tileid, lastnight in zip(tileids, lastnights)
            ]
            pool = multiprocessing.Pool(processes=numproc)
            with pool:
                ds = pool.starmap(compute_qso_tileid_lastnight, myargs)

            d = vstack(ds)
            d.meta["SURVEY"], d.meta["PROGRAM"] = survey, str(program)

            if prev_d is not None:
                d = vstack([prev_d, d])
            tmpfn = get_tempfilename(outecsv)
            d.write(tmpfn)
            os.rename(tmpfn, outecsv)

            plot_qso_stats(outpng, outecsv, lastnights[-1])

    log.info(
        "{}\tEND process_qso (took {:.1f}s)".format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), time() - start
        )
    )


def get_daily_lastnights(tileids):
    """
    Returns the last night.

    Args:
        tileids: list of tile TILEIDs (np.array of ints)

    Return:
        lastnight: list of last night (YYYYMMDD) (np.array of ints)
    """
    # AR read tiles-daily
    fns = get_fns(specprod="daily")
    d = Table.read(fns["spec"]["tiles"])
    ii = match_to(d["TILEID"], tileids)
    assert ii.size == len(tileids)
    return d["LASTNIGHT"][ii]


def get_overlap_tiles(tileid, survey, tileids=None, indesi=True):
    """
    Returns tiles overlapping a given tile.

    Args:
        tileid: tileid (int)
        survey: survey name (str)
        tileids (optional, defaults to None): if provided, only deals with those tiles
                    if not provided, we consider all tiles from the same program
                    (list or numpy array of ints)
        indesi (optional, defaults to True): restrict to IN_DESI tiles? (bool)

    Returns:
        array of tileids (float)
        array of R.A. for tileids (float)
        array of Dec. for tileids (float)

    Notes:
        We exclude from this list the input tileid.
    """
    fn = get_fns(survey=survey)["ops"]["tiles"]
    t = Table.read(fn)
    cs = SkyCoord(t["RA"] * u.degree, t["DEC"] * u.degree, frame="icrs")

    sel = t["TILEID"] != tileid
    if tileids is None:
        program = t["PROGRAM"][t["TILEID"] == tileid][0]
        sel &= t["PROGRAM"] == program
    else:
        sel &= np.isin(t["TILEID"], tileids)
    if indesi:
        sel &= t["IN_DESI"]
    sel &= (
        cs.separation(cs[t["TILEID"] == tileid][0]).value <= 2 * get_tile_radius_deg()
    )
    t = t[sel]

    return t["TILEID"], t["RA"], t["DEC"]


def compute_qso_tileid_lastnight(survey, specprod, tileid, lastnight):
    """
    Compute the qso/lya/newlya statistics for a given {tileid,lastnight}.

    Args:
        survey: survey (str)
        specprod: spectroscopic production (e.g. daily) (str)
        tileid: tileid (int)
        lastnight: lastnight (int)

    Returns:
        outd: an astropy.table.Table with various customs keys; in particular:
            N_QSOTARG, DENS_QSONEW,DENS_LYANEW,N_LYACAND_BEFORE,N_LYCAND_AFTER,N_LYACAND_ALL
    """
    # AR output table
    outd = Table()
    outd["TILEID"] = [tileid]
    outd["LASTNIGHT"] = lastnight

    #
    # AR tile area
    # AR we do *not* account for discarded petals,
    # AR it will be accounted for in the frac_assgn
    tile_radius_deg = get_tile_radius_deg()
    tile_area = np.pi * tile_radius_deg**2

    # AR files
    fns = get_fns(survey=survey, specprod=specprod)

    # AR observed tiles, less then 2 radii from our tileid
    obs_tiles, _, _, _ = get_obsdone_tiles(survey, specprod)
    tileids = get_overlap_tiles(tileid, survey, tileids=obs_tiles)
    t = Table.read(fns["ops"]["tiles"])
    sel = np.isin(t["TILEID"], tileids)
    t = t[sel]

    # AR fiberqa, cut on valid fiber for QSO targets
    fiberqa_fn = os.path.join(
        fns["spec"]["cumuldir"],
        "{}".format(tileid),
        "{}".format(lastnight),
        "tile-qa-{}-thru{}.fits".format(tileid, lastnight),
    )
    fiberqa_hdr = fitsio.read_header(fiberqa_fn, "FIBERQA")
    for key in [
        "TILERA",
        "TILEDEC",
        "SURVEY",
        "FAPRGRM",
        "EFFTIME",
    ]:
        outd[key] = fiberqa_hdr[key]
    fiberqa = fitsio.read(fiberqa_fn, "FIBERQA")
    badqa_val, _ = get_qa_badmsks()
    sel = (fiberqa["QAFIBERSTATUS"] & badqa_val) == 0
    sel &= (fiberqa["DESI_TARGET"] & desi_mask["QSO"]) > 0
    fiberqa = fiberqa[sel]
    tids = fiberqa["TARGETID"]

    # AR Lya candidates from zmtl files
    fns = sorted(
        glob(
            os.path.join(
                os.path.dirname(fiberqa_fn),
                "zmtl-?-{}-thru{}.fits".format(tileid, lastnight),
            )
        )
    )
    if len(fns) == 0:
        qso = np.zeros(len(tids), dtype=bool)
        lya = np.zeros(len(tids), dtype=bool)
    else:
        zmtl = vstack([Table.read(fn) for fn in fns], metadata_conflicts="silent")
        ii = match_to(zmtl["TARGETID"], tids)
        if len(ii) != len(fiberqa):
            log.warning(
                "only {}/{} matched between zmtl and fiberqa; returning None".format(
                    len(ii), len(fiberqa)
                )
            )
            return None
        zmtl = zmtl[ii]
        qso = (fiberqa["SPECTYPE"] == "QSO") | (
            zmtl["IS_QSO_QN"] == 1
        )  # AR cf. email Christophe 20211103
        lya = (zmtl["Z"] >= 2.1) | ((zmtl["Z_QN"] >= 2.1) & (zmtl["IS_QSO_QN"] == 1))

    # AR priorities from fiberassign
    # AR grabbing fiberassign from $DESI_ROOT/spectro/data, so that we are sure it is there...
    fa_fn = glob(
        os.path.join(
            os.getenv("DESI_ROOT"),
            "spectro",
            "data",
            "{}".format(lastnight),
            "*",
            "fiberassign-{:06d}.fits.gz".format(tileid),
        )
    )[0]
    if not os.path.isfile(fa_fn):
        log.warning("no {}; returning None".format(fa_fn))
        return None
    fa_hdr = fitsio.read_header(fa_fn, 0)
    hpdir = fa_hdr["TARG"].replace("DESIROOT", os.getenv("DESI_ROOT"))
    fa = fitsio.read(fa_fn, "FIBERASSIGN")
    ii = match_to(fa["TARGETID"], tids)
    if len(ii) != len(fiberqa):
        log.warning(
            "only {}/{} matched between fiberassign and fiberqa; returning None".format(
                len(ii), len(fiberqa)
            )
        )
        return None
    prios = fa["PRIORITY"][ii]

    # AR QSO parent targets
    targs = read_targets_in_cap(
        hpdir,
        [fiberqa_hdr["TILERA"], fiberqa_hdr["TILEDEC"], tile_radius_deg],
        quick=True,
    )
    sel = (targs["DESI_TARGET"] & desi_mask["QSO"]) > 0
    targs = targs[sel]
    outd["N_QSOTARG"] = len(targs)  # nb of qso targets

    # AR true QSO/Lya which got a first observation (priority=3400)
    first_obs = prios == desi_mask["QSO"].priorities["UNOBS"]
    frac_assgn = first_obs.sum() / len(targs)
    outd["DENS_QSONEW"] = (
        ((qso) & (first_obs)).sum() / tile_area / frac_assgn
    )  # reconstructed density of qso
    outd["DENS_LYANEW"] = (
        ((lya) & (first_obs)).sum() / tile_area / frac_assgn
    )  # reconstructed density of lya

    # AR Lya candidates before that observation (priority=3350)
    nlya = ((prios == desi_mask["QSO"].priorities["MORE_ZGOOD"]) & (lya)).sum()
    outd["N_LYACAND_BEFORE"] = nlya

    # AR Lya candidates newly identified
    nlya = ((prios != desi_mask["QSO"].priorities["MORE_ZGOOD"]) & (lya)).sum()
    outd["N_LYCAND_AFTER"] = nlya

    # AR all Lya candidates
    frac_assgn = len(tids) / len(targs)
    outd["N_LYACAND_ALL"] = lya.sum()

    return outd


def plot_qso_stats(outpng, qsoecsv, lastnight):
    """
    Generate the lya diagnosis status plot.

    Args:
        outpng: output file name (str)
        qsoecsv: the name of the file with all the informations (str)
        lastnight: the night to be highlighted (int)

    Notes:
        qsoecsv is the concatenation of all the per {tileid,lastnight}
            files generated with compute_qso_tileid_lastnight().
    """

    tile_radius_deg = get_tile_radius_deg()
    #
    d = Table.read(qsoecsv)
    survey, program = d.meta["SURVEY"], d.meta["PROGRAM"]

    #
    ylim = (0, 300)
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(1, 2, wspace=0.05, width_ratios=[1, 0.3])
    ax = plt.subplot(gs[0, 0])
    sel = d["LASTNIGHT"] == lastnight
    for key, c, label in zip(
        ["DENS_QSONEW", "DENS_LYANEW"],
        ["b", "r"],
        [
            "QSO: (SPECTYPE = QSO) or (IS_QSO_QN = 1) QSO",
            "Lya: (Z > 2.1) or (Z_QN > 2.1 and IS_QSO_QN = 1) QSO",
        ],
    ):
        ax.scatter(d["EFFTIME"], d[key], c=c, s=10, alpha=0.2, label=label)
        ax.scatter(
            d["EFFTIME"][sel], d[key][sel], facecolors="none", edgecolors="k", s=10
        )
    if sel.sum() > 0:
        ax.scatter(
            np.nan,
            np.nan,
            facecolors="none",
            edgecolors="k",
            s=10,
            label="Observed on {}".format(lastnight),
        )
    ax.set_title(
        "{}/{} : {} obs tiles up to {}".format(
            survey, program, np.unique(d["TILEID"]).size, lastnight
        )
    )
    ax.set_xlabel("EFFTIME_SPEC [s]")
    ax.set_ylabel("Density of newly identified QSO or Lya [/deg2]")
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.set_xlim(750, 2750)
    ax.set_ylim(ylim)
    ax.grid(True)
    ax.legend(loc=1)

    # AR hist
    bins = np.linspace(ylim[0], ylim[1], 200)
    ax = plt.subplot(gs[0, 1])
    for key, c in zip(["DENS_QSONEW", "DENS_LYANEW"], ["b", "r"]):
        _ = ax.hist(
            d[key], bins=bins, color=c, alpha=0.8, orientation="horizontal", label=label
        )
    ax.set_xlabel("N(tile)")
    ax.set_yticklabels([])
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.set_ylim(ylim)
    ax.grid(True)

    plt.savefig(outpng, bbox_inches="tight")
    plt.close()
