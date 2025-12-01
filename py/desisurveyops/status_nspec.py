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
from astropy.time import Time
from astropy import constants

# AR desisurveyops
from desisurveyops.status_utils import (
    get_filename,
    get_fns,
    get_obsdone_tiles,
    get_programs_passparams,
    get_spec_updated_mjd,
    get_tileid_night_str,
    get_shutdowns,
    table_read_for_pool,
)

# AR desitarget
from desitarget.targets import zcut as lya_zcut
from desitarget.geomask import match, match_to
from desitarget.targetmask import desi_mask, bgs_mask
from desitarget.targetmask import zwarn_mask as desitarget_zwarn_mask

# AR desiutil
from desiutil.log import get_logger

# AR matplotlib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec

log = get_logger()


def process_nspec(
    outdir, survey, specprod, specprod_ref, dchi2min, numproc, recompute=False
):
    """
    Wrapper function to generate the nspec=f(night) plot
        (along with the underlying zmtl files used for that).

    Args:
        outdir: output folder (str)
        survey: survey name (str)
        specprod: spectroscopic production (e.g. daily) (str)
        specprod_ref: reference spectroscopic production (e.g. loa) (str)
        dchi2_min: DELTACHI2 cut to select reliable zspecs (float)
        numproc: number of parallel processes to run (int)
        recompute (optional, defaults to False): if True recompute all maps;
            if False, only compute missing maps (bool)

    Notes:
        Usually use specprod=daily and specprod_ref=loa.
        For each {tileid,lastnight}, we used specprod_ref if in there, specprod else.
    """

    log.info(
        "{}\tBEGIN process_nspec".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    start = time()

    nspecfn = get_filename(outdir, survey, "nspec", ext="ecsv")
    nspecpng = get_filename(outdir, survey, "nspec", ext="png")

    # AR obs tiles
    obs_tiles, obs_nights, obs_progs, _ = get_obsdone_tiles(survey, specprod)
    obs_tileids_nights = get_tileid_night_str(obs_tiles, obs_nights)
    log.info(
        "\t{:.1f}s\tfound {} tileids_nights".format(
            time() - start, len(obs_tileids_nights)
        )
    )

    # AR grab the UPDATED column
    # AR use specprod_ref if available, specprod else
    # AR the UPDATED information
    _, obs_ref_updated_mjds = get_spec_updated_mjd(specprod_ref, obs_tiles, obs_nights)
    _, obs_updated_mjds = get_spec_updated_mjd(specprod, obs_tiles, obs_nights)
    sel = np.isfinite(obs_ref_updated_mjds)
    obs_updated_mjds[sel] = obs_ref_updated_mjds[sel]

    obs_yrs = obs_nights // 10000
    unq_obs_yrs = np.unique(obs_yrs)

    zmtlfns = np.array(
        [
            get_filename(outdir, survey, "nspec", case="zmtl", yr=obs_yr, ext="fits")
            for obs_yr in unq_obs_yrs
        ]
    )
    isprevs = np.array([os.path.isfile(_) for _ in zmtlfns])
    zmtls = np.array([None for _ in zmtlfns])

    # AR for sanity checks
    hdr_keys = ["SURVEY", "PROD", "REFPROD", "DCHI2MIN"]
    hdr_vals = [survey, specprod, specprod_ref, dchi2min]

    # AR read existing zmtl files
    if isprevs.sum() > 0:
        use_fitsio, keys = True, None
        log.info(
            "\t{:.1f}s\tstart reading {}".format(
                time() - start,
                ",".join([os.path.basename(_) for _ in zmtlfns[isprevs]]),
            )
        )
        myargs = [(zmtlfn, use_fitsio, keys) for zmtlfn in zmtlfns[isprevs]]
        pool = multiprocessing.Pool(processes=np.min([numproc, len(myargs)]))
        with pool:
            zmtls[isprevs] = pool.starmap(table_read_for_pool, myargs)
        log.info(
            "\t{:.1f}s\tdone reading {}".format(
                time() - start,
                ",".join([os.path.basename(_) for _ in zmtlfns[isprevs]]),
            )
        )

        # AR sanity check
        hdrs = [fitsio.read_header(_, 1) for _ in zmtlfns[isprevs]]
        for key, val in zip(hdr_keys, hdr_vals):
            assert np.all(np.array([hdr[key] for hdr in hdrs]) == val)

    # AR first update zmtl files, if need be
    log.info("\t{:.1f}s\tcheck if need to update the zmtl files".format(time() - start))

    n_recompute = 0

    # AR if zmtlfn already exists and no recompute, only processing new/updated files
    for i, (obs_yr, prev_d) in enumerate(zip(unq_obs_yrs, zmtls)):

        sel = obs_yrs == obs_yr

        # AR case where the zmtlfn file is here and recompute=False
        # AR we look for:
        # AR - (tileid, night) not present
        # AR - (tileid, night) with earlier UPDATED value (ie which have been
        # AR        reprocessed in the meantime)
        if (prev_d is not None) & (~recompute):

            prev_tileids_nights = get_tileid_night_str(
                prev_d["ZTILEID"], prev_d["LASTNIGHT"]
            )
            prev_tileids_nights, ii = np.unique(prev_tileids_nights, return_index=True)
            prev_updated_mjds = prev_d["UPDATED_MJD"][ii]
            log.info(
                "\t{:.1f}s\t{}\tfound {} prev_tileids_nights".format(
                    time() - start,
                    os.path.basename(zmtlfns[i]),
                    prev_tileids_nights.size,
                )
            )

            # AR new (tileid, night)
            sel_new = (sel) & (~np.isin(obs_tileids_nights, prev_tileids_nights))

            # AR updated (tileid, night)
            prev_ii, obs_ii = match(prev_tileids_nights, obs_tileids_nights)
            sel_reprocessed = np.zeros(len(obs_tileids_nights), dtype=bool)
            if prev_ii.size > 0:
                common_prev_tileids_nights = prev_tileids_nights[prev_ii]
                common_prev_updated_mjds = prev_updated_mjds[prev_ii]
                common_obs_tileids_nights = obs_tileids_nights[obs_ii]
                common_obs_updated_mjds = obs_updated_mjds[obs_ii]
                common_sel = common_obs_updated_mjds > common_prev_updated_mjds
                updated_tileids_nights = common_obs_tileids_nights[common_sel]
                sel_reprocessed = (sel) & (
                    np.isin(obs_tileids_nights, updated_tileids_nights)
                )
            sel = (sel_new) | (sel_reprocessed)
        log.info(
            "\t{:.1f}s\t{}\tfound {} obs_tileids_nights to process".format(
                time() - start, os.path.basename(zmtlfns[i]), sel.sum()
            )
        )

        myargs = [
            (obs_tileid, obs_night, specprod, specprod_ref, dchi2min)
            for obs_tileid, obs_night in zip(obs_tiles[sel], obs_nights[sel])
        ]
        n_recompute += len(myargs)

        # AR process?
        if len(myargs) > 0:

            log.info(
                "\t{:.1f}\t{}\tprocessing {}".format(
                    time() - start,
                    os.path.basename(zmtlfns[i]),
                    obs_tileids_nights[sel],
                )
            )
            pool = multiprocessing.Pool(processes=numproc)
            with pool:
                ds = pool.starmap(gather_tileid_zmtls, myargs)
            d = vstack(ds, metadata_conflicts="silent")

            # AR append / write
            if (prev_d is not None) & (~recompute):
                # AR rows to update: we delete those
                tileids_nights = get_tileid_night_str(d["ZTILEID"], d["LASTNIGHT"])
                tileids_nights = np.unique(tileids_nights)
                assert np.all(tileids_nights == np.sort(obs_tileids_nights[sel]))
                rmv_tileids_nights = prev_tileids_nights[
                    np.isin(prev_tileids_nights, tileids_nights)
                ]
                rmv_rows = np.where(
                    np.isin(
                        get_tileid_night_str(prev_d["ZTILEID"], prev_d["LASTNIGHT"]),
                        rmv_tileids_nights,
                    )
                )[0]
                log.info(
                    "\t{:.1f}\t{}\tremove {} rows from existing table".format(
                        time() - start, os.path.basename(zmtlfns[i]), rmv_rows.size
                    )
                )
                prev_d.remove_rows(rmv_rows)
                prev_tileids_nights = get_tileid_night_str(
                    prev_d["ZTILEID"], prev_d["LASTNIGHT"]
                )
                prev_tileids_nights = np.unique(prev_tileids_nights)
                assert np.all(~np.isin(prev_tileids_nights, tileids_nights))
                # AR we stack
                d = vstack([prev_d, d], metadata_conflicts="silent")

            # AR we sort by night, tileid, fiber
            ii = np.lexsort([d["FIBER"], d["ZTILEID"], d["LASTNIGHT"]])
            d = d[ii]

            # AR remove possible different lastnights for a given tileid
            # AR in case a tile is observed over different nights
            unq_tileids = np.unique(d["ZTILEID"])
            rmv_rows = []
            for tileid in unq_tileids:
                unq_lastnights = np.unique(d["LASTNIGHT"][d["ZTILEID"] == tileid])
                rows = np.where(
                    (d["ZTILEID"] == tileid) & (d["LASTNIGHT"] != unq_lastnights[-1])
                )[0]
                if rows.size > 0:
                    log.info(
                        "\t{:.1f}s\t{}\tremove {} rows from deprecated LASTNIGHT".format(
                            time() - start, os.path.basename(zmtlfns[i]), rows.size
                        )
                    )
                rmv_rows += rows.tolist()
            d.remove_rows(rmv_rows)

            # AR header infos
            d.meta["SURVEY"] = survey
            d.meta["PROD"] = specprod
            d.meta["REFPROD"] = specprod_ref
            d.meta["DCHI2MIN"] = dchi2min

            # AR write
            d.write(zmtlfns[i], overwrite=True)
            zmtls[i] = d

    # AR now build the nspec.ecsv file
    if (n_recompute > 0) | (not os.path.isfile(nspecfn)) | (recompute):
        log.info("\t{:.1f}s\tvstack {} zmtl files".format(time() - start, len(zmtls)))
        zmtl = vstack(zmtls.tolist())
        for key, val in zip(hdr_keys, hdr_vals):
            zmtl.meta[key] = val
        # AR make the catalog lighter...
        keys = [
            "TARGETID",
            "DESI_TARGET",
            "BGS_TARGET",
            "SCND_TARGET",
            "ZTILEID",
            "LASTNIGHT",
            "TGT_VALID",
            "ZOK",
            "ZSTAR",
            "LYA",
        ]
        zmtl.keep_columns(keys)
        log.info("\t{:.1f}s\tcompute nspec".format(time() - start))
        compute_nspec(nspecfn, zmtl, dchi2min, numproc)

    # AR then plot
    log.info("\t{:.1f}s\tplot nspec".format(time() - start))
    plot_nspec(survey, nspecfn, nspecpng)

    log.info(
        "{}\tEND process_nspec (took {:.1f}s)".format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), time() - start
        )
    )


def gather_tileid_zmtls(tileid, lastnight, specprod, specprod_ref, dchi2min):
    """
    Gather the informations from the zmtl files for a {tileid,lastnight}.

    Args:
        tileid: tileid (int)
        night: night (int)
        specprod: spectroscopic production, expected to be daily ..(string)
        specprod_ref: reference spec. prod (string)
        dchi2_min: DELTACHI2 cut to select reliable zspecs (float)

    Returns:
        d: an astropy.table.Table() with various informations, coming from
            the fiberassign file, the zmtl files.
    """

    # AR find the zmtl files
    # AR first try specprod_ref, then specprod
    sp = specprod_ref
    fns = sorted(
        glob(
            os.path.join(
                os.getenv("DESI_ROOT"),
                "spectro",
                "redux",
                sp,
                "tiles",
                "cumulative",
                "{}".format(tileid),
                "{}".format(lastnight),
                "zmtl-?-{}-thru{}.fits".format(tileid, lastnight),
            )
        )
    )
    if len(fns) == 0:
        sp = specprod
        fns = sorted(
            glob(
                os.path.join(
                    os.getenv("DESI_ROOT"),
                    "spectro",
                    "redux",
                    sp,
                    "tiles",
                    "cumulative",
                    "{}".format(tileid),
                    "{}".format(lastnight),
                    "zmtl-?-{}-thru{}.fits".format(tileid, lastnight),
                )
            )
        )
    if len(fns) == 0:
        msg = "no zmtl-?-{}-thru{}.fits files found neither in {} nor in {}; exiting".format(
            tileid, night, specprod_ref, specprod
        )
        log.error(msg)
        raise ValueError(msg)

    # AR read the zmtl files
    d = vstack(
        [Table(fitsio.read(fn, "ZMTL")) for fn in fns], metadata_conflicts="silent"
    )
    d["LASTNIGHT"] = lastnight
    d["SPECPROD"] = sp
    d.meta["DCHI2MIN"] = dchi2min

    # AR grab FIBER, PRIORITY_INIT, PRIORITY
    keys = ["FIBER", "PRIORITY_INIT", "PRIORITY"]
    fm = vstack(
        [
            Table(
                fitsio.read(
                    fn.replace("zmtl", "redrock"),
                    ext="FIBERMAP",
                    columns=["TARGETID"] + keys,
                )
            )
            for fn in fns
        ],
        metadata_conflicts="silent",
    )
    assert np.all(fm["TARGETID"] == d["TARGETID"])
    for key in keys:
        d[key] = fm[key]

    # AR sky fibers
    d["SKY"] = False
    for name in ["SKY", "BAD_SKY", "SUPP_SKY"]:
        d["SKY"] |= (d["DESI_TARGET"] & desi_mask[name]) > 0

    # AR not valid fibers
    # AR https://github.com/desihub/desitarget/blob/38612d939c10ac60b0b6749dc32b1913158fe2ab/py/desitarget/mtl.py#L529-L541
    nodata = d["ZWARN"] & desitarget_zwarn_mask["NODATA"] != 0
    badqa = d["ZWARN"] & desitarget_zwarn_mask.mask("BAD_SPECQA|BAD_PETALQA") != 0

    # AR valid TGT fibers
    # AR and except for Lya, with dchi2 > dchi2min and zwarn = 0
    d["TGT_VALID"] = (d["TARGETID"] > 0) & (~d["SKY"]) & (~nodata) & (~badqa)

    d["ZOK"] = (d["DELTACHI2"] > dchi2min) & (d["ZWARN"] == 0)
    # AR star (see David s suggestion in [desi-survey 3760])
    d["ZSTAR"] = constants.c.to("km/s").value * d["Z"] < 600
    # AR lya
    d["LYA"] = (d["PRIORITY"] == desi_mask["QSO"].priorities["MORE_ZGOOD"]) | (
        (d["PRIORITY"] == desi_mask["QSO"].priorities["UNOBS"])
        & ((d["Z"] > lya_zcut) | ((d["Z_QN"] > lya_zcut) & (d["IS_QSO_QN"] == 1)))
    )

    # AR the UPDATED information
    d["UPDATED"], d["UPDATED_MJD"] = get_spec_updated_mjd(specprod, tileid, lastnight)

    return d


def compute_nspec_prog_name_thrunight(zmtl, isprogs, progshort, name):
    """
    Compute the nspec=f(night) for a given {progshort, name}.

    Args:
        zmtl: astropy.table.Table concatenation of the zmtl-YYYY.fits file,
            "pre-processed" (see Notes)
        isprogs: dictionary listing the tileids for each program (dict)
        progshort: the program name (without the 1B) (str)
        name: custom names of subsamples:
            "MWS_ANY", "BGS_BRIGHT", "BGS_FAINT"
            "LRG", "ELG", "QSO", "SCND_ANY", SCND_ANY_ONLY"
            "ALL"

    Returns:
        nspec: astropy.table.Table with the counting of spectra per night,
            for the requested {progshort, name}.

    Notes:
        WARNING: the zmtl table needs here to be "pre-processed" (see compute_nspec()):
            - sorted first by targetid, then by increasing night
            - cut on valid tgt
        Also, we do not want to modify zmtl, as it is used in mulitple processes
            so work (carefully) with indexes; that is a bit painful...
            but zmtl is a very large table, so we don t want to copy it.
    """

    log.info("\t\trun for (progshort, name) = ({}, {})".format(progshort, name))

    # AR sample selection for that prog
    sel = isprogs[progshort].copy()
    if name == "SCND_ANY_ONLY":
        sel &= zmtl["DESI_TARGET"] == desi_mask["SCND_ANY"]
    elif name == "ALL":
        sel &= np.ones(len(zmtl), dtype=bool)
    elif name in ["BGS_BRIGHT", "BGS_FAINT"]:
        sel &= (zmtl["BGS_TARGET"] & bgs_mask[name]) > 0
    else:
        assert name in ["MWS_ANY", "LRG", "ELG", "QSO", "SCND_ANY"]
        sel &= (zmtl["DESI_TARGET"] & desi_mask[name]) > 0
    ii = np.where(sel)[0]

    # AR now just cumsum how many counts we have per night
    nspec = Table()
    nspec["THRUNIGHT"] = np.unique(zmtl["LASTNIGHT"][ii])
    nspec["PROGRAM"], nspec["NAME"] = progshort, name
    nspec["MJD"] = [
        Time(datetime.strptime("{}".format(night), "%Y%m%d")).mjd
        for night in nspec["THRUNIGHT"]
    ]

    # AR ntile
    _, jj = np.unique(zmtl["ZTILEID"][ii], return_index=True)
    jj = ii[jj]
    _, counts = np.unique(zmtl["LASTNIGHT"][jj], return_counts=True)
    assert np.all(_ == nspec["THRUNIGHT"])
    nspec["NTILE"] = np.cumsum(counts)

    # AR per-type
    for key, sel in [
        ("UNQ_STAR", (zmtl["ZOK"]) & (zmtl["ZSTAR"])),
        ("UNQ_GALQSO", (zmtl["ZOK"]) & (~zmtl["ZSTAR"])),
        ("UNQ_LYA", zmtl["LYA"]),
        ("UNQ_ALL", (zmtl["ZOK"]) | (zmtl["LYA"])),
        ("UNQ_ALL", zmtl["ZOK"]),
    ]:
        jj = np.where(sel)[0]
        jj = jj[np.isin(jj, ii)]

        # AR retrieve the nights and counts for that key
        _, kk = np.unique(zmtl["TARGETID"][jj], return_index=True)
        jj = jj[kk]
        nights, counts = np.unique(zmtl["LASTNIGHT"][jj], return_counts=True)

        # AR now populate nspec
        jj = match_to(nspec["THRUNIGHT"], nights)
        assert np.all(nspec["THRUNIGHT"][jj] == nights)
        nspec[key] = 0
        nspec[key][jj] = np.cumsum(counts)

        # AR in case no observations for a night, fill with values from previous index
        # AR except for first night..
        # AR and go one by one, to handle cases where several nights in a row didn t get obs.
        jj = np.where(
            (nspec[key] == 0) & (nspec["THRUNIGHT"] != nspec["THRUNIGHT"][0])
        )[0]
        for j in jj:
            nspec[key][j] = nspec[key][j - 1]

    return nspec


def compute_nspec(outfn, zmtl, numproc, max_numproc=16):
    """
    Compute the nspec=f(night) for programs.

    Args:
        outfn: output ecsv file name (str)
        zmtl: astropy.table.Table concatenation of the zmtl-YYYY.fits file
        numproc: number of parallel processes to run (int)
        max_numproc (optional, defaults to 16): max. number of parallel processes (int)

    Notes:
        We use here max_numproc, as the parallel processes deal with the zmtl table,
            which is a very large one.
    """

    survey = zmtl.meta["SURVEY"]
    specprod = zmtl.meta["PROD"]
    specprod_ref = zmtl.meta["REFPROD"]
    dchi2min = zmtl.meta["DCHI2MIN"]

    # AR cut on targets with a valid fiber
    zmtl = zmtl[zmtl["TGT_VALID"]]

    # AR sort first by unique tid, then by increasing night
    ii = np.lexsort([zmtl["LASTNIGHT"], zmtl["TARGETID"]])
    zmtl = zmtl[ii]

    # AR files
    fns = get_fns(survey=survey, specprod=specprod)

    # AR first night of the survey
    e = Table.read(fns["spec"]["exps"])
    sel = (e["SURVEY"] == survey) & (e["EFFTIME_SPEC"] > 0)
    e = e[sel]
    survey_first_night = e["NIGHT"].min()

    # AR get tile programs
    t = Table.read(fns["ops"]["tiles"])
    t = t[t["IN_DESI"]]

    # AR identify tileids per program (based on main...)
    all_programs, _, _ = get_programs_passparams(survey=survey)
    all_programs = np.unique(all_programs)
    all_progshorts = np.array([program.replace("1B", "") for program in all_programs])
    progdict = {
        progshort: all_programs[all_progshorts == progshort]
        for progshort in np.unique(all_progshorts)
    }
    prog_tileids = {}
    for progshort in progdict:
        sel = np.isin(t["PROGRAM"], progdict[progshort])
        prog_tileids[progshort] = t["TILEID"][sel]

    # AR programs
    isprogs = {
        progshort: np.isin(zmtl["ZTILEID"], prog_tileids[progshort])
        for progshort in prog_tileids
    }
    isprogs["ALL"] = np.ones(len(zmtl), dtype=bool)

    # AR compute
    # TODO: try to generalize the per-tracer choice for a survey != main
    myargs = []
    prog_name_thrunights = []
    for progshort in isprogs:
        isprogshort = isprogs[progshort]
        names = ["ALL"]
        if progshort == "BACKUP":
            names += ["MWS_ANY"]
        if progshort == "BRIGHT":
            names += ["MWS_ANY", "BGS_BRIGHT", "BGS_FAINT", "SCND_ANY", "SCND_ANY_ONLY"]
        if progshort == "DARK":
            names += ["LRG", "ELG", "QSO", "SCND_ANY", "SCND_ANY_ONLY"]
        for name in names:
            myargs.append((zmtl, isprogs, progshort, name))
    log.info(
        "\tlaunch compute_nspec_prog_name_thrunight() on {} prog_name_thrunight".format(
            len(myargs)
        ),
    )
    pool = multiprocessing.Pool(processes=np.min([numproc, max_numproc]))
    with pool:
        nspecs = pool.starmap(compute_nspec_prog_name_thrunight, myargs)
    log.info("\t_compute_nspec on prog_name_thrunights done")
    d = vstack(nspecs)

    d.meta["SURVEY"] = survey
    d.meta["PROD"] = specprod
    d.meta["REFPROD"] = specprod_ref
    d.meta["DCHI2MIN"] = dchi2min

    d.write(outfn, overwrite=True)


def plot_nspec(survey, nspecfn, nspecpng):
    """
    Make the nspec=f(night) plot.

    Args:
        survey: survey name (str)
        nspecfn: path to the nspec.ecsv file (str)
        nspecpng: output image file name (str)

    Notes:
        nspecfn is the file generated with compute_nspec().
    """
    # AR read per-year nspec files
    d = Table.read(nspecfn)
    dchi2min = d.meta["DCHI2MIN"]

    # AR
    if survey == "main":
        progs = ["BRIGHT", "DARK", "ALL"]
        proglabs = ["BRIGHT{1B}", "DARK{1B}", "BACKUP+BRIGHT{1B}+DARK{1B}"]
        title_fss = [12, 12, 10]
    else:
        msg = "survey={} not handled yet; only main survey".format(survey)
        log.error(msg)
        raise ValueError(msg)

    # AR start plot
    fig = plt.figure(figsize=(20, 5))
    gs = gridspec.GridSpec(1, 3, wspace=0.25)

    # TODO: if possible, generalize for survey != main
    for iy, (prog, proglab, title_fs) in enumerate(zip(progs, proglabs, title_fss)):
        isprog = d["PROGRAM"] == prog
        ax = fig.add_subplot(gs[iy])
        if prog == "BRIGHT":
            names = ["BGS_BRIGHT", "BGS_FAINT", "MWS_ANY", "SCND_ANY_ONLY", "ALL"]
            cols = ["r", "b", "y", "g", "k"]
        elif prog == "DARK":
            names = ["LRG", "ELG", "QSO", "SCND_ANY_ONLY", "ALL"]
            cols = ["r", "b", "y", "g", "k"]
        elif prog == "ALL":
            names = ["ALL"]
            cols = ["k"]
        else:
            sys.exit("wrong prog; exiting")

        for name, col in zip(names, cols):
            sel = (isprog) & (d["NAME"] == name)
            if name == "ALL":
                ax.plot(
                    d["MJD"][sel],
                    1e-6 * d["UNQ_GALQSO"][sel],
                    color=col,
                    lw=4,
                    label="{:.1f}M SPEC_GAL/QSO".format(
                        1e-6 * d["UNQ_GALQSO"][sel].max()
                    ),
                )
                # ax.plot(d["MJD"][sel], 1e-6 * d["UNQ_GALQSO_ZGT1"][sel], color=col, lw=4, label="{:.1f}M SPEC_Z>1".format(1e-6 * d["UNQ_GALQSO_ZGT1"][sel].max()))
                ax.plot(
                    d["MJD"][sel],
                    1e-6 * d["UNQ_STAR"][sel],
                    color=col,
                    lw=2,
                    label="{:.1f}M SPEC_STAR".format(1e-6 * d["UNQ_STAR"][sel].max()),
                )
            else:
                ax.plot(
                    d["MJD"][sel],
                    1e-6 * d["UNQ_ALL"][sel],
                    color=col,
                    label="{:.1f}M {} targets".format(
                        1e-6 * d["UNQ_ALL"][sel].max(), name
                    ),
                )
                if (prog == "DARK") & (name == "QSO"):
                    ax.plot(
                        d["MJD"][sel],
                        1e-6 * d["UNQ_LYA"][sel],
                        color="k",
                        lw=2,
                        ls="--",
                        label="{:.2f}M SPEC_LYA ".format(
                            1e-6 * d["UNQ_LYA"][sel].max()
                        ),
                    )

        # AR shutdowns
        start_nights, end_nights, _ = get_shutdowns(survey)
        for start_night, end_night in zip(start_nights, end_nights):
            start_mjd = d["MJD"][np.abs(d["THRUNIGHT"] - start_night).argmin()]
            end_mjd = d["MJD"][np.abs(d["THRUNIGHT"] - end_night).argmin()]
            ax.axvspan(start_mjd, end_mjd, color="k", alpha=0.1, zorder=0)

        # AR first create labels for each month
        xticklabels = [20210514 + i * 100 for i in range(8)]
        for year in [2022, 2023, 2024, 2025, 2026]:
            xticklabels += [year * 10000 + 114 + i * 100 for i in range(12)]
        xticklabels = np.array(xticklabels)
        xticks = np.array(
            [
                Time(datetime.strptime("{}".format(night), "%Y%m%d")).mjd
                for night in xticklabels
            ]
        )

        # AR restrict to obs
        ii = np.where(xticks <= d["MJD"].max())[0]
        if xticks[ii].max() < d["MJD"].max():
            ii = np.append(ii, [ii.max() + 1])
        xticklabels, xticks = xticklabels[ii], xticks[ii]
        ax.set_xlim(xticks[0], xticks[-1])

        # AR restrict to ~3 elements
        ii = ii[:: int(ii.size / 5)]
        ax.set_xticks(xticks[ii])
        ax.set_xticklabels(xticklabels[ii].astype(str), rotation=45, ha="right")

        ax.set_title(
            "{} {} tiles up to {}".format(
                d["NTILE"][(isprog) & (d["NAME"] == "ALL")].max(),
                proglab,
                d["THRUNIGHT"][isprog].max(),
            ),
            fontsize=title_fs,
        )
        ax.set_ylabel(
            "Number of unique spectra [million]\nwith DELTACHI2 > {}".format(dchi2min)
        )
        ax.set_ylim(0, None)
        ax.grid()
        ax.legend(loc=2)

    plt.savefig(nspecpng, bbox_inches="tight")
    plt.close()


# AR see email Sam Brieden from Feb. 22, 2024
def plot_nspec_summary(specfn, firstnight, lastnight, outpng):
    """
    Make a simplified version of the nspec=f(night) plot.

    Args:
        specfn: full path to the nspec.ecsv file (str)
        firstnight: first night to consider (int)
        lastnight: last night to consider (int)
        outpng: output file name (str)

    Notes:
        Prompted by a request from S. Brieden (Feb. 22, 2024), for a social
            media post about DESI reaching 40M spectra.
    """
    d = Table.read(specfn)
    dchi2min = d.meta["DCHI2MIN"]
    sel = (d["THRUNIGHT"] >= firstnight) & (d["THRUNIGHT"] <= lastnight)
    d = d[sel]
    #
    fig, ax = plt.subplots()
    # all, gal/qso
    sel = (d["PROGRAM"] == "ALL") & (d["NAME"] == "ALL")
    lab = "{:.1f}M galaxies/quasars".format(1e-6 * d["UNQ_GALQSO"][sel].max())
    ax.plot(d["MJD"][sel], 1e-6 * d["UNQ_GALQSO"][sel], color="k", lw=4, label=lab)
    # dark, gal/qso
    sel = (d["PROGRAM"] == "DARK") & (d["NAME"] == "ALL")
    lab = "{:.1f}M galaxies/quasars in Dark Time".format(
        1e-6 * d["UNQ_GALQSO"][sel].max()
    )
    ax.plot(d["MJD"][sel], 1e-6 * d["UNQ_GALQSO"][sel], color="r", lw=1, label=lab)
    # bright, gal/qso
    sel = (d["PROGRAM"] == "BRIGHT") & (d["NAME"] == "ALL")
    lab = "{:.1f}M galaxies/quasars in Bright Time".format(
        1e-6 * d["UNQ_GALQSO"][sel].max()
    )
    ax.plot(d["MJD"][sel], 1e-6 * d["UNQ_GALQSO"][sel], color="b", lw=1, label=lab)
    # all, stars
    sel = (d["PROGRAM"] == "ALL") & (d["NAME"] == "ALL")
    lab = "{:.1f}M stars".format(1e-6 * d["UNQ_STAR"][sel].max())
    ax.plot(d["MJD"][sel], 1e-6 * d["UNQ_STAR"][sel], color="0.5", lw=4, label=lab)

    # AR shutdowns
    start_nights, end_nights, _ = get_shutdowns(survey)
    for start_night, end_night in zip(start_nights, end_nights):
        start_mjd = d["MJD"][np.abs(d["THRUNIGHT"] - start_night).argmin()]
        end_mjd = d["MJD"][np.abs(d["THRUNIGHT"] - end_night).argmin()]
        ax.axvspan(start_mjd, end_mjd, color="k", alpha=0.1, zorder=0)

    # AR first create labels for each month
    xticklabels = [20210514 + i * 100 for i in range(8)]
    for year in [2022, 2023, 2024, 2025, 2026]:
        xticklabels += [year * 10000 + 114 + i * 100 for i in range(12)]
    xticklabels = np.array(xticklabels)
    xticks = np.array(
        [
            Time(datetime.strptime("{}".format(night), "%Y%m%d")).mjd
            for night in xticklabels
        ]
    )

    # AR restrict to obs
    ii = np.where(xticks <= d["MJD"].max())[0]
    if xticks[ii].max() < d["MJD"].max():
        ii = np.append(ii, [ii.max() + 1])
    xticklabels, xticks = xticklabels[ii], xticks[ii]
    ax.set_xlim(xticks[0], xticks[-1])

    # AR restrict to ~3 elements
    ii = ii[:: int(ii.size / 5)]
    ax.set_xticks(xticks[ii])
    ax.set_xticklabels(xticklabels[ii].astype(str), rotation=45, ha="right")

    firstnight = d["THRUNIGHT"].min()
    lastnight = d["THRUNIGHT"].max()
    ax.set_title(
        "DESI survey progress from {}-{:02d}-{:02d} to {}-{:02d}-{:02d}".format(
            firstnight // 10000,
            firstnight % 10000 // 100,
            firstnight % 100,
            lastnight // 10000,
            lastnight % 10000 // 100,
            lastnight % 100,
        )
    )
    ax.set_ylabel("Number of spectra [million]")
    ax.set_ylim(0, None)
    ax.grid()
    ax.legend(loc=2)

    plt.savefig(outpng, bbox_inches="tight")
    plt.close()
