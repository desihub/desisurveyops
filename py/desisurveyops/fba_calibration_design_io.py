#!/usr/bin/env python

import os
from glob import glob
import numpy as np
import fitsio
from astropy.table import Table
import healpy as hp
from desimodel.focalplane.geometry import get_tile_radius_deg
from desimodel.footprint import tiles2pix, is_point_in_desi
from desitarget.io import read_targets_in_hp
from desitarget.targets import encode_targetid
from desitarget.targetmask import desi_mask, bgs_mask, mws_mask, scnd_mask
from fiberassign.fba_tertiary_io import get_priofn, get_targfn
from fiberassign.utils import Logger
from desisurveyops.fba_tertiary_design_io import (
    create_tiles_table,
    format_pmradec_refepoch,
)

# AR https://desi.lbl.gov/trac/wiki/SurveyOps/CalibrationFields

log = Logger.get()

# AR settings
def get_calibration_settings(prognum):
    """
    Retrieves the calibration program properties for a given prognum.

    Args:
        prognum: prognum (int)

    Returns:
        program: "DARK" or "BRIGHT" (str)
        field_ra: R.A. center of the calibration field (float)
        field_dec: Dec. center of the calibration field (float)
        tileid_start: first TILEID of the calibration program (int)
        tileid_end: last TILEID of the calibration program (int)

    Notes:
        Source: https://desi.lbl.gov/trac/wiki/SurveyOps/CalibrationFields
        So far the allowed prognums are 5,6,7,8,9,10,11,12.
    """
    # AR XMMLSS DARK
    if prognum == 5:
        program, field_ra, field_dec, tileid_start, tileid_end = (
            "DARK",
            35.7,
            -4.75,
            83000,
            83019,
        )
    # AR XMMLSS BRIGHT
    elif prognum == 6:
        program, field_ra, field_dec, tileid_start, tileid_end = (
            "BRIGHT",
            35.7,
            -4.75,
            83020,
            83039,
        )
    # AR COSMOS DARK
    elif prognum == 7:
        program, field_ra, field_dec, tileid_start, tileid_end = (
            "DARK",
            150.1,
            2.182,
            83040,
            83059,
        )
    # AR COSMOS BRIGHT
    elif prognum == 8:
        program, field_ra, field_dec, tileid_start, tileid_end = (
            "BRIGHT",
            150.1,
            2.182,
            83060,
            83079,
        )
    # AR MBHB1 DARK
    elif prognum == 9:
        program, field_ra, field_dec, tileid_start, tileid_end = (
            "DARK",
            203.5,
            17.5,
            83080,
            83099,
        )
    # AR MBHB1 BRIGHT
    elif prognum == 10:
        program, field_ra, field_dec, tileid_start, tileid_end = (
            "BRIGHT",
            203.5,
            17.5,
            83100,
            83119,
        )
    # AR GAMA15 DARK
    elif prognum == 11:
        program, field_ra, field_dec, tileid_start, tileid_end = (
            "DARK",
            215.7,
            -0.7,
            83120,
            83139,
        )
    # AR GAMA15 BRIGHT
    elif prognum == 12:
        program, field_ra, field_dec, tileid_start, tileid_end = (
            "BRIGHT",
            215.7,
            -0.7,
            83140,
            83159,
        )
    else:
        msg = "prognum = {} not recognized; exiting".format(prognum)
        log.error(msg)
        raise ValueError(msg)
    return program, field_ra, field_dec, tileid_start, tileid_end


# AR
def get_calibration_tile_centers(field_ra, field_dec, ntile):
    """
    Get the tile centers for a calibration program.

    Args:
        field_ra: R.A. center of the calibration field (float)
        field_dec: Dec. center of the calibration field (float)
        ntile: number of tiles (int)

    Returns:
        ras: R.A. of the tiles (np.array of floats)
        decs: Dec. of the tiles (np.array of floats)

    Notes:
        Source: https://desi.lbl.gov/trac/wiki/SurveyOps/CalibrationFields
    """
    # AR offsets in degrees
    offset_ras = np.array([0, 0.048, 0, -0.048, 0, 0, 1.000, 0, 0])
    offset_decs = np.array([0, 0, 1.000, 0, -1.000, 0.048, 0, -0.048, -1.000])
    ras = field_ra + offset_ras / np.cos(np.radians(field_dec))
    decs = field_dec + offset_decs
    ras, decs = np.tile(ras, 10000)[:ntile], np.tile(decs, 10000)[:ntile]
    ras[ras >= 360] -= 360
    return ras, decs


def get_calibration_tiles(program, field_ra, field_dec, tileid_start, tileid_end):
    """
    Generate the tiles table for a calibration program.

    Args:
        program: "DARK" or "BRIGHT" (str)
        field_ra: R.A. center of the calibration field (float)
        field_dec: Dec. center of the calibration field (float)
        tileid_start: first TILEID of the calibration program (int)
        tileid_end: last TILEID of the calibration program (int)

    Returns:
        d: a Table() structure properly formatted for the tertiary program,
            created with the desisurveyops.fba_tertiary_design_io.create_tiles_table()
            function (Table() structure)
    """

    ntile = tileid_end - tileid_start + 1
    tileids = np.arange(tileid_start, tileid_end + 1, dtype=int)
    log.info(
        "will define {} tiles (TILEID={:06d}-{:06d})".format(
            ntile, tileid_start, tileid_end
        )
    )

    # AR no ra wrapping needed for the calibration field centers
    tileras, tiledecs = get_calibration_tile_centers(field_ra, field_dec, ntile)

    # AR create + write table
    # AR for backwards-compatibility reasons, we set here
    # AR    IN_DESI=True (instead of the usual IN_DESI=1
    d = create_tiles_table(tileids, tileras, tiledecs, program, in_desis=np.ones(ntile, dtype=bool))

    return d


def get_main_primary_priorities(program):
    """
    Retrieve the simplified list of target classes and associated PRIORITY_INIT values
        from the DESI Main primary targets.

    Args:
        program: "DARK" or "BRIGHT" (str)


    Returns:
        names: (tertiary-adapted) names of the target classes (np.array() of str)
        initprios: PRIORITY_INIT values for names (np.array() of int)
        calib_or_nonstds: is the target class from calibration and other non-standard targets, like sky (np.array() of bool)

    Notes:
        The approach is to define a TERTIARY_TARGET class for each DESI Main
            primary target class.
        We parse the following masks: desi_mask, mws_mask, bgs_mask, scnd_mask.
        The names are built as "{prefix}_{target_class}", where prefix is desi, mws, bgs, scnd
        The observation scheme for the calib_or_nonstds=True is non-standard,
            so one may want to treat them in a custom way in some tertiary designs
            (note that the standard stars are independently picked in fba_launch anyway).
        The list of calib_or_nonstds=True DARK targets is:
            DESI_SKY,DESI_STD_FAINT,DESI_STD_WD,DESI_SUPP_SKY,DESI_NO_TARGET,DESI_NEAR_BRIGHT_OBJECT,DESI_SCND_ANY
            MWS_GAIA_STD_FAINT,MWS_GAIA_STD_WD
        The list of calib_or_nonstds=True BRIGHT targets is:
            DESI_SKY,DESI_STD_WD,DESI_STD_BRIGHT,DESI_SUPP_SKY,DESI_NO_TARGET,DESI_NEAR_BRIGHT_OBJECT,DESI_SCND_ANY
            MWS_GAIA_STD_WD,MWS_GAIA_STD_BRIGHT
    """

    # AR we discard some names
    black_names = {}
    for prefix, mask in zip(
        ["DESI", "MWS", "BGS", "SCND"],
        [desi_mask, mws_mask, bgs_mask, scnd_mask],
    ):
        black_names[prefix] = [
            key for key in mask.names() if key[-5:] in ["NORTH", "SOUTH"]
        ]

    # AR loop on masks
    names, initprios, calib_or_nonstds = [], [], []
    for prefix, mask in zip(
        ["DESI", "MWS", "BGS", "SCND"],
        [desi_mask, mws_mask, bgs_mask, scnd_mask],
    ):
        mask_names = [name for name in mask.names() if name not in black_names[prefix]]
        for name in mask_names:
            if program in mask[name].obsconditions:
                names.append("{}_{}".format(prefix, name))
                initprios.append(mask[name].priorities["UNOBS"] if "UNOBS" in mask[name].priorities else None)
                calib_or_nonstds.append("UNOBS" not in mask[name].priorities)

    names = np.array(names)
    initprios = np.array(initprios)
    calib_or_nonstds = np.array(calib_or_nonstds)

    return names, initprios, calib_or_nonstds


def get_main_primary_targets(
    program,
    field_ras,
    field_decs,
    radius=None,
    dtver="1.1.1",
    remove_stds=False,
):
    """
    Read the DESI Main primary targets inside a set of field positions.

    Args:
        program: "DARK" or "BRIGHT" (str)
        field_ras: R.A. center of the calibration field (float or np.array() of floats)
        field_decs: Dec. center of the calibration field (float or np.array() of floats)
        radius (optional, defaults to the DESI tile radius): radius in deg. to query around
            the field centers (float)
        dtver (optional, defaults to 1.1.1): main desitarget catalog version (str)
        remove_stds (optional, defaults to False): remove STD_{BRIGHT,FAINT} targets (bool)

    Returns:
        d: a Table() structure with the regular desitarget.io functions formatting.

    Notes:
        There is no high-level desitarget.io routines doing that, because we want to allow
            the possibility to query several tiles with a custom radius.
        The remove_stds argument is in case one wants to remove standard stars,
            as those can be independently picked up by fba_launch.
    """

    # AR default to desi tile radius
    if radius is None:
        radius = get_tile_radius_deg()

    # AR desitarget folder
    hpdir = os.path.join(
        os.getenv("DESI_TARGET"),
        "catalogs",
        "dr9",
        dtver,
        "targets",
        "main",
        "resolve",
        program.lower(),
    )

    # AR get the file nside
    fn = sorted(
        glob(os.path.join(hpdir, "targets-{}-hp-*fits".format(program.lower())))
    )[0]
    nside = fitsio.read_header(fn, 1)["FILENSID"]

    # AR get the list of pixels overlapping the tiles
    tiles = Table()
    if isinstance(field_ras, float):
        tiles["RA"], tiles["DEC"] = [field_ras], [field_decs]
    else:
        tiles["RA"], tiles["DEC"] = field_ras, field_decs
    pixs = tiles2pix(nside, tiles=tiles, radius=radius)

    # AR read the targets in these healpix pixels
    d = Table(read_targets_in_hp(hpdir, nside, pixs, quick=True))
    d.meta["TARG"] = hpdir

    # AR cut on actual radius
    sel = is_point_in_desi(tiles, d["RA"], d["DEC"], radius=radius)
    d = d[sel]

    # AR remove STD_BRIGHT,STD_FAINT?
    # AR    as those will also be included in the fba_launch
    # AR    call with the --targ_std_only argument
    if remove_stds:
        names = ["STD_BRIGHT", "STD_FAINT"]
        reject = np.zeros(len(d), dtype=bool)
        for name in names:
            reject |= (d["DESI_TARGET"] & desi_mask[name]) > 0
        print("removing {} names={} targets".format(reject.sum(), ",".join(names)))
        d = d[~reject]

    return d


def get_main_primary_targets_names(
    targ,
    program,
    tertiary_targets=None,
    initprios=None,
    do_ignore_gcb=False,
):
    """
    Get the TERTIARY_TARGET values for the DESI Main primary targets.

    Args:
        targ: Table() array with the DESI Main primary targets;
            typically output of get_main_primary_targets() (Table() array)
        program: "DARK" or "BRIGHT" (str)
        tertiary_targets (optional, defaults to get_main_primary_priorities()):
            (tertiary-adapted) names of the target classes (np.array() of str)
        initprios (optional, defaults to get_main_primary_priorities()): PRIORITY_INIT values for names (np.array() of int)
        do_ignore_gcb (optional, defaults to False): ignore the GC_BRIGHT targets for the sanity check; this
            is used for PROGNUM=8 (bool)

    Returns:
        names: the TERTIARY_TARGET values (np.array() of str)
    """

    # AR tertiary names + priorities
    if tertiary_targets is not None:
        assert initprios is not None
    else:
        assert initprios is None
        tertiary_targets, initprios, calib_or_nonstds = get_main_primary_priorities(program)
    sel = np.array([_ is not None for _ in initprios])
    tertiary_targets, initprios = tertiary_targets[sel], initprios[sel]

    # AR TERTIARY_TARGET
    # AR loop on np.unique(tertiary_targets), for backwards-reproducibility
    dtype = "|S{}".format(np.max([len(x) for x in tertiary_targets]))
    names = np.array(["-" for i in range(len(targ))], dtype=dtype)
    myprios = -99 + np.zeros(len(targ))
    ii = tertiary_targets.argsort()
    for tertiary_target, initprio in zip(tertiary_targets[ii], initprios[ii]):
        if tertiary_target[:5] == "DESI_":
            name, dtkey, mask = tertiary_target[5:], "DESI_TARGET", desi_mask
        if tertiary_target[:4] == "BGS_":
            name, dtkey, mask = tertiary_target[4:], "BGS_TARGET", bgs_mask
        if tertiary_target[:4] == "MWS_":
            name, dtkey, mask = tertiary_target[4:], "MWS_TARGET", mws_mask
        if tertiary_target[:5] == "SCND_":
            name, dtkey, mask = tertiary_target[5:], "SCND_TARGET", scnd_mask
        sel = ((targ[dtkey] & mask[name]) > 0) & (myprios < initprio)
        # AR assuming all secondaries have OVERRIDE=False
        # AR i.e. a primary target will keep its properties if it
        # AR also is a secondary
        if dtkey == "SCND_TARGET":
            sel &= targ["DESI_TARGET"] == desi_mask["SCND_ANY"]
        myprios[sel] = initprio
        names[sel] = tertiary_target

    # AR verify that all rows got assigned a TERTIARY_TARGET
    assert (names.astype(str) == "-").sum() == 0

    # AR verify that we set for PRIORITY_INIT the same values as in desitarget
    # AR except for STD_BRIGHT,STD_FAINT
    # AR except for WD_BINARIES_BRIGHT, WD_BINARIES_DARK...
    # AR            where the above assumption is false.. (their scnd priority
    # AR            1998 is higher than their primary_priority, 1400-1500
    # AR            from MWS_BROAD,MWS_MAIN_BLUE..
    # AR except for (6) GC_BRIGHT in COSMOS/BRIGHT (same reason as above)
    myprios = -99 + np.zeros(len(targ))
    for t, p in zip(tertiary_targets, initprios):
        myprios[names == t] = p
    ignore_std = np.in1d(
        names.astype(str), ["DESI_STD_BRIGHT", "DESI_STD_FAINT"]
    )
    log.info(
        "priority_check: ignoring {} DESI_STD_BRIGHT|DESI_STD_FAINT".format(
            ignore_std.sum()
        )
    )
    ignore_wd = (targ["SCND_TARGET"] & scnd_mask["WD_BINARIES_BRIGHT"]) > 0
    ignore_wd |= (targ["SCND_TARGET"] & scnd_mask["WD_BINARIES_DARK"]) > 0
    log.info(
        "priority check: ignoring {} WD_BINARIES_BRIGHT|WD_BINARIES_DARK".format(
            ignore_wd.sum()
        )
    )
    ignore = (ignore_std) | (ignore_wd)
    if do_ignore_gcb:
        ignore_gcb = (targ["SCND_TARGET"] & scnd_mask["GC_BRIGHT"]) > 0
        log.info("priority check: ignoring {} GC_BRIGHT".format(ignore_gcb.sum()))
        ignore |= ignore_gcb
    sel = (myprios != targ["PRIORITY_INIT"]) & (~ignore)

    # AR sanity check
    # AR do not raise an error, as according to the usage it could be ok
    # AR    (because this as been tested only on calibration fields)
    if sel.sum() > 0:
        msg = "the set PRIORITY_INIT differs from the desitarget one for {}/{} rows; please check!".format(
            sel.sum(), len(targ)
        )
        log.warning(msg)

    return names


def finalize_calibration_target_table(
    d,
    prognum,
    program,
    checker="AR",
):
    """
    Properly format a table for the tertiary use.

    Args:
        d: a Table() array

    Returns:
        d: a Table() array with:
            these columns: TARGETID,CHECKER (and SUPRIORITY if not present)
            clean PMRA, PMDEC, REF_EPOCH values
            these header keywords: EXTNAME,FAPRGRM,OBSCONDS,SBPROF,GOALTIME

    Notes:
        Close to desisurveyops.fba_tertiary_design_io.finalize_target_table()
            but as yaml files were not used for calibration programs,
            need to have this version.
    """

    # AR TARGETID, CHECKER
    d["TARGETID"] = encode_targetid(
        release=8888, brickid=prognum, objid=np.arange(len(d))
    )
    d["CHECKER"] = checker

    # AR put TARGETID, CHECKER first, for ~backwards-compatibility
    keys = ["TARGETID", "CHECKER"] + [
        _ for _ in d.colnames if _ not in ["TARGETID", "CHECKER"]
    ]
    d = d[keys]

    # AR pmra, pmdec, ref_epoch
    d = format_pmradec_refepoch(d)

    # AR subpriority
    if "SUBPRIORITY" not in d.colnames:
        d["SUBPRIORITY"] = np.random.uniform(size=len(d))

    # AR header
    d.meta["EXTNAME"] = "TARGETS"
    d.meta["FAPRGRM"] = "tertiary{}".format(prognum)
    d.meta["OBSCONDS"] = program
    assert program in ["BRIGHT", "DARK"]
    d.meta["SBPROF"] = "ELG" if program == "DARK" else "BGS"
    d.meta["GOALTIME"] = 1000.0 if program == "DARK" else 180.0

    return d
