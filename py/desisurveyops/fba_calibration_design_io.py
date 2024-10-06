#!/usr/bin/env python

import os
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
import healpy
from desisurvey.tileqa import lb2uv
from desitarget.io import read_targets_in_cap
from desitarget.targets import encode_targetid
from desitarget.targetmask import desi_mask, bgs_mask, mws_mask, scnd_mask
from fiberassign.fba_tertiary_io import get_priofn, get_targfn
from fiberassign.utils import Logger
from desisurveyops.fba_tertiary_design_io import create_tiles_table, format_pmradec_refepoch

# AR https://desi.lbl.gov/trac/wiki/SurveyOps/CalibrationFields

log = Logger.get()

# AR settings
def get_calibration_settings(prognum):
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
    # AR offsets in degrees
    offset_ras = np.array([0, 0.048, 0, -0.048, 0, 0, 1.000, 0, 0])
    offset_decs = np.array([0, 0, 1.000, 0, -1.000, 0.048, 0, -0.048, -1.000])
    ras = field_ra + offset_ras / np.cos(np.radians(field_dec))
    decs = field_dec + offset_decs
    ras, decs = np.tile(ras, 10000)[:ntile], np.tile(decs, 10000)[:ntile]
    ras[ras >= 360] -= 360
    return ras, decs


def get_calibration_tiles(
    program,
    field_ra,
    field_dec,
    tileid_start,
    tileid_end
):

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
    d = create_tiles_table(tileids, tileras, tiledecs, program)

    return d


def get_calibration_priorities(program):

    # AR keys
    keys = ["TERTIARY_TARGET", "NUMOBS_DONE_MIN", "NUMOBS_DONE_MAX", "PRIORITY"]

    # AR note: STD_BRIGHT,STD_FAINT have no priorities
    # AR       we assign them PRIORITY=3000
    desi_mask["STD_BRIGHT"].priorities["UNOBS"] = 3000
    desi_mask["STD_FAINT"].priorities["UNOBS"] = 3000

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
    myd = {key: [] for key in keys}
    for prefix, mask in zip(
        ["DESI", "MWS", "BGS", "SCND"],
        [desi_mask, mws_mask, bgs_mask, scnd_mask],
    ):
        names = [name for name in mask.names() if name not in black_names[prefix]]
        for name in names:
            if program in mask[name].obsconditions:
                if "UNOBS" in mask[name].priorities:
                    if mask[name].priorities["UNOBS"] == 0:
                        myd["TERTIARY_TARGET"].append("{}_{}".format(prefix, name))
                        myd["NUMOBS_DONE_MIN"].append(0)
                        myd["NUMOBS_DONE_MAX"].append(99)
                        myd["PRIORITY"].append(mask[name].priorities["UNOBS"])
                    else:
                        #
                        myd["TERTIARY_TARGET"].append("{}_{}".format(prefix, name))
                        myd["NUMOBS_DONE_MIN"].append(0)
                        myd["NUMOBS_DONE_MAX"].append(0)
                        myd["PRIORITY"].append(mask[name].priorities["UNOBS"])
                        #
                        myd["TERTIARY_TARGET"].append("{}_{}".format(prefix, name))
                        myd["NUMOBS_DONE_MIN"].append(1)
                        myd["NUMOBS_DONE_MAX"].append(98)
                        myd["PRIORITY"].append(5000 + mask[name].priorities["UNOBS"])
                        #
                        myd["TERTIARY_TARGET"].append("{}_{}".format(prefix, name))
                        myd["NUMOBS_DONE_MIN"].append(99)
                        myd["NUMOBS_DONE_MAX"].append(99)
                        myd["PRIORITY"].append(5000 + mask[name].priorities["UNOBS"])

    # AR create table
    d = Table()
    for key in keys:
        d[key] = myd[key]

    # AR assert there are no duplicated names
    d0 = d[d["NUMOBS_DONE_MIN"] == 0]
    assert len(d0) == np.unique(d0["TERTIARY_TARGET"]).size

    return d


def get_main_primary_targets(
    program,
    field_ra,
    field_dec,
    radius,
    do_ignore_gcb=False,
    priofn=None,
    dtver="1.1.1",
):

    # AR tertiary priorities
    if priofn is not None:
        prios = Table.read(priofn)
    else:
        prios = get_calibration_priorities(program)
    # AR cutting on NUMOBS_DONE_MIN=0
    prios = prios[prios["NUMOBS_DONE_MIN"] == 0]

    # AR desitarget targets
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
    radecrad = [field_ra, field_dec, radius]
    targ = Table(read_targets_in_cap(hpdir, radecrad, quick=True))

    # AR remove STD_BRIGHT,STD_FAINT
    # AR    as those will also be included in the fba_launch
    # AR    call with the --targ_std_only argument
    names = ["STD_BRIGHT", "STD_FAINT"]
    reject = np.zeros(len(targ), dtype=bool)
    for name in names:
        reject |= (targ["DESI_TARGET"] & desi_mask[name]) > 0
    print("removing {} names={} targets".format(reject.sum(), ",".join(names)))
    targ = targ[~reject]

    # AR create table
    d = Table()
    d.meta["TARG"] = hpdir

    # AR TERTIARY_TARGET
    dtype = "|S{}".format(np.max([len(x) for x in prios["TERTIARY_TARGET"]]))
    d["TERTIARY_TARGET"] = np.array(["-" for i in range(len(targ))], dtype=dtype)
    myprios = -99 + np.zeros(len(d))
    for tertiary_target in np.unique(prios["TERTIARY_TARGET"]):
        if tertiary_target[:5] == "DESI_":
            name, dtkey, mask = tertiary_target[5:], "DESI_TARGET", desi_mask
        if tertiary_target[:4] == "BGS_":
            name, dtkey, mask = tertiary_target[4:], "BGS_TARGET", bgs_mask
        if tertiary_target[:4] == "MWS_":
            name, dtkey, mask = tertiary_target[4:], "MWS_TARGET", mws_mask
        if tertiary_target[:5] == "SCND_":
            name, dtkey, mask = tertiary_target[5:], "SCND_TARGET", scnd_mask
        prio = prios["PRIORITY"][prios["TERTIARY_TARGET"] == tertiary_target][0]
        sel = ((targ[dtkey] & mask[name]) > 0) & (myprios < prio)
        # AR assuming all secondaries have OVERRIDE=False
        # AR i.e. a primary target will keep its properties if it
        # AR also is a secondary
        if dtkey == "SCND_TARGET":
            sel &= targ["DESI_TARGET"] == desi_mask["SCND_ANY"]
        myprios[sel] = prio
        d["TERTIARY_TARGET"][sel] = tertiary_target

    # AR verify that all rows got assigned a TERTIARY_TARGET
    assert (d["TERTIARY_TARGET"] == "").sum() == 0

    # AR verify that we set for PRIORITY_INIT the same values as in desitarget
    # AR except for STD_BRIGHT,STD_FAINT
    # AR except for WD_BINARIES_BRIGHT, WD_BINARIES_DARK...
    # AR            where the above assumption is false.. (their scnd priority
    # AR            1998 is higher than their primary_priority, 1400-1500
    # AR            from MWS_BROAD,MWS_MAIN_BLUE..
    # AR except for (6) GC_BRIGHT in COSMOS/BRIGHT (same reason as above)
    myprios = -99 + np.zeros(len(d))
    for t, p in zip(prios["TERTIARY_TARGET"], prios["PRIORITY"]):
        myprios[d["TERTIARY_TARGET"] == t] = p
    ignore_std = np.in1d(
        d["TERTIARY_TARGET"].astype(str), ["DESI_STD_BRIGHT", "DESI_STD_FAINT"]
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
    assert sel.sum() == 0

    # AR other columns
    for key in ["RA", "DEC", "PMRA", "PMDEC", "REF_EPOCH", "SUBPRIORITY"]:
        d[key] = targ[key]

    # AR rename with ORIG_ prefix some columns to keep the original information
    for key in [
        "TARGETID",
        "DESI_TARGET",
        "BGS_TARGET",
        "MWS_TARGET",
        "SCND_TARGET",
        "PRIORITY_INIT",
    ]:
        d["ORIG_{}".format(key)] = targ[key]

    return d


def finalize_calibration_target_table(
    d,
    prognum,
    program,
    checker="AR",
):

    # AR TARGETID, CHECKER
    d["TARGETID"] = encode_targetid(
        release=8888, brickid=prognum, objid=np.arange(len(d))
    )
    d["CHECKER"] = checker

    # AR put TARGETID, CHECKER first, for ~backwards-compatibility
    keys = ["TARGETID", "CHECKER"] + [_ for _ in d.colnames if _ not in ["TARGETID", "CHECKER"]]
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
