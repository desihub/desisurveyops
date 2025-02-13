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
    d = create_tiles_table(
        tileids, tileras, tiledecs, program, in_desis=np.ones(ntile, dtype=bool)
    )

    return d


def finalize_calibration_target_table(
    d,
    prognum,
    program,
    goaltime=None,
    sbprof=None,
    checker="AR",
):
    """
    Properly format a table for the tertiary use.

    Args:
        d: a Table() array
        prognum: tertiary prognum (int)
        program: "BRIGHT", "DARK", or "BACKUP" (str)
        goaltime (optional, defaults to 180/1000/60 for BRIGHT/DARK/BACKUP): tile goaltime in seconds (int)
        sbprof (optional, defaults to BGS/ELG/PSF for BRIGHT/DARK/BACKUP): tile SBPROF for ETC (str)
        checker (optional, defaults to "AR"): initials of the checker (str)

    Returns:
        d: a Table() array with:
            these columns: TARGETID,CHECKER (and SUPRIORITY if not present)
            clean PMRA, PMDEC, REF_EPOCH values
            these header keywords: EXTNAME,FAPRGRM,OBSCONDS,SBPROF,GOALTIME

    Notes:
        Close to desisurveyops.fba_tertiary_design_io.finalize_target_table()
            but as yaml files were not used for calibration programs,
            need to have this version.
        20250212: BACKUP enabled (used for for non-calibration tiles)
        20250212: optional goaltime/sbprof
    """

    assert program in ["DARK", "BRIGHT", "BACKUP"]

    # AR goaltime
    if goaltime is None:
        goaltime = {"DARK": 1000, "BRIGHT": 180, "BACKUP": 60}[program]

    # AR sbprof
    if sbprof is None:
        sbprof = {"DARK": "ELG", "BRIGHT": "BGS", "BACKUP": "PSF"}[program]

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
    d.meta["SBPROF"] = sbprof
    d.meta["GOALTIME"] = goaltime

    return d
