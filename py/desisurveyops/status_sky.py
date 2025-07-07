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
import healpy as hp

# AR astropy
from astropy.table import Table
from astropy.time import Time, TimezoneInfo
from astropy.coordinates import EarthLocation, SkyCoord
from astropy import units as u

# AR desisurveyops
from desisurveyops.status_utils import (
    get_filename,
    get_history_tilesfn,
    get_fns,
    get_obsdone_tiles,
    get_mjd,
    get_moon_radecphase,
    get_expfacs,
    create_mp4,
)

# AR desimodel
from desimodel.footprint import tiles2pix
from desimodel.focalplane.geometry import get_tile_radius_deg

# AR desitarget
from desitarget.geomask import match

# AR desispec
from desispec.tile_qa_plot import get_quantz_cmap

# AR desiutil
from desiutil.log import get_logger
from desiutil.plots import init_sky

# AR matplotlib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

log = get_logger()

nest = True  # for tiles2pix

_skygoal_dict = {}


def process_skymap(
    outdir,
    survey,
    specprod,
    programs,
    npassmaxs,
    program_strs,
    numproc,
    recompute=False,
):
    """
    Wrapper function to generate the coverage plots
        (individual maps, movie, completeness per ra or lst slice).

    Args:
        outdir: output folder (str)
        survey: survey name (str)
        specprod: spectroscopic production (e.g. daily) (str)
        programs: list of programs (str)
        npassmaxs: list of npassmaxs (str)
        program_strs: list of program_strs (str)
        numproc: number of parallel processes to run (int)
        recompute (optional, defaults to False): if True recompute all maps;
            if False, only compute missing maps (bool)

    Notes:
        For (programs, npassmaxs, program_strs), see desisurveyops.sky_utils.get_programs_npassmaxs().
        Files will be in {outdir}/{program_str}/.
        Usually use specprod=daily.
    """

    log.info(
        "{}\tBEGIN process_skymap".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    start = time()

    fns = get_fns(survey=survey)
    obs_tiles, obs_nights, obs_progs, done_tiles = get_obsdone_tiles(survey, specprod)

    prog_obs_nights = {}
    prog_done_night = {}
    myargs = []

    global _skygoal_dict

    max_numproc = 128

    for program, npassmax, program_str in zip(programs, npassmaxs, program_strs):

        # AR nights for this program
        sel = obs_progs == program
        if npassmax is not None:
            fn = fns["ops"]["tiles"]
            t = Table.read(fn)
            t = t[t["PASS"] < npassmax]
            sel &= np.in1d(obs_tiles, t["TILEID"])
        log.info("{}\tfound {} observed tiles".format(program_str, sel.sum()))

        # AR handle e.g. bright1b which does not exist yet
        if sel.sum() == 0:
            continue

        # AR obs. nights for that program_str
        prog_obs_nights[program_str] = np.unique(obs_nights[sel]).tolist()

        # AR list nights to process
        # AR "done" case for bright/bright1b/dark/dark1b only
        cases = ["obs"]
        nightss = [prog_obs_nights[program_str]]
        if program in ["BRIGHT", "BRIGHT1B", "DARK", "DARK1B"]:
            sel &= np.in1d(obs_tiles, done_tiles)
            # AR handle e.g. dark1b which does not have done tiles yet
            if sel.sum() > 0:
                prog_done_night[program_str] = np.unique(obs_nights[sel])[-1]
                cases += ["done"]
                nightss += [[prog_done_night[program_str]]]

        # AR update myargs
        skygoal_nights = []
        for case, nights in zip(cases, nightss):
            for night in nights:
                for quant in ["ntile", "fraccov"]:
                    outpng = get_filename(
                        outdir,
                        survey,
                        "skymap",
                        program_str=program_str,
                        case=case,
                        quant=quant,
                        night=night,
                        ext="png",
                    )
                    if (not os.path.isfile(outpng)) | (recompute):
                        myargs.append(
                            (
                                outpng,
                                survey,
                                specprod,
                                program,
                                npassmax,
                                program_str,
                                case,
                                quant,
                                night,
                            )
                        )
                        skygoal_nights.append(night)

        # AR compute cached _skygoal?
        skygoal_nights = np.unique(skygoal_nights)
        skygoal_tilesfns = np.unique(
            [get_history_tilesfn(survey, opsnight=night) for night in skygoal_nights]
        )
        log.info("{}\tskygoal_nights: {}".format(program_str, skygoal_nights))
        log.info("{}\tskygoal_tilesfns: {}".format(program_str, skygoal_tilesfns))
        for skygoal_tilesfn in skygoal_tilesfns:
            _ = get_skygoal(skygoal_tilesfn, program, npassmax=npassmax)

    # AR process, if any
    if len(myargs) > 0:
        log.info(
            "{}\tGenerate {} per-night skymap plots for {}".format(
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                len(myargs),
                ",".join(program_strs),
            )
        )
        pool = multiprocessing.Pool(processes=np.min([numproc, max_numproc]))
        with pool:
            _ = pool.starmap(plot_skymap, myargs)

    # AR return prog_obs_nights, prog_done_night
    # AR mp4 movie for case=obs
    process_skymaps_mp4(
        outdir,
        survey,
        programs,
        npassmaxs,
        program_strs,
        prog_obs_nights,
        prog_done_night,
        numproc,
    )

    # AR sky ra / lst coverage for the latest night
    process_skycompl_ralst(
        outdir,
        survey,
        specprod,
        programs,
        npassmaxs,
        program_strs,
        prog_obs_nights,
        prog_done_night,
        numproc,
    )

    for program, npassmax, program_str in zip(programs, npassmaxs, program_strs):

        # AR latest files with no night in filename
        create_skymaps_nonight_in_filename(
            outdir,
            survey,
            program,
            npassmax,
            program_str,
            prog_obs_nights,
            prog_done_night,
        )

        # AR pending tiles
        if program in ["BRIGHT", "BRIGHT1B", "DARK", "DARK1B"]:
            if program_str not in prog_obs_nights:
                log.warning(
                    "no found observed tiles for {}, not running plot_sky_pending()".format(
                        program_str
                    )
                )
            else:
                plot_sky_pending(
                    outdir, survey, specprod, program, npassmax, program_str
                )

    log.info(
        "{}\tEND process_skymap (took {:.1f}s)".format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), time() - start
        )
    )


def process_skyseq(
    outdir,
    survey,
    specprod,
    programs,
    npassmaxs,
    program_strs,
    numproc,
    recompute=False,
):
    """
    Generate the per-night skyseq-{night}.png maps.

    Args:
        outdir: output folder (str)
        survey: survey name (str)
        specprod: spectroscopic production (e.g. daily) (str)
        programs: list of programs (str)
        npassmaxs: list of npassmaxs (str)
        program_strs: list of program_strs (str)
        numproc: number of parallel processes to run (int)
        recompute (optional, defaults to False): if True recompute all maps;
            if False, only compute missing maps (bool)
    """
    log.info(
        "{}\tBEGIN process_skyseq".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    start = time()

    # AR obs, donetiles
    obs_tiles, obs_nights, obs_progs, done_tiles = get_obsdone_tiles(survey, specprod)

    # AR all nights
    nights = np.unique(obs_nights).tolist()
    myargs = []
    for night in nights:
        outpng = get_filename(outdir, survey, "skyseq", night=night, ext="png")
        if (not os.path.isfile(outpng)) | (recompute):
            myargs.append((outpng, survey, specprod, night))
            log.info("Generate {}".format(outpng))

    # AR process?
    if len(myargs) > 0:
        log.info(
            "{}\tLaunch {} per-night skyseq plots".format(
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"), len(myargs)
            )
        )
        pool = multiprocessing.Pool(processes=numproc)
        with pool:
            _ = pool.starmap(plot_skyseq, myargs)

    # AR latest files with no night in filename
    night = nights[-1]
    cmd = "cp {} {}".format(
        get_filename(outdir, survey, "skyseq", night=night, ext="png"),
        get_filename(outdir, survey, "skyseq", night=None, ext="png"),
    )
    log.info(cmd)
    os.system(cmd)

    log.info(
        "{}\tEND process_skyseq (took {:.1f}s)".format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), time() - start
        )
    )


def custom_plot_sky_circles(ax, ras, decs, field_of_view, **kwargs):
    """
    Utility function to plot circle in the Mollweide projections.

    Args:
        ax: the ax object
        ras: the R.A. center of the circles (float)
        decs: the Dec. center of the circles (float)
        field_of_view: diameter of the circles (in proj. deg.) (float)

    Notes:
        Similar to desiutil.plots.plot_sky_circles but with propagating **kwargs.
        ax should be created with desiutil.plots.init_sky().
    """
    if isinstance(ras, int) | isinstance(ras, float):
        ras, decs = np.array([ras]), np.array([decs])
    proj_edge = ax._ra_center - 180
    while proj_edge < 0:
        proj_edge += 360

    angs = np.linspace(2 * np.pi, 0, 101)
    for ra, dec in zip(ras, decs):
        ras = ra + 0.5 * field_of_view / np.cos(np.radians(dec)) * np.cos(angs)
        decs = dec + 0.5 * field_of_view * np.sin(angs)
        for sel in [ras > proj_edge, ras <= proj_edge]:
            if sel.sum() > 0:
                ax.fill(
                    ax.projection_ra(ras[sel]), ax.projection_dec(decs[sel]), **kwargs
                )


def custom_plot_sky_line(ax, ras, decs, **kwargs):
    """
    Plot a line in a Mollweide projection.

    Args:
        ax: the ax object
        ras: the R.A. values to plot (float)
        decs: the Dec. values to plot (float)

    Notes:
        ax should be created with desiutil.plots.init_sky().
        Try to plot in the right order... Approach a bit hacky, but worked so far.
    """
    if isinstance(ras, int) | isinstance(ras, float):
        ras, decs = np.array([ras]), np.array([decs])

    proj_edge = ax._ra_center - 180
    while proj_edge < 0:
        proj_edge += 360

    npts = 10
    for i in range(len(ras) - 1):
        ra0_i, ra1_i = ras[i], ras[i + 1]
        if np.abs(ra0_i - ra1_i) > 180:
            if np.abs(ra0_i - 360 - ra1_i) < 180:
                ra0_i -= 360
            else:
                ra1_i -= 360
        ras_i = np.linspace(ra0_i, ra1_i, npts)
        decs_i = np.linspace(decs[i], decs[i + 1], npts)
        for sel in [ras_i > proj_edge, ras_i < proj_edge]:
            if sel.sum() > 0:
                ax.plot(
                    ax.projection_ra(ras_i[sel]),
                    ax.projection_dec(decs_i[sel]),
                    **kwargs,
                )


# AR DES
def plot_des(
    ax,
    **kwargs,
):
    """
    Plot the DES footprint.

    Args:
        ax: the ax object.
    """
    fn = get_fns()["desfoot"]
    ras, decs = np.loadtxt(fn, unpack=True)
    ax.plot(ax.projection_ra(ras), ax.projection_dec(decs), **kwargs)


# AR galactic, ecliptic plane
def plot_gp_ep(ax, frame, npt=1000, **kwargs):
    """
    Plot the Galactic or the Ecliptic plane.

    Args:
        ax: the ax object.
        frame: "galactic" or "barycentricmeanecliptic" (str)
        npt (optiona, defaults to 1000): number of points used for the plotting (int)
    """
    cs = SkyCoord(
        np.linspace(0, 360, npt) * u.degree,
        np.zeros(npt) * u.degree,
        frame=frame,
    )
    ras, decs = cs.icrs.ra.degree, cs.icrs.dec.degree
    ii = ax.projection_ra(ras).argsort()
    _ = ax.plot(ax.projection_ra(ras[ii]), ax.projection_dec(decs[ii]), **kwargs)


def create_skygoal(tilesfn, program, outfn=None, nside=1024, npassmax=None):
    """
    Create a table base on a healpix map with, for each healpix pixel:
    - which IN_DESI=True Main tile overlaps it
    - number of IN_DESI=True Main tiles overlapping it
    - per-tile and average EXPFAC

    Args:
        tilesfn: path to the tiles-{survey}.ecsv file (str)
        program: "BACKUP", "BRIGHT{1B}", or "DARK{1B}" (str)
        outfn (optional, defaults to None): if set, write the table to outfn (str)
        nside (optional, defaults to 1024): healpix nside (int)
        npassmax (optional, defaults to None): if set, restrict to npassmax (PASS<npassmax) (int)
    """
    log.info(
        "(outfn, tilesfn, program, npassmax)\t= ({}, {}, {}, {})".format(
            outfn, tilesfn, program, npassmax
        )
    )

    # AR prepare healpix file
    d = Table()
    hdr = fitsio.FITSHDR()
    hdr["HPXNSIDE"], hdr["HPXNEST"] = nside, nest
    d.meta = dict(hdr)
    npix = hp.nside2npix(nside)
    d["HPXPIXEL"] = np.arange(npix, dtype=int)
    d["RA"], d["DEC"] = hp.pix2ang(nside, np.arange(npix), nest=nest, lonlat=True)

    # AR tiles file
    t = Table.read(tilesfn)
    sel = t["PROGRAM"] == program
    sel &= t["IN_DESI"]
    if npassmax is not None:
        sel &= t["PASS"] < npassmax
    t = t[sel]
    if npassmax is not None:
        log.info(
            "found {} tiles with PROGRAM = {} and PASS < {}".format(
                len(t), program, npassmax
            )
        )
    else:
        log.info("found {} tiles with PROGRAM = {}".format(len(t), program))

    # AR handle e.g. BRIGHT1B which does not exist yet
    if len(t) == 0:
        log.warning("no tiles found for PROGRAM = {}, no file created".format(program))
        return None

    passids = np.unique(t["PASS"])
    # AR for backup/bright/dark, the n-first passes where in the program
    # AR but for dark1b, it s different: so far only passes 7 and 8
    # AR   but we will add the extension with pass=0,..,6
    # AR   so we prepare for that
    # npass = len(passes) # deprecated
    npass = 1 + passids.max()
    # listing the area covered by 1, 2, ..., npass passids
    d["TILEIDS"], d["EXPFACS"] = (
        np.zeros((npix, npass), dtype=int),
        np.zeros((npix, npass)),
    )
    for i in range(len(t)):
        if i % 1000 == 0:
            log.info("{}\t{}/{}".format(program, i, len(t) - 1))
        ipixs = tiles2pix(nside, tiles=Table(t[i]))
        d["TILEIDS"][ipixs, t["PASS"][i]] = t["TILEID"][i]
        d["EXPFACS"][ipixs, t["PASS"][i]] = get_expfacs(
            np.array([t["DEC"][i]]), np.array([t["EBV_MED"][i]])
        )
    d["NPASS"] = (d["TILEIDS"] != 0).sum(axis=1)
    d["EXPFAC_MEAN"] = d["EXPFACS"].sum(axis=1)
    sel = d["NPASS"] > 0
    d["EXPFAC_MEAN"][sel] /= d["NPASS"][sel]

    if outfn is not None:
        d.write(outfn, overwrite=True)
    else:
        return d


def get_skygoal(tilesfn, program, npassmax=None):
    """
    Get a table base on a healpix map with, for each healpix pixel:
    - which IN_DESI=True Main tile overlaps it
    - number of IN_DESI=True Main tiles overlapping it
    - per-tile and average EXPFAC

    Args:
        tilesfn: path to the tiles-{survey}.ecsv file (str)
        program: "BACKUP", "BRIGHT{1B}", or "DARK{1B}" (str)
        npassmax (optional, defaults to None): if set, restrict to npassmax (PASS<npassmax) (int)
    """

    global _skygoal_dict

    # AR recompute? (it takes ~30s)
    if tilesfn not in _skygoal_dict:
        _skygoal_dict[tilesfn] = {}
    if program not in _skygoal_dict[tilesfn]:
        _skygoal_dict[tilesfn][program] = {}
    if npassmax not in _skygoal_dict[tilesfn][program]:
        _skygoal_dict[tilesfn][program][npassmax] = create_skygoal(
            tilesfn, program, npassmax=npassmax
        )

    return _skygoal_dict[tilesfn][program][npassmax]


def plot_skymap(
    outpng,
    survey,
    specprod,
    program,
    npassmax,
    program_str,
    case,
    quant,
    night,
    tilesfn=None,
    outfits=None,
):
    """
    Create a sky map of the observations up to a given night.

    Args:
        outpng: output file (str)
        survey: survey name (str)
        specprod: spectroscopic production (e.g. "daily") (str)
        program: "BACKUP", "BRIGHT", or "DARK" (str)
        npassmax: if not None, restrict to npassmax (int)
        program_str: program_str name (str)
        case: "obs" or "done" (str)
        quant: "ntile" or "fraccov" (str)
        night: we consider all observations up to that night (included) (int)
        tilesfn (optional, defaults to tiles-{survey}.ecsv in $DESI_SURVEYOPS): path to tiles-{survey}.ecsv (str)
        outfits (optional, defaults to None): if set write the coverage data to a fits file (str)

    Notes:
        For (programs, npassmaxs, program_strs), see desisurveyops.sky_utils.get_programs_npassmaxs().
    """

    log.info("Generate {}".format(outpng))

    assert case in ["obs", "done"]
    assert quant in ["ntile", "fraccov"]

    global _skygoal_dict

    # AR files
    fns = get_fns(survey=survey, specprod=specprod, opsnight=night, program=program)

    # AR tiles for this program
    if tilesfn is None:
        tilesfn = fns["ops"]["tiles"]
    t = Table.read(tilesfn)
    sel = t["PROGRAM"] == program
    sel &= t["IN_DESI"]
    if npassmax is not None:
        sel &= t["PASS"] < npassmax
    t = t[sel]
    passids = np.unique(t["PASS"])
    npass = len(passids)

    # AR obs/done tiles:
    # AR - obs : for this program and with obs_nights <= night
    # AR - done : for this program
    obs_tiles, obs_nights, obs_progs, done_tiles = get_obsdone_tiles(survey, specprod)
    sel = np.in1d(obs_tiles, t["TILEID"])
    sel &= obs_nights <= night
    obs_tiles, obs_nights, obs_progs = obs_tiles[sel], obs_nights[sel], obs_progs[sel]
    sel = np.in1d(done_tiles, t["TILEID"])
    done_tiles = done_tiles[sel]

    # AR exposures
    e = Table.read(fns["spec"]["exps"])

    # AR skygoal (listing the area covered by 1, 2, ..., npass passes)
    # AR we adopt a "dynamic" approach where we compute it from the tilesfn
    # AR to speed up things, we use cache
    d = get_skygoal(tilesfn, program, npassmax=npassmax)
    nside = d.meta["HPXNSIDE"]
    assert d.meta["HPXNEST"] == nest
    # AR to handle overlapping tiles in a single pass
    # AR for BRIGHT1B, first compute the per-tile pixels
    # AR note that tiles2pix() is not an exact solution
    # AR so a tiny fraction may be false positive:
    # AR - for nside=1024, dark pass=0, 0.01% pixels are said to be touched by two tiles
    # AR - that s negligible enough to not go to nside=2048 which would slow things
    #goal_ns, goal_expfacs = d["NPASS"], d["EXPFAC_MEAN"]
    per_tile_ipixs = tiles2pix(nside, tiles=t, per_tile=True)
    ipixs, counts = np.unique(np.hstack(per_tile_ipixs), return_counts=True)
    goal_ns = np.zeros(len(d), dtype=int)
    goal_ns[ipixs] = counts
    goal_expfacs = d["EXPFAC_MEAN"].copy()
    goal_expfacs[ipixs] *= counts

    # AR for colormap:
    # AR    https://stackoverflow.com/questions/14777066/matplotlib-discrete-colorbar
    if quant == "ntile":
        ntilemax = goal_ns.max()
        cmap = get_quantz_cmap(matplotlib.cm.jet, ntilemax + 1, 0, 1)
        cmaplist = [cmap(i) for i in range(cmap.N)]
        cmaplist[0] = ListedColormap(["lightgray"])(0)
        cmap = LinearSegmentedColormap.from_list("Custom cmap", cmaplist, cmap.N)
        cmin, cmax = -0.5, ntilemax + 0.5
        clabel = "Covered by N tiles"
        if ntilemax > 10:
            dn = 5
        else:
            dn = 1
        cticks = np.arange(0, ntilemax + dn, dn, dtype=int)

    if quant == "fraccov":
        cmap = get_quantz_cmap(matplotlib.cm.jet, 11, 0, 1)
        cmaplist = [cmap(i) for i in range(cmap.N)]
        cmaplist[0] = ListedColormap(["lightgray"])(0)
        cmap = LinearSegmentedColormap.from_list("Custom cmap", cmaplist, cmap.N)
        cmin, cmax = 0, 1
        clabel = "Fraction of final coverage"
        cticks = np.arange(0, 1.1, 0.1)

    # AR healpix
    npix = hp.nside2npix(nside)
    pixarea = hp.nside2pixarea(nside, degrees=True)

    # AR number of tiles and summed expfacs
    ns = np.nan + np.zeros(len(d))
    ntiles = 0
    ns[goal_ns > 0] = 0.0
    expfac = 0.
    for i in range(npass):
        sel = t["PASS"] == passids[i]
        if case == "obs":
            sel &= np.in1d(t["TILEID"], obs_tiles)
        else:
            sel &= np.in1d(t["TILEID"], done_tiles)
        if sel.sum() > 0:
            # AR to handle overlapping tiles in a single pass
            # AR for BRIGHT1B, first compute the per-tile pixels
            #ipixs = tiles2pix(nside, tiles=t[sel])
            #ns[ipixs] += 1
            per_tile_ipixs = tiles2pix(nside, tiles=t[sel], per_tile=True)
            ipixs, icounts = np.unique(np.hstack(per_tile_ipixs), return_counts=True)
            ns[ipixs] += icounts
            expfac += (icounts * d["EXPFACS"][ipixs, passids[i]]).sum()
        ntiles += sel.sum()
    # AR fractional coverage
    fracns = np.nan + np.zeros(len(d))
    sel = goal_ns > 0
    fracns[sel] = ns[sel] / goal_ns[sel]
    if quant == "ntile":
        cs = ns
    if quant == "fraccov":
        cs = fracns

    # AR save the coverage map to a fits file?
    # AR TODO: may need to update code to reflect the changes above for bright1b
    if outfits is not None:
        outd = Table()
        outhdr = fitsio.FITSHDR()
        outhdr["HPXNSIDE"], myhdr["HPXNEST"] = nside, nest
        outhdr["NIGHT"] = night
        outd.meta = dict(outhdr)
        # AR first copy all keys (some overwriting after)
        for key in d.dtype.names:
            outd[key] = d[key]
        # AR null TILEIDS and EXPFACS for not-observed tiles
        for i in range(outd["TILEIDS"].shape[1]):
            if case == "obs":
                reject = ~np.in1d(outd["TILEIDS"][:, i], obs_tiles)
            else:
                reject = ~np.in1d(outd["TILEIDS"][:, i], done_tiles)
            outd["TILEIDS"][reject, i] = 0
            outd["EXPFACS"][reject, i] = 0
        outd["NPASS"] = (outd["TILEIDS"] != 0).sum(axis=1)
        outd["EXPFAC_MEAN"] = outd["EXPFACS"].sum(axis=1)
        outsel = outd["NPASS"] > 0
        outd["EXPFAC_MEAN"][sel] /= outd["NPASS"][outsel]
        #
        outd.write(outfits)

    # AR start plotting
    if case == "obs":
        if program == "BACKUP":
            tmpcase = "observed"
        else:
            tmpcase = "completed"
    if case == "done":
        tmpcase = "done"
    title = "{}/{} : {}/{} {} tiles up to {} (={:.0f}%, weighted={:.0f}%)".format(
        survey.capitalize(),
        program_str,
        ntiles,
        len(t),
        tmpcase,
        night,
        100.0 * ntiles / len(t),
        100.0 * expfac / d["EXPFACS"].sum(),
    )
    log.info(title)

    fig = plt.figure(figsize=(10, 5), dpi=300)
    ax = fig.add_subplot(111, projection="mollweide")
    ax = init_sky(galactic_plane_color="none", ecliptic_plane_color="none", ax=ax)
    ax.set_axisbelow(True)
    sel = ns >= 0
    sc = ax.scatter(
        ax.projection_ra(d["RA"][sel]),
        ax.projection_dec(d["DEC"][sel]),
        c=cs[sel],
        facecolors=ns,
        marker=".",
        s=0.1,
        linewidths=0,
        alpha=0.8,
        cmap=cmap,
        vmin=cmin,
        vmax=cmax,
        zorder=0,
    )
    ax.set_title(title)

    # AR DES, galactic, ecliptic plane
    plot_des(ax, c="k", lw=0.5, alpha=1, zorder=1)
    plot_gp_ep(ax, "galactic", c="k", lw=1, alpha=1, zorder=1)
    plot_gp_ep(ax, "barycentricmeanecliptic", c="k", lw=0.25, alpha=1, zorder=1)

    # AR Moon position for exposures taken that night + tiles completed
    if case == "obs":
        sel = np.in1d(e["TILEID"], obs_tiles[obs_nights == night])
        sel &= e["NIGHT"] == night
        log.info(
            "found {} exposures from {} observed on {}".format(
                sel.sum(), program_str, night
            )
        )
        mjds = e["MJD"][sel]
        mjds = np.linspace(mjds.min(), mjds.max(), 10)
        moon_ras, moon_decs, moon_phases = get_moon_radecphase(mjds)
        ax.scatter(
            ax.projection_ra(moon_ras),
            ax.projection_dec(moon_decs),
            c=0.5 + moon_phases / 2,
            s=10 * moon_phases**2,
            cmap=matplotlib.cm.binary,
            vmin=0,
            vmax=1,
        )
        ax.text(
            -0.05,
            0.04,
            "Stats for the {} night:".format(night),
            fontsize=7,
            transform=ax.transAxes,
        )
        ax.text(
            -0.04,
            0.01,
            "Moon illumination: {:.2f}".format(moon_phases.mean()),
            fontsize=7,
            transform=ax.transAxes,
        )

        # AR tiles completed in the last night
        sel = np.in1d(t["TILEID"], obs_tiles[obs_nights == night])
        custom_plot_sky_circles(
            ax,
            t["RA"][sel],
            t["DEC"][sel],
            2 * get_tile_radius_deg(),
            ec="k",
            fc="none",
            lw=1,
        )
        ax.text(
            -0.04,
            -0.02,
            "{} {} tiles {}".format(sel.sum(), program_str, tmpcase),
            fontsize=7,
            transform=ax.transAxes,
        )

    # AR colorbar
    pos = ax.get_position().get_points().flatten()
    cax = fig.add_axes(
        [
            pos[0] + 0.10 * (pos[2] - pos[0]),
            pos[1] + 0.28 * (pos[3] - pos[1]),
            0.4 * (pos[2] - pos[0]),
            0.02,
        ]
    )
    cbar = plt.colorbar(sc, cax=cax, fraction=0.025, orientation="horizontal")
    cbar.set_label(clabel)
    cbar.set_ticks(cticks)

    # AR blank region for text
    ax.fill_between(
        [0.6, 1.0], [0, 0], [0.3, 0.3], color="w", zorder=2, transform=ax.transAxes
    )

    # AR text
    xs = [0.65, 0.74, 0.83, 0.95]
    y, dy = 0.25, -0.04
    if quant == "ntile":
        txts = ["Npass", "Area", "Fraction", "ExpfacFraction"]
    if quant == "fraccov":
        txts = ["FracCov", "Area", "Fraction", "ExpfacFraction"]
    for x, txt in zip(xs, txts):
        ax.text(
            x,
            y,
            txt,
            color="k",
            ha="center",
            fontsize=10,
            fontweight="bold",
            zorder=3,
            transform=ax.transAxes,
        )
    y += dy
    if quant == "ntile":
        for i in range(npass):
            sel = ns >= i + 1
            goal_sel = goal_ns >= i + 1
            area = pixarea * sel.sum()
            frac = sel.sum() / (goal_sel).sum()
            expfrac = goal_expfacs[sel].sum() / goal_expfacs[goal_sel].sum()
            txts = [
                "{}{}".format(r"$\geq$", i + 1),
                "{:.0f}".format(area),
                "{:.2f}".format(frac),
                "{:.2f}".format(expfrac),
            ]
            for x, txt in zip(xs, txts):
                ax.text(
                    x,
                    y,
                    txt,
                    color=cmap(i + 1),
                    ha="center",
                    fontsize=10,
                    zorder=3,
                    transform=ax.transAxes,
                )
            y += dy
    if quant == "fraccov":
        goal_sel = goal_ns > 0
        for val in [0.00, 0.25, 0.50, 0.75, 1.00]:
            if val == 0:
                sel = fracns > val
                txts_0 = "> {:.2f}".format(val)
                color = "k"
            else:
                sel = fracns >= val
                txts_0 = "{}{:.2f}".format(r"$\geq$", val)
                color = cmap(val)
            area = pixarea * sel.sum()
            frac = sel.sum() / goal_sel.sum()
            expfrac = goal_expfacs[sel].sum() / goal_expfacs[goal_sel].sum()
            txts = [
                txts_0,
                "{:.0f}".format(area),
                "{:.2f}".format(frac),
                "{:.2f}".format(expfrac),
            ]
            for x, txt in zip(xs, txts):
                ax.text(
                    x,
                    y,
                    txt,
                    color=color,
                    ha="center",
                    fontsize=10,
                    zorder=3,
                    transform=ax.transAxes,
                )
            y += dy

    plt.savefig(outpng, bbox_inches="tight")
    plt.close()


def create_skymaps_nonight_in_filename(
    outdir,
    survey,
    program,
    npassmax,
    program_str,
    prog_obs_nights,
    prog_done_night,
):
    """
    Utility functions to make copies of the latest skymaps
        to files with no night in the basename.

    Args:
        outdir: output folder (str)
        survey: survey (str)
        program: program name (str)
        npassmax: if not None, restrict to npassmax (int)
        program_str: program_str name (str)

    Notes:
        For (programs, npassmaxs, program_strs), see desisurveyops.sky_utils.get_programs_npassmaxs().
    """

    # AR handle e.g. bright1b which does not exist yet
    if program_str not in prog_obs_nights:
        return None

    cmds = ["cp"]
    cases = ["obs"]
    nights = [prog_obs_nights[program_str][-1]]

    if program in ["BRIGHT", "BRIGHT1B", "DARK", "DARK1B"]:
        if program_str in prog_done_night:
            cmds += ["mv"]
            cases += ["done"]
            nights += [prog_done_night[program_str]]

    for cmd, case, night in zip(cmds, cases, nights):
        for quant in ["ntile", "fraccov"]:
            log.info("{}\t{}\t{}\t{}".format(program_str, cmd, case, night))
            fullcmd = "{} {} {}".format(
                cmd,
                get_filename(
                    outdir,
                    survey,
                    "skymap",
                    program_str=program_str,
                    case=case,
                    night=night,
                    quant=quant,
                    ext="png",
                ),
                get_filename(
                    outdir,
                    survey,
                    "skymap",
                    program_str=program_str,
                    case=case,
                    night=None,
                    quant=quant,
                    ext="png",
                ),
            )
            log.info(fullcmd)
            os.system(fullcmd)


def process_skymaps_mp4(
    outdir,
    survey,
    programs,
    npassmaxs,
    program_strs,
    prog_obs_nights,
    prog_done_night,
    numproc,
):
    """
    Create a movie from all the per-night skymaps.

    Args:
        outdir: output folder (str)
        survey: survey (str)
        programs: list of program names (str)
        npassmax: if not None, restrict to npassmax (int)
        program_str: program_str name (str)
        prog_obs_nights: dictionary with the list of nights with observations for a given program (dict)
        prog_done_night: dictionary with the list of nights with observations completing tiles for a given program (dict)
        numproc: number of parallel processes (int)

    Notes:
        For (programs, npassmaxs, program_strs), see desisurveyops.sky_utils.get_programs_npassmaxs().
    """
    # AR mp4 for case=obs only
    case = "obs"

    myargs = []
    for program, npassmax, program_str in zip(programs, npassmaxs, program_strs):
        for quant in ["ntile", "fraccov"]:
            myargs.append(
                (
                    outdir,
                    survey,
                    program,
                    npassmax,
                    program_str,
                    prog_obs_nights,
                    prog_done_night,
                    case,
                    quant,
                )
            )
    pool = multiprocessing.Pool(processes=numproc)
    with pool:
        _ = pool.starmap(process_skymaps_program_mp4, myargs)


def process_skymaps_program_mp4(
    outdir,
    survey,
    program,
    npassmax,
    program_str,
    prog_obs_nights,
    prog_done_night,
    case,
    quant,
):
    """
    Create a movie from all the per-night skymaps for a given program.

    Args:
        outdir: output folder (str)
        survey: survey (str)
        program: program name (str)
        npassmax: if not None, restrict to npassmax (int)
        program_str: program_str name (str)
        prog_obs_nights: dictionary with the list of nights with observations for a given program (dict)
        prog_done_night: dictionary with the list of nights with observations completing tiles for a given program (dict)
        case: "obs" or "done" (str)
        quant: "ntile" or "fraccov" (str)

    Notes:
        For (programs, npassmaxs, program_strs), see desisurveyops.sky_utils.get_programs_npassmaxs().
    """

    assert case in ["obs", "done"]
    assert quant in ["ntile", "fraccov"]

    # AR handle e.g. bright1b which does not exist yet
    if program_str not in prog_obs_nights:
        return None

    nights = prog_obs_nights[program_str]
    log.info(
        "{}\t{}\t{}".format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            program_str,
            nights,
            case,
            quant,
        )
    )
    fns = [
        get_filename(
            outdir,
            survey,
            "skymap",
            program_str=program_str,
            case=case,
            quant=quant,
            night=night,
            ext="png",
        )
        for night in nights
    ]
    # AR repeating the last image to pause on it...
    # fns += [fns[-1] for i in range(int(0.2 * len(fns)))]
    outmp4 = get_filename(
        outdir,
        survey,
        "skymap",
        program_str=program_str,
        case=case,
        quant=quant,
        ext="mp4",
    )
    log.info(
        "{}\tGenerate {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), outmp4)
    )
    create_mp4(fns, outmp4, duration=15)


# AR wrap up for sky ra / lst completeness for the latest night
def process_skycompl_ralst(
    outdir,
    survey,
    specprod,
    programs,
    npassmaxs,
    program_strs,
    prog_obs_nights,
    prog_done_night,
    numproc,
):
    """
    Create sky completeness maps per ra and lst slices.

    Args:
        outdir: output folder (str)
        survey: survey (str)
        specprod: spectroscopic production (e.g. daily) (str)
        programs: list of program names (str)
        npassmax: if not None, restrict to npassmax (int)
        program_str: program_str name (str)
        prog_obs_nights: dictionary with the list of nights with observations for a given program (dict)
        prog_done_night: dictionary with the list of nights with observations completing tiles for a given program (dict)
        numproc: number of parallel processes (int)

    Notes:
        For (programs, npassmaxs, program_strs), see desisurveyops.sky_utils.get_programs_npassmaxs().
    """

    myargs = []
    for program, npassmax, program_str in zip(programs, npassmaxs, program_strs):
        # AR to handle e.g. bright1b which does not exist yet
        if program_str not in prog_obs_nights:
            continue
        night = prog_obs_nights[program_str][-1]
        for case in ["complra", "compllst"]:
            outpng = get_filename(
                outdir, survey, "skymap", program_str=program_str, case=case, ext="png"
            )
            myargs.append(
                (
                    outpng,
                    survey,
                    specprod,
                    program,
                    npassmax,
                    program_str,
                    night,
                    case.replace("compl", ""),
                )
            )
    pool = multiprocessing.Pool(processes=numproc)
    with pool:
        _ = pool.starmap(plot_skycompl_ralst, myargs)


def plot_skycompl_ralst(
    outpng,
    survey,
    specprod,
    program,
    npassmax,
    program_str,
    night,
    raquant,
    rabins=np.arange(13) * 30,
    tilesfn=None,
):
    """
    Create sky completeness maps per ra and lst slices.

    Args:
        outpng: output image file (str)
        survey: survey (str)
        specprod: spectroscopic production (e.g. daily) (str)
        program: program name (str)
        npassmax: if not None, restrict to npassmax (int)
        program_str: program_str name (str)
        night: night to plot the completeness at (int)
        raquant: "ra" or "lst" (str)
        rabins (optional, defaults to 30-deg wide bins (np.array of floats)
        prog_obs_nights: dictionary with the list of nights with observations for a given program (dict)
        prog_done_night: dictionary with the list of nights with observations completing tiles for a given program (dict)
        tilesfn (optional, defaults to tiles-{survey}.ecsv in $DESI_SURVEYOPS): path to tiles-{survey}.ecsv (str)

    Notes:
        For (programs, npassmaxs, program_strs), see desisurveyops.sky_utils.get_programs_npassmaxs().
    """
    #
    assert raquant in ["ra", "lst"]

    # AR files
    fns = get_fns(survey=survey, specprod=specprod)

    # AR tiles for this program
    if tilesfn is None:
        tilesfn = fns["ops"]["tiles"]
    t = Table.read(tilesfn)
    sel = t["PROGRAM"] == program
    sel &= t["IN_DESI"]
    if npassmax is not None:
        sel &= t["PASS"] < npassmax
    t = t[sel]

    # AR cap
    t["CAP"] = "NGC"
    cs = SkyCoord(ra=t["RA"] * u.degree, dec=t["DEC"] * u.degree, frame="icrs")
    bs = cs.galactic.b.value
    t["CAP"][bs < 0] = "SGC"

    # AR expfac
    t["EXPFAC"] = get_expfacs(t["DEC"], t["EBV_MED"])

    # AR R.A. or LST?
    if raquant == "ra":
        xlabel = "R.A. [deg]"
        clabel = "Fraction of final coverage per R.A. slice"
    if raquant == "lst":
        t["RA"] += t["DESIGNHA"]
        xlabel = "LST [deg]"
        clabel = "Fraction of final coverage per LST slice"

    # AR obs/done tiles:
    # AR - obs : for this program and with obs_nights <= night
    obs_tiles, obs_nights, _, _ = get_obsdone_tiles(survey, specprod)
    sel = np.in1d(obs_tiles, t["TILEID"])
    sel &= obs_nights <= night
    obs_tiles, obs_nights = obs_tiles[sel], obs_nights[sel]

    t["ISOBS"] = np.isin(t["TILEID"], obs_tiles)

    fracs, expfracs = {}, {}
    for key, sel in zip(
        ["NGC+SGC", "NGC", "SGC"],
        [np.ones(len(t), dtype=bool), t["CAP"] == "NGC", t["CAP"] == "SGC"],
    ):
        selobs = (sel) & (t["ISOBS"])
        fracs[key] = selobs.sum() / sel.sum()
        expfracs[key] = t["EXPFAC"][selobs].sum() / t["EXPFAC"][sel].sum()
    t["FRAC"], t["EXPFRAC"] = 0.0, 0.0
    for i in range(len(rabins) - 1):
        ramin, ramax = rabins[i], rabins[i + 1]
        key = "{}RA{}".format(ramin, ramax)
        sel = (t["RA"] >= ramin) & (t["RA"] < ramax)
        selobs = (sel) & (t["ISOBS"])
        fracs[key] = selobs.sum() / sel.sum()
        expfracs[key] = t["EXPFAC"][selobs].sum() / t["EXPFAC"][sel].sum()
        t["FRAC"][selobs] = fracs[key]
        t["EXPFRAC"][selobs] = expfracs[key]

    title = "{}/{} : {}/{} {} tiles up to {} (={:.0f}%, weighted={:.0f}%)".format(
        survey.capitalize(),
        program_str,
        t["ISOBS"].sum(),
        len(t),
        "observed",
        night,
        100.0 * t["ISOBS"].mean(),
        100.0 * t["EXPFAC"][t["ISOBS"]].sum() / t["EXPFAC"].sum(),
    )
    alpha = 1 / (t["PASS"].max() + 2)
    clim = (0, 1)
    cmap = get_quantz_cmap(matplotlib.cm.jet, 11, 0, 1)

    fig = plt.figure(figsize=(10, 5), dpi=300)
    ax = fig.add_subplot(111, projection="mollweide")
    ax = init_sky(galactic_plane_color="none", ecliptic_plane_color="none", ax=ax)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel)
    ax.set_title(title)

    # AR DES, galactic, ecliptic plane
    plot_des(ax, c="k", lw=0.5, alpha=1, zorder=1)
    plot_gp_ep(ax, "galactic", c="k", lw=1, alpha=1, zorder=1)
    plot_gp_ep(ax, "barycentricmeanecliptic", c="k", lw=0.25, alpha=1, zorder=1)

    # AR blank text region
    ax.fill_between(
        [0.6, 1.0], [0, 0], [0.3, 0.3], color="w", zorder=2, transform=ax.transAxes
    )

    # AR text
    xs = [0.70, 0.83, 0.95]
    y, dy = 0.25, -0.04
    txts = ["Cap", "Fraction", "ExpfacFraction"]
    for x, txt in zip(xs, txts):
        ax.text(
            x,
            y,
            txt,
            color="k",
            ha="center",
            fontsize=10,
            fontweight="bold",
            zorder=3,
            transform=ax.transAxes,
        )
    y += dy
    for key in ["NGC+SGC", "NGC", "SGC"]:
        txts = [
            key,
            "{:.2f}".format(fracs[key]),
            "{:.2f}".format(expfracs[key]),
        ]
        for x, txt in zip(xs, txts):
            ax.text(
                x,
                y,
                txt,
                color="k",
                ha="center",
                fontsize=10,
                zorder=3,
                transform=ax.transAxes,
            )
        y += dy

    for i in range(len(rabins) - 1):
        ramin, ramax = rabins[i], rabins[i + 1]
        key = "{}RA{}".format(ramin, ramax)
        x = ax.projection_ra(np.array([0.5 * (ramin + ramax)]))[0]
        for dec, f in zip([40, 32], [fracs[key], expfracs[key]]):
            y = ax.projection_dec(np.array([dec]))[0]
            ax.annotate(
                "{:.0f}%".format(100 * f),
                (x, y),
                color="k",
                ha="center",
                fontsize=10,
                fontweight="bold",
                zorder=2,
            )

    for x in np.unique(t["EXPFRAC"]).tolist():
        sel = t["EXPFRAC"] == x
        if x == 0:
            c, zorder = "lightgray", 0
        else:
            c, zorder = cmap(x), 1
        custom_plot_sky_circles(
            ax,
            t["RA"][sel],
            t["DEC"][sel],
            2 * get_tile_radius_deg(),
            c=c,
            lw=1,
            alpha=alpha,
            zorder=zorder,
        )
    # AR colorbar
    sc = ax.scatter(None, None, c=0.0, vmin=clim[0], vmax=clim[1], cmap=cmap)
    pos = ax.get_position().get_points().flatten()
    cax = fig.add_axes(
        [
            pos[0] + 0.10 * (pos[2] - pos[0]),
            pos[1] + 0.28 * (pos[3] - pos[1]),
            0.4 * (pos[2] - pos[0]),
            0.02,
        ]
    )
    cbar = plt.colorbar(sc, cax=cax, fraction=0.025, orientation="horizontal")
    cbar.set_label(clabel)
    cbar.set_ticks(np.arange(0, 1.1, 0.1))
    #
    plt.savefig(outpng, bbox_inches="tight")
    plt.close()


def plot_sky_pending(
    outdir,
    survey,
    specprod,
    program,
    npassmax,
    program_str,
):
    """
    Create the sky map of the observations with pending status.

    Args:
        outdir: output folder (str)
        survey: survey (str)
        specprod: spectroscopic production (str)
        program: "BACKUP", "BRIGHT", or "DARK" (str)
        specprod (optional, defaults to "daily"): spectroscopic production (str)
        programs: list of programs (str)
        npassmaxs: list of npassmaxs (str)
        program_strs: list of program_strs (str)

    Notes:
        For (programs, npassmaxs, program_strs), see desisurveyops.sky_utils.get_programs_npassmaxs().
    """

    global _skygoal_dict

    # AR output files
    outecsv = get_filename(
        outdir, survey, "skymap", case="pending", program_str=program_str, ext="ecsv"
    )
    outpng = get_filename(
        outdir, survey, "skymap", case="pending", program_str=program_str, ext="png"
    )

    # AR exposures, tiles
    fns = get_fns(survey=survey, specprod=specprod)
    fn = fns["spec"]["exps"]
    e = Table.read(fn)
    tilesfn = fns["ops"]["tiles"]
    t = Table.read(tilesfn)
    sel = t["PROGRAM"] == program
    sel &= t["IN_DESI"]
    sel &= (t["STATUS"] != "unobs") & (t["STATUS"] != "done")
    if npassmax is not None:
        sel &= t["PASS"] < npassmax
    t = t[sel]

    # AR add some columns from tiles-specstatus
    # AR    note that some tiles may not be in tiles-specstatus yet,
    # AR    for instance if the daily processing failed
    keys = [
        "EXPTIME",
        "EFFTIME_SPEC",
        "OBSSTATUS",
        "ZDONE",
        "LASTNIGHT",
        "QA",
        "USER",
        "OVERRIDE",
        "QANIGHT",
        "ARCHIVEDATE",
    ]
    fn = fns["ops"]["status"]

    # AR modification time
    m_time = os.path.getmtime(fn)
    night = datetime.fromtimestamp(m_time).strftime("%Y%m%d")  # AR modification time
    s = Table.read(fn)
    iid, iis = match(t["TILEID"], s["TILEID"])
    log.info("{}\tmatched {} / {} tiles".format(program_str, iid.size, len(t)))
    log.info(
        "{}\tunmatched tiles: {}".format(
            program_str, t["TILEID"][~np.isin(t["TILEID"], s["TILEID"][iis])].tolist()
        )
    )
    for key in keys:
        t[key] = np.zeros_like(s[key], shape=(len(t)))
        t[key][iid] = s[key][iis]

    # AR remove QA="good" : that happens if tiles-specstatus.ecsv has been updated with recent obs.,
    #       but tiles-main.ecsv has not been updated yet, so does not know
    reject = t["QA"] == "good"
    if reject.sum() > 0:
        log.info(
            "discard the following {} TILEID-LASTNIGHT, as they have QA=good:".format(
                reject.sum()
            )
        )
        log.info(
            t["TILEID", "LASTNIGHT", "OBSSTATUS", "QA", "USER", "ARCHIVEDATE"][reject]
        )
        t = t[~reject]

    # AR sanity checks
    assert np.all(np.isin(t["STATUS"], ["obsstart", "obsend"]))
    assert np.all(np.isin(t["QA"], ["", "none", "bad", "unsure"]))

    # AR adding few infos
    t.meta["MODTIME"] = datetime.fromtimestamp(m_time).strftime("%Y-%m-%d %H:%M:%S")
    for key in ["EXPID", "NIGHT", "EFFTIME_SPEC"]:
        t["EXPIDS_{}".format(key)] = np.zeros(len(t), dtype=object)
    for i in range(len(t)):
        sel = e["TILEID"] == t["TILEID"][i]
        for key in ["EXPID", "NIGHT", "EFFTIME_SPEC"]:
            t["EXPIDS_{}".format(key)][i] = ",".join(e[key][sel].astype(str))
    for key in ["EXPID", "NIGHT", "EFFTIME_SPEC"]:
        t["EXPIDS_{}".format(key)] = t["EXPIDS_{}".format(key)].astype(str)
    log.info("{}\tfound {} tiles".format(program.lower(), len(t)))
    t.write(outecsv, overwrite=True)

    # AR all tiles
    goal = get_skygoal(tilesfn, program, npassmax=npassmax)
    sel = goal["NPASS"] > 0
    goal = goal[sel]

    # AR start plot
    fig = plt.figure(figsize=(10, 5), dpi=300)
    ax = fig.add_subplot(111, projection="mollweide")
    ax = init_sky(galactic_plane_color="none", ecliptic_plane_color="none", ax=ax)
    ax.set_axisbelow(True)
    ax.scatter(
        ax.projection_ra(goal["RA"]),
        ax.projection_dec(goal["DEC"]),
        c="lightgray",
        facecolors="lightgray",
        marker=".",
        s=0.1,
        linewidths=0,
        alpha=0.8,
        zorder=0,
    )
    for status, marker, s in zip(
        ["obsstart", "obsend"],
        ["s", "o"],
        [15, 30],
    ):
        for qa, col in zip(
            ["", "none", "bad", "unsure"],
            ["orange", "g", "r", "b"],
        ):
            sel = (t["STATUS"] == status) & (t["QA"] == qa)
            sc = ax.scatter(
                ax.projection_ra(t["RA"][sel]),
                ax.projection_dec(t["DEC"][sel]),
                marker=marker,
                s=s,
                c=col,
                alpha=0.5,
                label="STATUS={}, QA={} ({} tiles)".format(status, qa, sel.sum()),
            )
    ax.set_title(
        "{}/{} : {} pending tiles as of {}".format(
            survey.capitalize(), program_str, len(t), night
        )
    )
    ax.legend(loc=4, ncol=2, framealpha=1.0)

    # AR DES, galactic, ecliptic plane
    plot_des(ax, c="k", lw=0.5, alpha=1, zorder=1)
    plot_gp_ep(ax, "galactic", c="k", lw=1, alpha=1, zorder=1)
    plot_gp_ep(ax, "barycentricmeanecliptic", c="k", lw=0.25, alpha=1, zorder=1)

    plt.savefig(outpng, bbox_inches="tight")
    plt.close()


def plot_skyseq(
    outpng,
    survey,
    specprod,
    night,
    tilesfn=None,
):
    """
    Create the sky map of the sequence of observations for a given night.

    Args:
        outpng: output file name (str)
        survey: survey (str)
        specprod: spectroscopic production (str)
        night: night of observations (int)
        tilesfn (optional, defaults to tiles-{survey}.ecsv in $DESI_SURVEYOPS): path to tiles-{survey}.ecsv (str)
    """

    # AR files
    fns = get_fns(survey=survey, specprod=specprod)

    # AR tiles
    if tilesfn is None:
        tilesfn = fns["ops"]["tiles"]

    # AR what main programs to display + other
    prognames = np.array(
        [
            "Main/backup",
            "Main/bright",
            "Main/bright1b",
            "Main/dark",
            "Main/dark1b",
            "Other",
        ]
    )
    progmarkers = np.array(["^", "s", "D", "o", "*", "x"])
    progss = np.array([10, 10, 10, 10, 20, 50])

    # AR some display settings
    xtxt, ytxt, dytxt = -0.05, 0.95, -0.05
    fontsize = 6
    ytxt0 = 0.25
    xtxt, ytxt, dxtxt, dytxt = -0.05, ytxt0, 0.22, -0.03
    dxtxt_expid, dxtxt_time, dxtxt_tileid = 0.02, 0.085, 0.125

    # AR KPNO
    kpno = EarthLocation.of_site("Kitt Peak")
    kpno_tz = TimezoneInfo(utc_offset=-7 * u.hour)

    # AR dark tiles for desi footprint
    t = Table.read(tilesfn)
    sel = (t["PROGRAM"] == "DARK") & (t["IN_DESI"])
    t = t[sel]

    # AR exposures
    e = Table.read(fns["spec"]["exps"])
    sel = (e["NIGHT"] == night) & (e["EFFTIME_SPEC"] > 0)
    e = e[sel]
    e["MJD"] = [get_mjd(expid, night) for expid in e["EXPID"]]
    e["MOON_RA"], e["MOON_DEC"], e["MOON_PHASE"] = get_moon_radecphase(e["MJD"])

    title = "Observations from {} ({} exposures, Moon_Illum={:.2f})".format(
        night,
        len(e),
        e["MOON_PHASE"].mean(),
    )

    # AR start plot
    fig = plt.figure(figsize=(10, 5), dpi=300)
    ax = fig.add_subplot(111, projection="mollweide")
    ax = init_sky(galactic_plane_color="none", ecliptic_plane_color="none", ax=ax)
    ax.set_axisbelow(True)

    # AR line with all exposures
    custom_plot_sky_line(
        ax, e["TILERA"], e["TILEDEC"], color="k", ls="--", lw=1, zorder=1
    )

    # AR DESI/Main footprint
    ax.scatter(
        ax.projection_ra(t["RA"]),
        ax.projection_dec(t["DEC"]),
        c="k",
        alpha=0.01,
        zorder=0,
    )

    # AR loop on science exposures
    jj_legend = []
    for i in range(len(e)):

        # AR what symbol + color to use
        col = plt.rcParams["axes.prop_cycle"].by_key()["color"][i % 5]
        j = np.where(prognames == "Other")[0][0]
        if e["SURVEY"][i] == survey:
            exp_progname = "{}/{}".format(survey.capitalize(), e["FAPRGRM"][i].lower())
            jj = np.where(prognames == exp_progname)[0]
            if jj.size > 0:
                j = jj[0]
        marker, s = progmarkers[j], progss[j]
        jj_legend.append(j)
        ax.scatter(
            ax.projection_ra(e["TILERA"][[i]]),
            ax.projection_dec(e["TILEDEC"][[i]]),
            c=col,
            marker=marker,
            s=s,
            zorder=1,
        )

        # AR exposure number
        if i % 5 == 0:
            ax.text(
                ax.projection_ra(e["TILERA"][[i]]),
                ax.projection_dec(e["TILEDEC"][[i]] + 5),
                str(i),
                c=col,
                fontsize=1.5 * fontsize,
                ha="center",
                va="center",
                zorder=2,
            )

        # AR exposure obs. time
        time_i = (
            Time(e["MJD"][i], format="mjd")
            .to_datetime(timezone=kpno_tz)
            .strftime("%H:%M")
        )
        if i % 10 == 0:
            if i != 0:
                xtxt += dxtxt
            ytxt = ytxt0
            for dx, txt in zip(
                [0, dxtxt_expid, dxtxt_time, dxtxt_tileid],
                ["N", "EXPID", "MST", "TILEID"],
            ):
                ax.text(
                    xtxt + dx,
                    ytxt,
                    txt,
                    color="k",
                    fontsize=7,
                    zorder=10,
                    transform=ax.transAxes,
                )
            ytxt += dytxt
        for dx, txt in zip(
            [0, dxtxt_expid, dxtxt_time, dxtxt_tileid],
            [
                str(i),
                "{:08d}".format(e["EXPID"][i]),
                time_i,
                "{:06d}".format(e["TILEID"][i]),
            ],
        ):
            ax.text(
                xtxt + dx,
                ytxt,
                txt,
                color=col,
                fontsize=fontsize,
                zorder=10,
                transform=ax.transAxes,
            )
        ytxt += dytxt

    ax.set_title(title)
    # AR DES, galactic, ecliptic plane
    plot_des(ax, c="k", lw=0.5, alpha=1, zorder=1)
    # plot_gp_ep(ax, "galactic", c="k", lw=1, alpha=1, zorder=1)
    plot_gp_ep(ax, "barycentricmeanecliptic", c="k", lw=0.25, alpha=1, zorder=1)

    # AR Moon position for exposures taken that night + tiles completed
    mjds = np.linspace(e["MJD"].min(), e["MJD"].max(), 10)
    moonras, moondecs, moonphases = get_moon_radecphase(mjds)
    ax.scatter(
        ax.projection_ra(moonras),
        ax.projection_dec(moondecs),
        c=0.5 + moonphases / 2,
        s=10 * moonphases**2,
        cmap=matplotlib.cm.binary,
        vmin=0,
        vmax=1,
    )

    # AR 50-deg-radius circle around the Moon
    for ra, dec in zip(moonras, moondecs):
        custom_plot_sky_circles(
            ax,
            ra,
            dec,
            2 * 50,
            ec="k",
            fc="k",
            alpha=0.01,
            lw=0.5,
            zorder=0,
        )

    # AR blank text region
    ax.fill_between(
        [0.0, 1.0], [0, 0], [0.3, 0.3], color="w", zorder=5, transform=ax.transAxes
    )

    # AR legend
    jj = np.unique(jj_legend)
    for progname, progmarker, progs in zip(prognames[jj], progmarkers[jj], progss[jj]):
        ax.scatter(None, None, c="k", marker=progmarker, s=progs, label=progname)
    ax.legend(markerscale=2, loc="center right", bbox_to_anchor=(1.00, 0.85))

    plt.savefig(outpng, bbox_inches="tight")
    plt.close()
