#!/usr/bin/env python

import matplotlib

matplotlib.use("Agg")
# AR general
import subprocess
import os
from datetime import datetime
from time import time
import multiprocessing

# AR scientifical
import numpy as np

# AR astropy
from astropy.table import Table
from astropy.time import Time, TimeDelta
from astropy.coordinates import EarthLocation, SkyCoord
from astropy import units as u

# AR desisurveyops
from desisurveyops.status_utils import (
    get_filename,
    get_fns,
    get_obsdone_tiles,
    get_mjd,
    get_moon_radecphase,
    get_ffmpeg,
)

# AR desimodel
from desimodel.focalplane.geometry import get_tile_radius_deg

# AR desiutil
from desiutil.log import get_logger

# AR matplotlib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import PatchCollection
from matplotlib.path import Path
from matplotlib.patches import Polygon, Rectangle, PathPatch

#
from PIL import Image

log = get_logger()

_ffmpeg = None


def process_spacewatch(
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
    Wrapper function to generate the spacewatch per-night movies.

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
        Usually use specprod=daily.
    """

    log.info(
        "{}\tBEGIN process_spacewatch".format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    )
    start = time()

    # AR ffmpeg
    global _ffmpeg
    if _ffmpeg is None:
        _ffmpeg = get_ffmpeg()

    # AR obs, donetiles
    obs_tiles, obs_nights, obs_progs, done_tiles = get_obsdone_tiles(survey, specprod)

    # AR all nights
    nights = np.unique(obs_nights)

    # AR the spacewatch was taken out for few nights, we drop on them..
    black_nights = [20240812, 20240813, 20240814]
    sel = ~np.isin(nights, black_nights)
    nights = nights[sel]

    myargs = []
    for night in nights:
        outmp4 = get_filename(outdir, survey, "spacewatch", night=night, ext="mp4")
        if (not os.path.isfile(outmp4)) | (recompute):
            # if True:
            myargs.append((outmp4, specprod, night))
            log.info(outmp4)

    if len(myargs) > 0:
        log.info(
            "{}\tGenerate {} per-night spacewatch movies".format(
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                len(myargs),
            )
        )
        pool = multiprocessing.Pool(processes=numproc)
        with pool:
            _ = pool.starmap(spacewatch_night, myargs)

    # AR latest files with no night in filename
    night = nights[-1]
    cmd = "cp {} {}".format(
        get_filename(outdir, survey, "spacewatch", night=night, ext="mp4"),
        get_filename(outdir, survey, "spacewatch", night=None, ext="mp4"),
    )
    log.info(cmd)
    os.system(cmd)

    log.info(
        "{}\tEND process_spacewatch (took {:.1f}s)".format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), time() - start
        )
    )


def get_spacewatch_dir():
    """
    Get the folder with the individual spacewatch images.

    Returns:
        the folder name (str)
    """

    return os.path.join(
        os.getenv("DESI_ROOT"),
        "external",
        "spacewatch",
        "images",
        "cropped",
    )


# AR https://github.com/dylanagreen/desipoint/blob/b75c08f5468272935c58993ef83a4163542cda90/desipoint.py#L21-L70
def radec_to_altaz(ra, dec, time):
    """
    Get the altitude and azimuth for a given (ra, dec, time)

    Args:
        ra: np.array of R.A. (float)
        dec: np.array of Dec. (float)
        time: time in ISOT format (str)

    Returns:
        alt: np.array of altitudes (float)
        az: np.array of azimuth (float)

    Notes:
        Taken from https://github.com/dylanagreen/desipoint.
    """
    camera = (31.959417 * u.deg, -111.598583 * u.deg)
    cameraearth = EarthLocation(lat=camera[0], lon=camera[1], height=2120 * u.meter)
    radeccoord = SkyCoord(
        ra=ra,
        dec=dec,
        unit="deg",
        obstime=time,
        location=cameraearth,
        frame="icrs",
        temperature=5 * u.deg_C,
        pressure=78318 * u.Pa,
    )
    altazcoord = radeccoord.transform_to("altaz")
    return (altazcoord.alt.degree, altazcoord.az.degree)


def altaz_to_xy(alt, az):
    """
    Get the spacewatch image (x, y) pixels for a set of (alt, az).

    Args:
        alt: np.array of altitudes (float)
        az: np.array of azimuth (float)

    Returns:
        x: list of pixel values (float)
        y: list of pixel values (float)

    Notes:
        Taken from https://github.com/dylanagreen/desipoint.
    """
    alt = np.asarray(alt)
    az = np.asarray(az)
    # Reverse of r interpolation
    r_sw = [0, 55, 110, 165, 220, 275, 330, 385, 435, 480, 510]
    theta_sw = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
    r = np.interp(90 - alt, xp=theta_sw, fp=r_sw)
    az = az + 0.1  # Camera rotated 0.1 degrees.
    # Angle measured from vertical so sin and cos are swapped from usual polar.
    # These are x,ys with respect to a zero.
    x = -1 * r * np.sin(np.radians(az))
    y = r * np.cos(np.radians(az))
    # y is measured from the top!
    center = (512, 512)
    x = x + center[0]
    y = center[1] - y
    # Spacewatch camera isn't perfectly flat, true zenith is 2 to the right
    # and 3 down from center.
    x += 2
    y += 3
    return (x.tolist(), y.tolist())


def radec_to_xy(ra, dec, time):
    """
    Get the spacewatch image (x, y) pixels for a set of (ra, dec, time).

    Args:
        ra: np.array of R.A. (float)
        dec: np.array of Dec. (float)
        time: time in ISOT format (str)

    Returns:
        x: list of pixel values (float)
        y: list of pixel values (float)

    Notes:
        Taken from https://github.com/dylanagreen/desipoint.
    """
    alt, az = radec_to_altaz(ra, dec, time)
    x, y = altaz_to_xy(alt, az)
    return (x, y)


def imgfn2time(fn):
    """
    Convert the spacewatch image file name to a Time() variable.

    Args:
        fn: path to the spacewatch image name (str)

    Returns:
        t: an astropy Time object.

    Notes:
        spacewatch file names are like: 20250530_080405.jpg.
    """
    night = os.path.basename(fn).split("_")[0][0:4]
    month = os.path.basename(fn).split("_")[0][4:6]
    day = os.path.basename(fn).split("_")[0][6:8]
    hr = os.path.basename(fn).split("_")[1][0:2]
    mn = os.path.basename(fn).split("_")[1][2:4]
    sc = os.path.basename(fn).split("_")[1][4:6]
    t = Time("{}-{}-{}T{}:{}:{}".format(night, month, day, hr, mn, sc))
    return t


def spacewatch_night(
    outmp4,
    specprod,
    night,
    swdir=None,
    desifn=None,
):
    """
    Generate the spacewatch movie for a given night.

    Args:
        outmp4: the output movie file name (str)
        specprod: spectroscopic production (e.g. daily) (str)
        night: the night to consider (int)
        swdir (optional, defaults to get_spacewatch_dir()): folder with the spacewatch images (str)
        desifn (optional, defaults to $DESI_ROOT/users/raichoor/desi-14k-footprint/desi-14k-footprint-dark.ecsv):
            file with the DESI nominal footprint (str)

    Notes:
        Credits to https://github.com/dylanagreen/desipoint
            to get the (ra, dec) to (x, y) conversion.
    """

    if swdir is None:
        swdir = get_spacewatch_dir()

    # AR ffmpeg
    global _ffmpeg
    if _ffmpeg is None:
        _ffmpeg = get_ffmpeg()

    # AR files
    fns = get_fns(specprod=specprod)
    if desifn is None:
        desifn = fns["desifoot"]

    # AR various settings
    dpi = 128
    angs = np.linspace(2 * np.pi, 0, 100)
    trad = get_tile_radius_deg()
    tdelta = TimeDelta(120, format="sec")
    mycols = {
        "mainbackup": "r",
        "mainbright": "orange",
        "mainbright1b": "gold",
        "maindark": "c",
        "maindark1b": "b",
        "other": "g",
    }

    # AR time window for the movie (5pm - 8am)
    t = Time(
        "{}-{}-{}T00:00:00".format(str(night)[:4], str(night)[4:6], str(night)[6:8]),
    )
    t += TimeDelta(24 * 3600, format="sec")  # midnight utc
    time_start = t - TimeDelta(-5, format="sec")  # 5:00:05pm the day before
    time_end = t + TimeDelta(15 * 3600 + 5, format="sec")  # 8:00:05am that day
    # time_start = t + TimeDelta(7*3600 - 1*3600 + 5, format="sec")  # 5:00:05pm the day before
    # time_end = t + TimeDelta(7*3600 + 1*3600 + 5, format="sec")    # 8:00:05am that day
    log.info(
        "night = {} -> time_start = {}, time_end = {}".format(
            night, time_start, time_end
        )
    )

    # AR
    e = Table.read(fns["spec"]["exps"])
    sel = (e["NIGHT"] == night) & (e["EFFTIME_SPEC"] > 0)
    e = e[sel]
    e["MJD_BEGIN"] = [get_mjd(expid, night) for expid in e["EXPID"]]
    e["MJD_END"] = e["MJD_BEGIN"] + e["EXPTIME"] / 24.0 / 3600.0
    _, _, e["MOON_PHASE"] = get_moon_radecphase(e["MJD_BEGIN"])

    # AR cut on the time window of the movie
    sel = e["MJD_END"] >= time_start.mjd
    sel &= e["MJD_BEGIN"] <= time_end.mjd
    e = e[sel]

    # AR better safe than sorry...
    ii = np.where((e["TILERA"] == 0) & (e["TILEDEC"] == 0))[0]
    for i in ii:
        tileid = e["TILEID"][i]
        tileidpad = "{:06d}".format(tileid)
        fn = os.path.join(
            os.getenv("DESI_TARGET"),
            "fiberassign",
            "tiles",
            "trunk",
            tileidpad[:3],
            "fiberassign-{}.fits.gz".format(tileidpad),
        )
        log.info("EXPID={}\tpopulate TILERA,TILEDEC from {}".format(e["EXPID"][i], fn))
        hdr = fitsio.read_header(fn, 0)
        e["TILERA"][i] = hdr["TILERA"]
        e["TILEDEC"][i] = hdr["TILEDEC"]

    # AR desi
    d = Table.read(desifn)
    desi = {}
    for cap in ["NGC", "SGC"]:
        sel = d["CAP"] == cap
        desi[cap] = {"RA": d["RA"][sel], "DEC": d["DEC"][sel]}

    # AR equator, ecliptic
    npts = 1000
    grid_ras, grid_decs = {}, {}
    for dec in np.arange(-75, 75 + 15, 15, dtype=int):
        key = "dec{}".format(dec)
        grid_ras[key], grid_decs[key] = np.linspace(0, 360, npts), dec + np.zeros(npts)
    for ra in np.arange(0, 330 + 30, 30, dtype=int):
        key = "ra{}".format(ra)
        grid_ras[key], grid_decs[key] = ra + np.zeros(npts), np.linspace(-75, 90, npts)
    ecl_lons, ecl_lats = np.linspace(0, 360, npts), np.zeros(npts)
    cs = SkyCoord(
        lon=ecl_lons * u.degree,
        lat=ecl_lats * u.degree,
        distance=1 * u.Mpc,
        frame="heliocentrictrueecliptic",
    )
    grid_ras["ecl"], grid_decs["ecl"] = cs.icrs.ra.degree, cs.icrs.dec.degree

    # AR list of spacewatch images
    fns = []
    t = time_start
    while t.mjd < (time_end - tdelta).mjd:
        fn = os.path.join(
            swdir,
            str(t.ymdhms[0]),
            "{:02d}".format(t.ymdhms[1]),
            "{:02d}".format(t.ymdhms[2]),
            "{}_{}.jpg".format(
                t.iso.split()[0].replace("-", ""),
                t.iso.split()[1].replace(":", "").split(".")[0],
            ),
        )
        msg = None
        if not os.path.isfile(fn):
            fn = fn.replace("05.jpg", "06.jpg")
            if not os.path.isfile(fn):
                fn = fn.replace("06.jpg", "07.jpg")
                if not os.path.isfile(fn):
                    msg = "missing {}".format(fn.replace("07.jpg", "0{5,6,7}.jpg"))
        if msg is None:
            fns.append(fn)
        else:
            log.warning(msg)
        t += tdelta
    log.info("night = {} -> found {} spacewatch images".format(night, len(fns)))

    # image
    fn = fns[0]
    img = np.asarray(Image.open(fn))
    t = imgfn2time(fn)

    # Set up the figure the same way we usually do for saving so the image is the
    # only thing on the axis.
    y = img.data.shape[0] / dpi
    x = img.data.shape[1] / dpi

    fig = plt.figure()
    fig.set_size_inches(x, y)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])  # 0 - 100% size of figure
    ax.set_axis_off()
    fig.add_axes(ax)

    # Adds the image into the axes and displays it
    im = ax.imshow(img.data, cmap="gray", vmin=0, vmax=255)
    coverup = ax.add_patch(Rectangle((0, 1024 - 50), 300, 50, fc="black"))

    patches = {}
    fontsize = 22
    patches["credits"] = ax.text(
        0, 40, "SPACEWATCH\u00AE", fontsize=int(0.5 * fontsize), color="white"
    )
    patches["MST"] = ax.text(
        0,
        100,
        "MST : {}".format(t.iso.split(" ")[1].split(".")[0]),
        fontsize=fontsize,
        color="white",
    )
    patches["expid"] = ax.text(1024, 60, "", fontsize=fontsize, color="w", ha="right")
    patches["faflavor"] = ax.text(
        1024, 100, "", fontsize=fontsize, color="w", ha="right"
    )
    patches["tileid"] = ax.text(
        1024, 1024 - 120, "", fontsize=fontsize, color="w", ha="right"
    )
    patches["efftime"] = ax.text(
        1024, 1024 - 80, "", fontsize=fontsize, color="w", ha="right"
    )
    patches["effexp"] = ax.text(
        1024, 1024 - 40, "", fontsize=fontsize, color="w", ha="right"
    )
    patches["moon_phase"] = ax.text(
        0, 1024 - 80, "", fontsize=fontsize, color="w", ha="left"
    )
    patches["sky_mag"] = ax.text(
        0, 1024 - 40, "", fontsize=fontsize, color="w", ha="left"
    )
    # AR desi
    for cap in desi.keys():
        xs, ys = radec_to_xy(desi[cap]["RA"], desi[cap]["DEC"], t.isot)
        xys = [(x, y) for (x, y) in zip(xs, ys)]
        patches[cap] = ax.add_patch(
            Polygon(xys, ec=(1, 0, 0, 1), fc=(1, 0, 0, 0.05), lw=1)
        )

    # AR grid, ecl
    for key in grid_ras:
        if key == "ecl":
            ec, ls = "y", "--"
        else:
            ec, ls = "w", "-"
        xs, ys = radec_to_xy(grid_ras[key], grid_decs[key], t.isot)
        xys = [(x, y) for (x, y) in zip(xs, ys)]
        codes = [Path.MOVETO]
        for xy in xys[1:]:
            codes.append(Path.LINETO)
        path = Path(xys, codes)
        patches[key] = ax.add_patch(PathPatch(path, ec=ec, fc="none", lw=0.05, ls=ls))
        if key[:2] == "ra":
            j = np.abs(grid_decs[key]).argmin()
            patches["{}txt".format(key)] = ax.text(
                xs[j],
                ys[j],
                "{:.0f}".format(grid_ras[key][j]),
                c="w",
                fontweight="ultralight",
                ha="center",
                va="center",
            )
    # AR exposures
    if len(e) > 0:

        for expid, ra, dec in zip(e["EXPID"], e["TILERA"], e["TILEDEC"]):
            ras = ra + trad * np.sin(angs) / np.cos(dec)
            decs = dec + trad * np.cos(angs)
            xs, ys = radec_to_xy(ras, decs, t.isot)
            xys = [(x, y) for (x, y) in zip(xs, ys)]
            patches[expid] = ax.add_patch(Polygon(xys, ec="none", fc="none", lw=1))

        # AR all exposures
        xs, ys = radec_to_xy(e["TILERA"], e["TILEDEC"], t.isot)
        xys = [(x, y) for (x, y) in zip(xs, ys)]
        codes = [Path.MOVETO]
        for xy in xys[1:]:
            codes.append(Path.LINETO)
        path = Path(xys, codes)
        patches["allexps"] = ax.add_patch(
            PathPatch(path, ec="none", fc="none", lw=0.25)
        )

    def update_img(fn):
        # print(fn)
        img = np.asarray(Image.open(fn))
        t = imgfn2time(fn)
        mst_t = t - TimeDelta(7 * 3600, format="sec")
        patches["MST"].set_text(
            "MST : {}".format(mst_t.iso.split(" ")[1].split(".")[0])
        )
        for cap in desi.keys():
            xs, ys = radec_to_xy(desi[cap]["RA"], desi[cap]["DEC"], t.isot)
            xys = [(x, y) for (x, y) in zip(xs, ys)]
            patches[cap].set_xy(xys)
        for key in grid_ras:
            xs, ys = radec_to_xy(grid_ras[key], grid_decs[key], t.isot)
            xys = [(x, y) for (x, y) in zip(xs, ys)]
            codes = [Path.MOVETO]
            for xy in xys[1:]:
                codes.append(Path.LINETO)
            path = Path(xys, codes)
            patches[key].set_path(path)
            if key[:2] == "ra":
                j = np.abs(grid_decs[key]).argmin()
                patches["{}txt".format(key)].set_x(xs[j])
                patches["{}txt".format(key)].set_y(ys[j])
        # AR "erasing" previous ones
        for expid in e["EXPID"]:
            patches[expid].set_ec("none")
        for key in [
            "expid",
            "faflavor",
            "tileid",
            "efftime",
            "effexp",
            "moon_phase",
            "sky_mag",
        ]:
            patches[key].set_text("")
        if len(e) > 0:
            patches["allexps"].set_fc("none")

        # AR all previous exposures
        ii = np.where(e["MJD_BEGIN"] < t.mjd)[0]
        if ii.size > 0:
            xs, ys = radec_to_xy(e["TILERA"][ii], e["TILEDEC"][ii], t.isot)
            xys = [(x, y) for (x, y) in zip(xs, ys)]
            codes = [Path.MOVETO]
            for xy in xys[1:]:
                codes.append(Path.LINETO)
            path = Path(xys, codes)
            patches["allexps"].set_path(path)
            patches["allexps"].set_ec("k")
            for i in ii:
                col = mycols["other"]
                if e["FAFLAVOR"][i] in [
                    "mainbackup",
                    "mainbright",
                    "mainbright1b",
                    "maindark",
                    "maindark1b",
                ]:
                    col = mycols[e["FAFLAVOR"][i]]
                expid, ra, dec = e["EXPID"][i], e["TILERA"][i], e["TILEDEC"][i]
                ras = ra + trad * np.cos(angs) / np.cos(np.radians(dec))
                decs = dec + trad * np.sin(angs)
                xs, ys = radec_to_xy(ras, decs, t.isot)
                xys = [(x, y) for (x, y) in zip(xs, ys)]
                patches[expid].set_xy(xys)
                patches[expid].set_ec(col)
                patches[expid].set_lw(0.1)

        # AR exposures on that image
        ii = np.where((e["MJD_BEGIN"] < t.mjd) & (e["MJD_END"] > t.mjd))[0]
        if ii.size > 1:
            raise ValueError(
                "found {} exposures ({}) for {}; only one expected".format(
                    ii.size,
                    ",".join(e["EXPID"][ii].astype(str)),
                    os.path.basename(fn),
                )
            )
        if ii.size == 1:
            i = ii[0]
            col = mycols["other"]
            if e["FAFLAVOR"][i] in [
                "mainbackup",
                "mainbright",
                "mainbright1b",
                "maindark",
                "maindark1b",
            ]:
                col = mycols[e["FAFLAVOR"][i]]
            expid, tileid, ra, dec = (
                e["EXPID"][i],
                e["TILEID"][i],
                e["TILERA"][i],
                e["TILEDEC"][i],
            )
            ras = ra + trad * np.cos(angs) / np.cos(np.radians(dec))
            decs = dec + trad * np.sin(angs)
            xs, ys = radec_to_xy(ras, decs, t.isot)
            xys = [(x, y) for (x, y) in zip(xs, ys)]
            patches[expid].set_xy(xys)
            patches[expid].set_ec(col)
            patches[expid].set_lw(1)
            patches["expid"].set_text("EXPID = {}".format(expid))
            patches["faflavor"].set_text("faflavor = {}".format(e["FAFLAVOR"][i]))
            patches["tileid"].set_text("tileid = {}".format(tileid))
            patches["efftime"].set_text(
                "efftime = {:.0f}s".format(e["EFFTIME_SPEC"][i])
            )
            patches["effexp"].set_text(
                "eff / exp = {:.2f}".format(e["EFFTIME_SPEC"][i] / e["EXPTIME"][i])
            )
            patches["moon_phase"].set_text(
                "Moon_Illum. = {:.2f}".format(e["MOON_PHASE"][i])
            )
            patches["sky_mag"].set_text(
                "sky = {:.2f} mag".format(e["SKY_MAG_R_SPEC"][i])
            )
            for key in [
                "expid",
                "faflavor",
                "tileid",
                "efftime",
                "effexp",
                "moon_phase",
                "sky_mag",
            ]:
                patches[key].set_color(col)
        #
        im.set_data(img.data)
        return im

    ani = animation.FuncAnimation(fig, update_img, fns, interval=30)
    plt.rcParams["animation.ffmpeg_path"] = _ffmpeg
    writer = animation.writers["ffmpeg"](fps=10)
    ani.save(outmp4, writer=writer, dpi=dpi)
