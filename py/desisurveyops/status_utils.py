#!/usr/bin/env python

import matplotlib

matplotlib.use("Agg")
# AR general
import os
from glob import glob
from datetime import datetime
import tempfile
import multiprocessing
from pkg_resources import resource_filename

# AR scientifical
import numpy as np
import fitsio

# AR astropy
from astropy.time import Time
from astropy.table import Table
from astropy.coordinates import EarthLocation

# from astropy import units as u
# AR desitarget
from desitarget.geomask import match_to, match

# AR desispec
from desispec.io.util import get_tempfilename

# AR desiutil
from desiutil.log import get_logger

# AR matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

#
from PIL import Image

log = get_logger()


def get_filename(
    outdir,
    survey,
    name,
    program_str=None,
    ext=None,
    case=None,
    quant=None,
    tileid=None,
    night=None,
    expid=None,
    yr=None,
):
    """
    Utility function to get status file names used throughout functions called in desi_survey_status.

    Args:
        outdir: output folder (str)
        survey: survey (str)
        name: "tiles" or "skymap" or "qso" or "zhist" or "nspec" or "fpstate" or "obsconds" or "spacewatch" or "html" or "exposures" (str)
        program_str (optional, defaults to None): "BACKUP", "BRIGHT", "BRIGHT4PASS", "BRIGHT1B", "DARK", or "DARK1B" (str)
        ext (optional, defaults to None): "ecsv", "fits", "png", "mp4", "html", "css" (str)
        case (optional, defaults to None):
            "history" for name=tiles,
            "goal", "obs", "done", "pending", "complra", "compllst" for name=skymap,
            "night" for skymap, hist, fpstate, obsconds, spacewatch, exposures
            "cumulative" for name=obsconds
            "zmtl" for name=nspec,
             (str)
        quant (optional, defaults to None): ntile, fraccov, fibxys (str)
        tileid (optional, defaults to None): tileid (int)
        yr (optional, defaults to None): year (int)

    Notes:
        Maybe I have forgotten some cases in the docstr..
    """

    def _get_str(x):
        return "" if x is None else "-{}".format(x)

    basefn = None
    if name == "skymap":
        assert program_str is not None
        outdir2 = os.path.join(outdir, name, program_str.lower())
    elif name == "html":
        outdir2 = outdir
    else:
        outdir2 = os.path.join(outdir, name)

    # AR history tiles
    if name == "tiles":
        assert program_str is None
        assert case == "history"
        assert ext == "ecsv"
        basefn = "tiles-{}-history.ecsv".format(survey)

    # AR qso
    if name == "qso":
        assert ext in ["ecsv", "png"]
        assert program_str is not None
        basefn = "{}-{}-qso.{}".format(survey, program_str.lower(), ext)

    # AR zhist
    elif name == "zhist":
        assert ext in ["ecsv", "png"]
        if tileid is not None:
            assert program_str is None
            assert ext == "ecsv"
            if night is None:
                basefn = "{}-zhist-{}-thru20??????.{}".format(survey, tileid, ext)
            else:
                basefn = "{}-zhist-{}-thru{}.{}".format(survey, tileid, night, ext)
        else:
            assert program_str is not None
            basefn = "{}-{}-zhist.{}".format(survey, program_str.lower(), ext)

    # AR skymap
    elif name == "skymap":
        assert ext in ["fits", "ecsv", "png", "mp4"]
        assert program_str is not None
        assert case in ["goal", "obs", "done", "pending", "complra", "compllst"]
        if quant is not None:
            assert case in ["obs", "done"]
            assert quant in ["ntile", "fraccov"]
        night_str = _get_str(night)
        quant_str = _get_str(quant)
        basefn = "{}-{}-skymap-{}{}{}.{}".format(
            survey, program_str.lower(), case, night_str, quant_str, ext
        )

    # AR obsconds (not survey needed)
    elif name == "obsconds":
        assert ext in ["pdf", "png"]
        if ext == "pdf":
            assert case is None
            assert night is not None
            basefn = "obsconds-{}.{}".format(night, ext)
        else:
            assert case in ["night", "cumulative"]
            if case == "night":
                assert night is not None
                basefn = "obsconds-{}.{}".format(night, ext)
            else:
                basefn = "obsconds-cumulative.{}".format(ext)

    # AR fpstate (no survey needed)
    elif name == "fpstate":
        assert ext in ["ecsv", "png", "mp4"]
        if night is not None:
            assert ext in ["ecsv", "png"]
            basefn = "fpstate-{}.{}".format(night, ext)
        else:
            if quant is not None:
                assert ext == "ecsv"
                assert quant == "fibxys"
                basefn = "fpstate-{}.{}".format(quant, ext)
            else:
                basefn = "fpstate.{}".format(ext)

    # AR nspec
    elif name == "nspec":
        assert ext in ["fits", "ecsv", "png"]
        assert case in [None, "zmtl"]
        if case == "zmtl":
            assert yr is not None
            basefn = "{}-zmtl-{}.{}".format(survey, yr, ext)
        else:
            assert yr is None
            basefn = "{}-nspec.{}".format(survey, ext)

    # AR skyseq, spacewatch, exposures (no survey needed)
    elif name in ["skyseq", "spacewatch", "exposures"]:
        assert ext in ["png", "ecsv", "mp4", "html"]
        night_str = _get_str(night)
        basefn = "{}{}.{}".format(name, night_str, ext)

    # AR exposures
    elif name == "exposures":
        assert ext == "html"
        assert night is not None
        basefn = "{}-{}.{}".format(name, night, ext)

    # AR html
    elif name == "html":
        assert case is None
        assert ext in ["html", "css"]
        basefn = "{}-status.{}".format(survey, ext)

    # AR are we good?
    if basefn is None:
        xs = [
            "name",
            "program_str",
            "ext",
            "case",
            "quant",
            "tileid",
            "night",
            "expid",
            "yr",
        ]
        msg = "not expected arguments: {}".format(
            ", ".join(["{}={}".format(_, eval("str({})".format(_))) for _ in xs])
        )
        log.error(msg)
        raise ValueError(msg)

    return os.path.join(outdir2, basefn)


def create_folders_structure(outdir):
    """
    Create the folder architecture needed by desi_survey_status.

    Args:
        outdir: the working folder (str)
    """
    mydirs = []
    for prog in ["backup", "bright4pass", "bright", "bright1b", "dark", "dark1b"]:
        mydirs.append(os.path.join(outdir, "skymap", prog))
    for name in [
        "qso",
        "obsconds",
        "fpstate",
        "nspec",
        "skyseq",
        "exposures",
        "zhist",
        "spacewatch",
        "tiles",
    ]:
        mydirs.append(os.path.join(outdir, name))
    for mydir in mydirs:
        if not os.path.isdir(mydir):
            log.info("create {}".format(mydir))
            os.makedirs(mydir)


def get_programs_npassmaxs(survey="main"):
    """
    Get the programs properties for a given survey.

    Args:
        survey (optional, defaults to main): survey name (str)

    Returns:
        programs: list of programs (np.array of str)
        npassmaxs: number of passes for each program (np.array of int)
        program_strs: list of program names used in the code (np.array of str)

    Notes:
        npassmax is None in general; e.g. set to 4 for BRIGHT4PASS.
        For instance, for the main survey:
            programs: BACKUP, BRIGHT, BRIGHT, BRIGHT1B, DARK, DARK1B
            npassmaxs: None, 4, None, None, None, None
            program_str: BACKUP, BRIGHT4PASS, BRIGHT, BRIGHT1B, DARK, DARK1B
    """

    programs, npassmaxs = None, None
    if survey == "main":
        xs = [
            ("BACKUP", None),
            ("BRIGHT", 4),
            ("BRIGHT", None),
            ("BRIGHT1B", None),
            ("DARK", None),
            ("DARK1B", None),
        ]
        programs = np.array([_[0] for _ in xs])
        npassmaxs = np.array([_[1] for _ in xs])
        program_strs = np.array(
            [
                "{}{}PASS".format(program, npassmax)
                if npassmax is not None
                else program
                for program, npassmax in zip(programs, npassmaxs)
            ]
        )

    return programs, npassmaxs, program_strs


def get_shutdowns(survey):
    """
    Returns infos about shutdowns.

    Args:
        survey: survey name (str)

    Returns:
        start_nights: first night of events (np.array of int)
        last_nights: last night of events (np.array of int)
        comments: events description (np.array of str)

    Notes:
        This function will need to be updated for any further shutdown.
    """

    start_nights, end_nights, comments = None, None, None
    if survey == "main":
        xs = [
            (20210711, 20210920, "Planned shutdown; upgrade focal plane electronics"),
            (20220614, 20220909, "Contreras fire"),
            (20230622, 20230804, "Planned shutdown; mirror realuminization"),
            (20240410, 20240602, "Fibers guider crush"),
            (20250403, 20250414, "Dome slip ring failure"),
        ]
        start_nights = np.array([_[0] for _ in xs])
        end_nights = np.array([_[1] for _ in xs])
        comments = np.array([_[2] for _ in xs])

    return start_nights, end_nights, comments


def get_history_tiles_dir(survey):
    """
    Returns the folder where the tiles-{survey}.ecsv files for various
        revisions are stored.

    Args:
        survey: survey name (str)

    Returns:
        folder name (str)

    Notes:
        So far I have hard-coded my local folder.
        In the future, we could for instance store those in github.
    """
    # TODO: store those files in github?
    assert survey == "main"
    return os.path.join(
        os.getenv("DESI_ROOT"), "users", "raichoor", "main-status-dev", "tiles"
    )


def get_history_tiles_infos(survey, outfn=None):
    """
    Get infos about the history of tiles-{survey}.ecsv.

    Args:
        survey: survey name (str)
        outfn (optional, defaults to None): if set, output will be writtento that file (str)

    Notes:
        This will have to be manually updated in the future.
        In the future, maybe that could be automated...
    """
    assert survey == "main"

    # TODO: find an automatic way to do that?
    d = Table()
    d.meta["FOLDER"] = get_history_tiles_dir(survey)

    # AR structure: programs, night, revision, comment
    # AR note, for revision=5129 (DARK1B): the revision was done on 20250507,
    # AR but three DARK1B tiles were turned on the day before, for testing
    # AR so we "artificially set" the date to 20250505
    xs = [
        #
        (20210512, 425, "BRIGHT", 0, 3, "Initial definition of the tiles"),
        (20210512, 425, "BRIGHT4PASS", 0, 3, "Initial definition of the tiles"),
        (20210512, 425, "DARK", 0, 6, "Initial definition of the tiles"),
        #
        (20211119, 869, "BACKUP", 0, 0, "Turn on BACKUP tiles"),
        #
        (20220314, 1332, "BACKUP", 0, 0, "Turn off Dec>80 BACKUP tiles"),
        #
        (20230927, 3091, "BRIGHT", 4, 4, "Add new PASS=4 tiles"),
        #
        (20241010, 4177, "BRIGHT", 0, 4, "Add new tiles in DES region"),
        (20241010, 4177, "BRIGHT4PASS", 0, 3, "Add new tiles in DES region"),
        (20241010, 4177, "DARK", 0, 6, "Add new tiles in DES region"),
        #
        (20250505, 5129, "DARK1B", 7, 8, "Turn on DARK1B tiles in DR9 region"),
    ]
    d["NIGHT"] = [x[0] for x in xs]
    d["REVISION"] = [x[1] for x in xs]
    d["PROGRAM_STR"] = [x[2] for x in xs]
    d["PASSMIN"] = [x[3] for x in xs]
    d["PASSMAX"] = [x[4] for x in xs]
    d["COMMENT"] = [x[5] for x in xs]

    if outfn is not None:
        d.write(outfn)

    return d


def get_history_tilesfn(survey, program_str, opsnight):
    """
    Get the relevant "historical" tiles-{survey} name for a given program and night.

    Args:
        survey: survey name (str)
        program_str: program full name (str)
        opsnight: night of observation (int)

    Returns:
        fn: full path of the tiles-{survey}.ecsv file (str)

    Notes:
        So far I have chosen to store those files under
            the tiles-{}-{}-rev{:05d}.ecsv.format(survey, night, revision)
            format.
    """

    # AR read "master" file
    # opsnightdir, basefn = get_master_history_tiles_dir_basename(survey)
    # d = Table.read(os.path.join(opsnightdir, basefn))
    d = get_history_tiles_infos(survey)

    # AR sort by increasing night, in case
    d = d[d["NIGHT"].argsort()]

    # AR cut on versions relevant for the program
    sel = d["PROGRAM_STR"] == program_str
    d = d[sel]

    # AR pick the relevant night
    if opsnight <= d["NIGHT"][0]:
        msg = "requested night ({}) is before the start of the {} {}/{} ({})".format(
            opsnight, survey, program, d["NIGHT"][0]
        )
        log.error(msg)
        raise ValueError(msg)
    i = np.where(d["NIGHT"] < opsnight)[0][-1]

    # AR if the selected night is the last listed one, it means that the "regular"
    # AR tiles file is ok, so we skip
    if i == len(d) - 1:
        fn = get_fns(survey=survey)["ops"]["tiles"]
        log.info(
            "as opsnight={} is larger than {}, we pick {}".format(
                opsnight, d["NIGHT"][-1], fn
            )
        )
    else:
        opsnightdir = get_history_tiles_dir(survey)
        basefn = "tiles-{}-{}-rev{:05d}.ecsv".format(
            survey, d["NIGHT"][i], d["REVISION"][i]
        )
        fn = os.path.join(opsnightdir, basefn)
        log.info("as opsnight={}, we pick {}".format(opsnight, fn))

    return fn


def get_fns(survey="main", specprod="daily", opsnight=None, program=None):
    """
    Utility function to retrieve the relevant files for a given survey and specprod.

    Args:
        survey (optional, defaults to "survey"): survey name (str)
        specprod (optional, defaults to "daily"): specprod name (str)
        opsnight (optional, defaults to None): if set, picks up the tiles-{survey}.ecsv file relevant for that night (str)
        program (optional, defaults to None): if set, picks up the tiles-{survey}.ecsv file relevant for that program (str)

    Returns:
        mydict: a dictionary with this structure:
            "ops":
                "tiles": the tiles-{survey}.ecsv file
                "status": the tiles-specstatus.ecsv
            "spec":
                "exps": the exposures-{specprod}.csv file
                "tiles": the tiles-{specprod}.csv file
                "arxivdir": the $DESI_ROOT/spectro/redux/daily/tiles/archive folder
                "cumuldir": the $DESI_ROOT/spectro/redux/{specprod}/tiles/cumulative folder
            "gfa": the latest $DESI_ROOT/survey/GFA/offline_matched_coadd_ccds_{survey}-thru-202?????.fits file
            "moon": $DESI_ROOT/users/ameisner/GFA/gfa_reduce_etc/gfa_ephemeris.fits
            "desifoot": $DESI_ROOT/users/raichoor/desi-14k-footprint/desi-14k-footprint-dark.ecsv
            "desfoot": desiutil data/DES_footprint.txt file

    Notes:
        This function is heavily used in the desi_survey_status calls.
    """

    if (opsnight is not None) & (survey != "main"):
        msg = "opsnight option only implemented for survey=main so far"
        log.error(msg)
        raise ValueError(msg)

    opsdir = os.path.join(os.getenv("DESI_SURVEYOPS"), "ops")
    specdir = os.path.join(os.getenv("DESI_ROOT"), "spectro", "redux", specprod)
    gfadir = os.path.join(os.getenv("DESI_ROOT"), "survey", "GFA")

    mydict = {
        "ops": {
            "tiles": os.path.join(opsdir, "tiles-{}.ecsv".format(survey)),
            "status": os.path.join(opsdir, "tiles-specstatus.ecsv"),
        },
        "spec": {
            "exps": os.path.join(specdir, "exposures-{}.csv".format(specprod)),
            "tiles": os.path.join(specdir, "tiles-{}.csv".format(specprod)),
            "arxivdir": os.path.join(
                os.getenv("DESI_ROOT"), "spectro", "redux", "daily", "tiles", "archive"
            ),  # AR keep daily
            "cumuldir": os.path.join(specdir, "tiles", "cumulative"),
        },
        "gfa": sorted(
            glob(
                os.path.join(
                    gfadir,
                    "offline_matched_coadd_ccds_{}-thru_202?????.fits".format(survey),
                )
            )
        )[-1],
        "moon": os.path.join(
            os.getenv("DESI_ROOT"),
            "users",
            "ameisner",
            "GFA",
            "gfa_reduce_etc",
            "gfa_ephemeris.fits",
        ),
        "desifoot": os.path.join(
            os.getenv("DESI_ROOT"),
            "users",
            "raichoor",
            "desi-14k-footprint",
            "desi-14k-footprint-dark.ecsv",
        ),
        "desfoot": resource_filename("desiutil", "data/DES_footprint.txt"),
    }

    # AR pick the tiles-main.ecsv file before some changes?
    if opsnight is not None:
        if survey == "main":
            mydict["ops"]["tiles"] = get_history_tilesfn(survey, program, opsnight)

    return mydict


def get_backup_minefftime():
    """
    EFFTIME_SPEC min value for backup tiles
        (as requiring > MINTFRAC * GOALTIME rejects too many tiles...)
    """
    return 1.0


def get_obsdone_tiles(survey, specprod, verbose=False):
    """
    Get the list of observed tiles (+nights, programs) and done tiles for a survey.

    Args:
        survey: "main", "sv1" or "sv3" (str)
        specprod: spectroscopic production; e.g. "daily" (str)
        verbose (optional, defaults to False): verbose? (bool)

    Returns:
        obs_tiles: observed tiles (numpy array)
        obs_nights: nights corresponding to obs_tiles (numpy array)
        done_tiles: QA-done tiles (numpy array)

    Notes:
        We restrict to IN_DESI tiles.
        We consider as "observed" tiles with:
            - bright/dark: EFFTIME_SPEC > MINTFRAC * GOALTIME
            - backup: (EFFTIME_SPEC > MINTFRAC * GOALTIME) or (STATUS = obsstart)
        obs_nights lists the LASTNIGHT from tiles-{specprod}.csv.
        done status from tiles-specstatus.ecsv.
    """

    # to get tileids for the survey
    fns = get_fns(survey=survey, specprod=specprod)
    t = Table.read(fns["ops"]["tiles"])
    sel = t["IN_DESI"]
    t = t[sel]

    # obs. tile with EFFTIME_SPEC > GOALTIME * MINTFRAC
    #   or STATUS = obsstart for BACKUP
    d = Table.read(fns["spec"]["tiles"])
    sel = np.isin(d["TILEID"], t["TILEID"])
    sel &= (d["EFFTIME_SPEC"] > d["GOALTIME"] * d["MINTFRAC"]) | (
        (d["FAFLAVOR"] == "mainbackup") & (d["EFFTIME_SPEC"] > get_backup_minefftime())
    )
    d = d[sel]
    obs_tiles, obs_nights = d["TILEID"], d["LASTNIGHT"]
    ii = match_to(t["TILEID"], obs_tiles)
    if len(ii) != len(obs_tiles):
        msg = "issue with matching tiles and obs_tiles"
        log.error(msg)
        raise ValueError(msg)
    obs_progs = t["PROGRAM"][ii]

    # done
    # tiles-specstatus.ecsv
    d = Table.read(fns["ops"]["status"])
    sel = np.isin(d["TILEID"], t["TILEID"])
    sel &= d["QA"] == "good"
    d = d[sel]
    done_tiles = d["TILEID"]

    return obs_tiles, obs_nights, obs_progs, done_tiles


def get_tileid_night_str(tileids, nights):
    """
    Get a {tileid}-{night} list of strings.

    Args:
        tileids: np.array of tileids (int)
        nights: np.array of nights (int)

    Returns:
        tileids_nights: list of {tileid}-{night} (str)

    Notes:
        Fast approach to handle long arrays.
        Inputs need to be np.array()
    """
    _ = np.char.add(tileids.astype(str), ",")
    return np.char.add(_, nights.astype(str))


def get_spec_updated_mjd(specprod, tileids, lastnights):
    """
    Get the UPDATED column for some tileids for a production, along with the corresponding MJD.

    Args:
        specprod: the spectroscopic production (e.g. "daily") (str)
        tileids: single tileid or np.array of tileids (int)
        lastnights: single night  or np.array of nights (int)

    Returns:
        updateds: the UPDATED values (str)
        mjds: corresponding MJD values (str)

    Notes:
        This UPDATED column tells when the spectro. pipeline was run for that (tileids,lastnights).
        UPDATED are in the format "%Y-%m-%dT%H:%M:%S%z".
        Returns empty values for (tileids,lastnights) not in specprod.
        If inputs are scalar, outputs are scalar.
    """
    input_scalar = False
    if not hasattr(tileids, "__len__"):
        input_scalar = True
        tileids, lastnights = np.atleast_1d(tileids), np.atleast_1d(lastnights)

    fns = get_fns(specprod=specprod)
    t = Table.read(fns["spec"]["tiles"])
    if "UPDATED" not in t.colnames:
        msg = "need the UPDATED column in {} to proceed!".format(fn)
        log.error(msg)
        raise ValueError(msg)

    tileids_nights = get_tileid_night_str(tileids, lastnights)
    prod_tileids_nights = get_tileid_night_str(t["TILEID"], t["LASTNIGHT"])

    # AR UPDATED column
    ii, prod_ii = match(tileids_nights, prod_tileids_nights)
    updateds = np.zeros(len(tileids), dtype=object)
    updateds[ii] = t["UPDATED"][prod_ii]
    updateds = updateds.astype(str)

    # AR MJD
    updated_mjds = np.nan + np.zeros(len(tileids))
    updated_mjds[ii] = [
        Time(datetime.strptime("{}".format(_), "%Y-%m-%dT%H:%M:%S%z")).mjd
        for _ in updateds[ii]
    ]

    if input_scalar:
        tileids, lastnights = tileids[0], lastnights[0]
        return updateds[0], updated_mjds[0]
    else:
        return updateds, updated_mjds


# AR fns: png files..
def create_pdf(fns, outpdf, dpi=None):
    """
    Utility function to create a pdf file from a list of png files.

    Inputs:
        fns: list of pngs files (str)
        outpdf: output pdf file name (str)
        dpi (optiona, defaults to None): dpi (resolution) for the pdf file (int)

    Notes:
        Input files can be in other format than png (read with Image.open(fn)).
    """
    # AR convert to pdf
    tmp_outpdf = get_tempfilename(outpdf)
    with PdfPages(tmp_outpdf) as pdf:
        for fn in fns:
            fig, ax = plt.subplots()
            img = Image.open(fn)
            ax.imshow(img, origin="upper")
            ax.axis("off")
            pdf.savefig(fig, bbox_inches="tight", dpi=dpi)
            plt.close()

    # AR move to final location
    os.rename(tmp_outpdf, outpdf)


def get_ffmpeg():
    """
    Get the executable for ffmpeg.

    Returns:
        ffmpeg: $DESI_ROOT/users/raichoor/main-status-dev/ffmpeg-git-20220910-amd64-static/ffmpeg

    Notes:
        For whatever reason, usr/bin/ffmpeg is not working for spacewatch_night().
        So I ended up using a static version (in Sep. 2022).
        This executablee should be moved elsewhere! or need to solve why /usr/bin/ffmpeg causes issue.
    """

    # TODO: move elsewhere! or need to solve why /usr/bin/ffmpeg causes issue.
    ffmpeg = os.path.join(
        os.getenv("DESI_ROOT"),
        "users",
        "raichoor",
        "main-status-dev",
        "ffmpeg-git-20220910-amd64-static",
        "ffmpeg",
    )
    # ffmpeg = subprocess.Popen("which ffmpeg", stdout=subprocess.PIPE, shell=True).communicate()[0].strip().decode("utf-8")
    return ffmpeg


def create_mp4(fns, outmp4, duration=15, verbose=False):
    """
    Utility function to create an animated .mp4 from a set of input files (usually pngs).

    Args:
        fns: list of input filenames, in the correct order (list of str)
        outmp4: output .mp4 filename (str)
        duration (optional, defaults to 15): video duration in seconds (float)
        verbose (optional, default to False): if False, redirect prompt output to >/dev/null 2>&1  (bool)

    Notes:
        Requires ffmpeg to be installed.
        The movie uses fns in the provided order.
    """
    # AR is ffmpeg installed
    ffmpeg = get_ffmpeg()

    # AR delete existing video mp4, if any
    if os.path.isfile(outmp4):
        os.remove(outmp4)

    # AR temporary folder
    tmpoutdir = tempfile.mkdtemp()

    # AR copying files to tmpoutdir
    n_img = len(fns)
    for i in range(n_img):
        _ = os.system("cp {} {}/tmp-{:04d}.png".format(fns[i], tmpoutdir, i))
    if verbose:
        log.info(fns)

    # AR ffmpeg settings
    default_fps = 25.0  # ffmpeg default fps
    pts_fac = "{:.1f}".format(duration / (n_img / default_fps))
    # AR following encoding so that mp4 are displayed in safari, firefox
    cmd = "{} -i {}/tmp-%04d.png -vf 'setpts={}*PTS,crop=trunc(iw/2)*2:trunc(ih/2)*2' -vcodec libx264 -pix_fmt yuv420p {}".format(
        ffmpeg, tmpoutdir, pts_fac, outmp4
    )
    # cmd = "{} -i {}/tmp-%04d.png -vf 'setpts={}*PTS,crop=trunc(iw/2)*2:trunc(ih/2)*2' -pix_fmt yuv420p {}".format(ffmpeg, tmpoutdir, pts_fac, outmp4)
    if not verbose:
        cmd += " >/dev/null 2>&1"
    _ = os.system(cmd)

    # AR deleting temporary tmp*png files
    for i in range(n_img):
        os.remove("{}/tmp-{:04d}.png".format(tmpoutdir, i))


def get_airmasses(decs, has=None, sitename="Kitt Peak"):
    """
    Get airmass value for observations.

    Args:
        decs: numpy array of Decs (degrees)
        has (optional, defaults to 15): HAs for decs (numpy array)
        sitename (optional, defaults to 'Kitt Peak'): astropy site name (str)

    Returns:
        airmasses: numpy array of airmasses (float)
    """
    # AR HAs
    if has is None:
        has = 15.0 + np.zeros(len(decs))

    # AR KPNO
    site_lat = EarthLocation.of_site(sitename).lat.value
    sin_lat = np.sin(np.radians(site_lat))
    cos_lat = np.cos(np.radians(site_lat))

    # AR airmasses
    sin_decs = np.sin(np.radians(decs))
    cos_decs = np.cos(np.radians(decs))
    cos_has = np.cos(np.radians(has))
    airmasses = 1.0 / (sin_decs * sin_lat + cos_decs * cos_lat * cos_has)

    return airmasses


def get_expfacs(decs, ebvs, has=None, sitename="Kitt Peak", alpha=2.165, beta=1.75):
    """
    Compute the EXPFAC for a set of Dec. and EBV.

    Args:
        decs: numpy array of Decs (degrees)
        ebvs: numpy array of EBV
        has (optional, defaults to 15): HAs for decs (numpy array)
        sitename (optional, defaults to 'Kitt Peak'): astropy site name (str)
        alpha (optional, defaults to 2.165): coefficient for EBV (float)
        beta (optional, defaults to 1.75): coefficient for AIRMASS (float)

    Returns:
        expfacs: 10**(2*alpha*ebvs/2.5) * airmasses**beta (float)
    """
    # AR HAs
    if has is None:
        has = 15.0 + np.zeros(len(decs))
    return (
        10 ** (2 * alpha * ebvs / 2.5)
        * get_airmasses(decs, has=has, sitename=sitename) ** beta
    )


def table_read_for_pool(fn, use_fitsio, columns):
    """
    Utility function to read table with multiprocessing.

    Args:
        fn: table file name (str)
        use_fitsio: read with fitsio? (otherwise use astropy.table.Table) (bool)
        columns: only read some columns? (only works if use_fitsio=True) (list of str)

    Returns:
        the read table

    Notes:
        Note that the fitsio reading blanks the header.
    """
    if use_fitsio:
        if columns is None:
            return Table(fitsio.read(fn))
        else:
            return Table(fitsio.read(fn, columns=columns))
    else:
        return Table.read(fn)


def get_mjd(expid, night):
    """
    Get the MJD of an exposure.

    Args:
        expid: the exposure id (int)
        night: the night of the exposure (int)

    Returns:
        mjd: the mjd (float)

    Notes:
        The MJD is read from the SPEC header (MJD-OBS) of the desi-{expid}.fits.fz file.
        It is the MJD of the start of the exposure.
    """

    fn = os.path.join(
        os.getenv("DESI_ROOT"),
        "spectro",
        "data",
        str(night),
        "{:08d}".format(expid),
        "desi-{:08d}.fits.fz".format(expid),
    )
    return fitsio.read_header(fn, "SPEC")["MJD-OBS"]


def get_moon_radecphase(mjds):
    """
    Get the Moon R.A., Dec., and phase for a list of MJDs.

    Args:
        mjds: list of MJD values (float)

    Returns:
        moonras: the R.A. values of the Moon positions (float)
        moondecs: the Dec. values of the Moon positions (float)
        moonphases: the phases (in 0,1) of the Moon (float)

    Notes:
        This is just reading Aaron's file (see get_fns()).
        It covers Jan. 1st 2019 to Jan. 1st 2030.
    Aaron s file covers Jan. 1st 2019 - Jan. 1st 2030.
    Ideally, the input MJDs should be spanning e.g. one night; otherwise it is a bit long to run.
    We could do something a bit smarter with rounding the MJD to e.g. 1h (1/24), as we are not
        chasing here super-detailed values.
    """
    # TODO: smarter coding with e.g. rounding the MJD to 1h (1/24)
    if (~np.isfinite(mjds)).sum() > 0:
        msg = "input mjds have {} np.nan".format((~np.isfinite(mjds)).sum())
        log.error(msg)
        raise ValueError(msg)
    mjd_min, mjd_max = np.min(mjds), np.max(mjds)

    fn = get_fns()["moon"]
    d = Table(fitsio.read(fn))
    if (d["MJD"].min() > mjd_min) | (d["MJD"].max() < mjd_max):
        msg = "{} does not cover the requested MJD range ({}, {})".format(
            mjd_min, mjd_max
        )
        log.error(msg)
        raise ValueError(msg)

    mjd_bin = np.unique(np.diff(d["MJD"]).round(6))[0]
    sel = (d["MJD"] >= mjd_min - mjd_bin) & (d["MJD"] <= mjd_max + mjd_bin)
    d = d[sel]
    ii = [np.abs(d["MJD"] - mjd).argmin() for mjd in mjds]
    d = d[ii]
    return d["MOONRA"], d["MOONDEC"], d["MPHASE"]


def get_speed(d, source):
    """
    Get the exposure speed.

    Args:
        d: Table with the exposures infos (ie reading of exposures-daily.csv)
        source: "spec" or "etc"

    Returns:
        speed: the computed speed (np.array of float)

    Notes:
        We use:
            EBVFAC = 10 ** (2 * 2.165 * EBV / 2.5)
            AIRFAC = AIRMASS ** 1.75
            SPEED = EFFTIME_{ETC,SPEC} * EBVFAC * AIRFAC / EXPTIME
    """
    # source: "spec" or "etc"
    d["EBVFAC"] = 10.0 ** (2 * 2.165 * d["EBV"] / 2.5)
    d["AIRFAC"] = d["AIRMASS"] ** 1.75
    speed = np.zeros(len(d))
    sel = d["EXPTIME"] > 0
    speed[sel] = (
        d["EFFTIME_{}".format(source.upper())][sel]
        * d["EBVFAC"][sel]
        * d["AIRFAC"][sel]
        / d["EXPTIME"][sel]
    )
    return speed
