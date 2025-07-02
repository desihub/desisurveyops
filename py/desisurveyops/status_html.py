#!/usr/bin/env python

# AR general
import os
import textwrap
from datetime import datetime, timedelta
from time import time
from pkg_resources import resource_filename

# AR scientifical
import numpy as np

# AR astropy
from astropy.table import Table, vstack

# AR desisurveyops
from desisurveyops.status_utils import (
    get_filename,
    get_fns,
    get_obsdone_tiles,
    get_programs_npassmaxs,
    get_shutdowns,
    get_history_tiles_infos,
    get_backup_minefftime,
    get_speed,
)

# AR desiutil
from desiutil.log import get_logger

log = get_logger()


def process_html(
    outdir, survey, specprod, specprod_ref, programs, npassmaxs, program_strs
):

    """
    Wrapper function to generate survey status html page.

    Args:
        outdir: output folder (str)
        survey: survey name (str)
        specprod: spectroscopic production (e.g. daily) (str)
        specprod_ref: reference spectroscopic production (e.g. loa) (str)
        programs: list of programs (str)
        npassmaxs: list of npassmaxs (str)
        program_strs: list of program_strs (str)

    Notes:
        For (programs, npassmaxs, program_strs), see desisurveyops.sky_utils.get_programs_npassmaxs().
        Usually use specprod=daily and specprod_ref=loa.
        The overall coding can surely been improved! but it works as is...
    """

    log.info(
        "{}\tBEGIN process_html".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    start = time()

    htmlfn = get_filename(outdir, survey, "html", ext="html")
    cssfn = get_filename(outdir, survey, "html", ext="css")

    # AR need to copy the css?
    git_cssfn = os.path.join(
        resource_filename("desisurveyops", "../../data"), os.path.basename(cssfn)
    )
    if os.path.isfile(cssfn):
        f = open(cssfn, "r").read()
        git_f = open(git_cssfn, "r").read()
        if f != git_f:
            log.warning(
                "{} and {} are different; {} page may look not as expected".format(
                    cssfn, git_cssfn, htmlfn
                )
            )
    else:
        cmd = "cp -p {} {}".format(git_cssfn, cssfn)
        log.info("run {}".format(cmd))
        os.system(cmd)

    # AR exposures
    fns = get_fns(survey=survey, specprod=specprod)
    e = Table.read(fns["spec"]["exps"])
    sel = (e["SURVEY"] == survey) & (e["EFFTIME_SPEC"] > 0)
    e = e[sel]

    # AR obs, donetiles
    obs_tiles, obs_nights, obs_progs, done_tiles = get_obsdone_tiles(survey, specprod)

    html = open(htmlfn, "w")

    # ADM set up the html file and write preamble to it.
    write_html_preamble(
        html,
        "{} Overview Page".format(survey.capitalize()),
        os.path.basename(cssfn),
    )

    # AR collapsibles
    collapsible_names = get_collapsible_names(survey)

    # AR sky/zhist
    for program, npassmax, program_str in zip(programs, npassmaxs, program_strs):

        # AR have we already observed this program?
        sel = obs_progs == program
        if npassmax is not None:
            fns = get_fns(survey=survey, specprod=specprod)
            fn = fns["ops"]["tiles"]
            t = Table.read(fn)
            t = t[t["PASS"] < npassmax]
            sel &= np.in1d(obs_tiles, t["TILEID"])
        log.info("{}\tfound {} observed tiles".format(program_str, sel.sum()))

        # AR handle e.g. bright1b which does not exist yet
        if sel.sum() == 0:
            continue

        # AR section
        html.write(
            "<button type='button' class='{}'><strong>{} program</strong></button>\n".format(
                collapsible_names[program_str],
                program_str.upper(),
            )
        )
        html.write("<div class='content'>\n")

        def _read_tiles(survey, night, rev, program, npassmax, tilesdir):
            fn = get_filename(
                tilesdir, survey, "tiles", night=night, rev=rev, ext="ecsv"
            )
            t = Table.read(fn)
            sel = (t["IN_DESI"]) & (t["PROGRAM"] == program)
            if npassmax is not None:
                sel &= t["PASS"] < npassmax
            t = t[sel]
            return t

        # AR history...
        d = get_history_tiles_infos(survey)
        tilesdir = d.meta["FOLDER"]
        tilesfns = [
            get_filename(
                tilesdir,
                survey,
                "tiles",
                night=d["NIGHT"][i],
                rev=d["REVISION"][i],
                ext="ecsv",
            )
            for i in range(len(d))
        ]
        html.write("\t<p>Program history:</p>\n")
        for i in range(len(d)):
            night, rev = d["NIGHT"][i], d["REVISION"][i]
            comment = d["COMMENT"][i]
            t = _read_tiles(survey, night, rev, program, npassmax, tilesdir)
            # AR handle e.g. BRIGHT1B which is not defined yet
            if len(t) == 0:
                continue
            if i == 0:
                n_add, n_rmv = len(t), 0
            else:
                prev_t = _read_tiles(
                    survey,
                    d["NIGHT"][i - 1],
                    d["REVISION"][i - 1],
                    program,
                    npassmax,
                    tilesdir,
                )
                n_add = (~np.isin(t["TILEID"], prev_t["TILEID"])).sum()
                n_rmv = (~np.isin(prev_t["TILEID"], t["TILEID"])).sum()
            if (n_add != 0) | (n_rmv != 0):
                prev_t = t
                html.write(
                    "\t\t<p>- {} (PASSES={}-{}): {} (added {} tiles, removed {} tiles).</p>\n".format(
                        night, t["PASS"].min(), t["PASS"].max(), comment, n_add, n_rmv
                    )
                )

        # AR initially allowed for case=obs or done
        # AR but just case=obs in the end, so no loop on case, to simplify
        case = "obs"
        if program == "BACKUP":
            caselab = "Observed"
            caselab2 = "EFFTIME_SPEC > {}s".format(get_backup_minefftime())
        else:
            caselab = "Completed"
            caselab2 = "EFFTIME_SPEC > MINTFRAC * GOALTIME"

        # AR loop on ntile, fraccov
        for quant, quantlab, quantlab2 in zip(
            ["ntile", "fraccov"],
            ["nb of tiles", "survey fraction"],
            ["number of tiles", "fraction of the final coverage"],
        ):

            # AR sub-section
            html.write(
                "\t<button style='margin-left:25px;' typ='button' class='{}'><strong>{} tiles</strong>: {} skymap and per-tile n(z)</button>\n".format(
                    collapsible_names["sub{}".format(program_str)],
                    caselab,
                    quantlab,
                )
            )
            html.write("\t<div class='content'>\n")

            # AR sky
            html.write("\t\t<p>We display here {} tiles.</p>\n".format(caselab2))
            html.write(
                "\t\t<p>The color-coding represents the {}.</p>\n".format(quantlab2)
            )
            if case == "obs":
                sel = obs_progs == program
                if sel.sum() == 0:
                    txt = "No {} tiles {} observed so far".format(
                        program_str, caselab.lower()
                    )
                else:
                    lastnight = obs_nights[sel][-1]
                    nlast = ((sel) & (obs_nights == lastnight)).sum()
                    txt = "{} {} tiles {} on {} are highlighted in black.".format(
                        nlast, program_str, caselab.lower(), lastnight
                    )
                html.write("\t\t<p>{}.</p>\n".format(txt))
            html.write("\t\t<br>\n")
            outpng = path_full2web(
                get_filename(
                    outdir,
                    survey,
                    "skymap",
                    program_str=program_str,
                    case=case,
                    quant=quant,
                    night=None,
                    ext="png",
                )
            )
            txt = "<a href='{}'><img SRC='{}' width=55% height=auto></a>".format(
                outpng, outpng
            )
            html.write("\t\t<td> {} </td>\n".format(txt))

            # AR zhist
            # TODO: use a more general way to handle bright1b
            if program in ["DARK", "BRIGHT", "DARK1B"]:
                outpng = path_full2web(
                    get_filename(
                        outdir, survey, "zhist", program_str=program_str, ext="png"
                    )
                )
                txt = "<a href='{}'><img SRC='{}' width=35% height=auto></a>".format(
                    outpng, outpng
                )
                html.write("\t\t<td> {} </td>\n".format(txt))
            html.write("\t\t<a&emsp;></a>\n")
            html.write("\t\t<tr>\n")
            html.write("\t\t</tr>" + "\n")
            html.write("\t\t<br>\n")
            html.write("\t\t</tr>\n")
            html.write("\t</div>\n")
            html.write("\n")

            # AR mp4
            if case == "obs":
                html.write(
                    "\t<button style='margin-left:25px;' type='button' class='{}'><strong>{} tiles</strong>: animated per-night {} skymap</button>\n".format(
                        collapsible_names["sub{}".format(program_str)],
                        caselab,
                        quantlab,
                    )
                )
                html.write("\t<div class='content'>\n")
                html.write(
                    "\t\t<p>The color-coding represents the {}.</p>\n".format(quantlab2)
                )
                html.write("\t\t<br>\n")
                outmp4 = path_full2web(
                    get_filename(
                        outdir,
                        survey,
                        "skymap",
                        program_str=program_str,
                        case=case,
                        quant=quant,
                        ext="mp4",
                    )
                )
                html.write("\t\t<tr>\n")
                html.write("\t\t<video width=55% height=auto controls autoplay loop>\n")
                html.write("\t\t\t<source src='{}' type='video/mp4'>\n".format(outmp4))
                html.write("\t\t</video>\n")
                html.write("\t\t</tr>\n")
                html.write("\t</div>\n")
                html.write("\n")

        # AR coverage per slice of ra / lst
        for racase, racaselab in zip(["complra", "compllst"], ["R.A.", "LST"]):
            html.write(
                "\t<button style='margin-left:25px;' type='button' class='{}'><strong>{} tiles</strong>: completeness per {} slice</button>\n".format(
                    collapsible_names["sub{}".format(program_str)],
                    caselab,
                    racase.replace("compl", ""),
                )
            )
            html.write("\t<div class='content'>\n")
            html.write("\t\t<p>We display here {} tiles.</p>\n".format(caselab2))
            html.write(
                "\t\t<p>The color-coding represents the (weighted) completeness per {} slice of 30 deg.</p>\n".format(
                    racaselab
                )
            )
            html.write(
                "\t\t<p>The corresponding completeness percentages are reported (first line: with no weighting, second line: with weighting).</p>\n"
            )
            html.write("\t\t<br>\n")
            html.write("\t\t<tr>\n")
            outpng = path_full2web(
                get_filename(
                    outdir,
                    survey,
                    "skymap",
                    program_str=program_str,
                    case=racase,
                    ext="png",
                )
            )
            txt = "<a href='{}'><img SRC='{}' width=55% height=auto></a>".format(
                outpng, outpng
            )
            html.write("\t\t<td> {} </td>\n".format(txt))
            html.write("\t\t</tr>\n")
            html.write("\t</div>\n")
            html.write("\n")

        # AR pending tiles
        if program in ["BRIGHT", "DARK", "DARK1B", "BRIGHT1B"]:

            # AR we highlight in the table tiles older than frac_year = 1
            frac_year = 1

            # AR we display at most 4 exposures...
            nexp_max = 3

            # AR read table
            fn = get_filename(
                outdir,
                survey,
                "skymap",
                case="pending",
                program_str=program_str,
                ext="ecsv",
            )
            d = Table.read(fn)
            ii = np.lexsort((d["TILEID"], d["QA"], d["OBSSTATUS"], d["LASTNIGHT"]))
            d = d[ii]
            for key in ["EXPIDS_EXPID", "EXPIDS_NIGHT", "EXPIDS_EFFTIME_SPEC"]:
                for i in range(len(d)):
                    if "mask" in dir(d[key][i]):
                        if d[key].mask[i]:
                            continue
                    x = d[key][i].split(",")
                    if len(x) > nexp_max:
                        d[key][i] = ",".join(x[:nexp_max]) + "+"
                d[key] = d[key].astype(object).astype(str)

            # AR write
            html.write(
                "\t<button style='margin-left:25px;' type='button' class='{}'><strong>Pending tiles</strong></button>\n".format(
                    collapsible_names["sub{}".format(program_str)]
                )
            )
            html.write("\t<div class='content'>\n")
            html.write(
                "\t\t<p>We report here the {} pending tiles as of {}.</p>\n".format(
                    len(d), d.meta["MODTIME"]
                )
            )
            html.write(
                "\t\t<p>We identify those from tiles-main.ecsv with: (IN_DESI=True) & ((STATUS == 'obsstart') | (STATUS == 'obsend').</p>\n"
            )
            html.write(
                "\t\t<p>Tiles with STATUS=='obsstart' are just waiting for more observations.</p>\n"
            )
            html.write(
                "\t\t<p>Tiles with STATUS=='obsend' are waiting for (further) QA.</p>\n"
            )
            html.write(
                "\t\t<p>In the tables below, we report in red tiles older than {} year; and at maximum {} exposures for a given TILEID.</p>\n".format(
                    frac_year, nexp_max
                )
            )
            html.write("\t\t<br>\n")

            # AR image
            outpng = path_full2web(
                get_filename(
                    outdir,
                    survey,
                    "skymap",
                    case="pending",
                    program_str=program_str,
                    ext="png",
                )
            )
            txt = "<a href='{}'><img SRC='{}' width=55% height=auto></a>".format(
                outpng, outpng
            )
            html.write("\t\t<td> {} </td>\n".format(txt))
            html.write("\t\t</tr>\n")

            # AR html table
            for status, txt in zip(
                ["obsend", "obsstart"],
                [
                    "Tiles waiting for (further) QA (STATUS == 'obssend'):",
                    "Tiles waiting for more observations (STATUS == 'obsstart'):",
                ],
            ):
                html.write("\t\t<p>{}</p>\n".format(txt))
                html.write("\t\t<table>\n")
                fields = [
                    "LASTNIGHT",
                    "TILEID",
                    "RA",
                    "DEC",
                    "QA",
                    "EXPTIME",
                    "EFFTIME_SPEC",
                    "EXPIDS_EXPID",
                    "EXPIDS_NIGHT",
                    "EXPIDS_EFFTIME_SPEC",
                ]

                # AR header
                html.write("\t\t\t<tr>\n")
                html.write("\t\t\t</tr>\n")
                html.write("\t\t\t<tr>\n")
                for field in fields:
                    html.write("\t\t\t\t<th>{}</th>\n".format(field))
                html.write("\t\t\t</tr>\n")

                oldnightcut = int(
                    (datetime.now() - timedelta(days=frac_year * 365)).strftime(
                        "%Y%m%d"
                    )
                )

                ii = np.where(d["STATUS"] == status)[0]
                for i in ii:
                    if (d["LASTNIGHT"][i] != 0) & (d["LASTNIGHT"][i] < oldnightcut):
                        color = "red"
                    else:
                        color = "black"
                    for field in fields:
                        html.write(
                            "\t\t\t<td style='color:{};'> {} </td>".format(
                                color, d[field][i]
                            )
                        )
                    html.write("\t</tr>\n")
                html.write("\t\t</table>\n")
                # html.write("\n")
            html.write("\t</div>\n")

        # AR QSO / Lya
        if (program == "DARK") | (program == "DARK1B"):
            html.write(
                "\t<button style='margin-left:25px;' type='button' class='{}'><strong>QSO / Lya diagnoses</strong></button>\n".format(
                    collapsible_names["sub{}".format(program_str)]
                )
            )
            html.write("\t<div class='content'>\n")
            html.write(
                "\t\t<p>The reported densities are corrected by the assigned fraction (typically 50% for QSOs in a first pass) i.e., those are Nobs / tile_area / frac_assign.</p>\n"
            )
            html.write(
                "\t\t<p>The considered assigned fraction is the number of *first* time observed QSOs divided by the number of QSO targets (i.e., it will decrease with increasing coverage).</p>\n"
            )
            html.write("\t\t<br>\n")
            outpng = path_full2web(
                get_filename(outdir, survey, "qso", program_str=program_str, ext="png")
            )
            txt = "<a href='{}'><img SRC='{}' width=55% height=auto></a>".format(
                outpng, outpng
            )
            html.write("\t\t<td> {} </td>\n".format(txt))
            html.write("\t\t</tr>\n")
            html.write("\t</div>\n")

        # AR vertical space
        html.write("</div>\n")
        html.write("\n")

    # AR Number of spectra
    yrs = obs_nights // 10000
    unq_yrs = np.unique(yrs)
    tmpd = Table.read(get_filename(outdir, survey, "nspec", ext="ecsv"))
    assert tmpd.meta["SURVEY"] == survey
    assert tmpd.meta["PROD"] == specprod
    assert tmpd.meta["REFPROD"] == specprod_ref
    dchi2min = tmpd.meta["DCHI2MIN"]
    tmpsel = (tmpd["PROGRAM"] == "ALL") & (tmpd["NAME"] == "ALL")
    nspectot = tmpd["UNQ_ALL"][tmpsel][-1]
    ntiletot = tmpd["NTILE"][tmpsel][-1]
    html.write(
        "<button type='button' class='collapsible'><strong>Number of spectra</strong> ({:.1f}M with {} tiles over {} nights)</button>\n".format(
            1e-6 * nspectot, ntiletot, np.unique(e["NIGHT"]).size
        )
    )
    html.write("<div class='content'>\n")
    html.write(
        "\t<p>We report here the number of observed spectra with a valid fiber (left: BRIGHT{1B} tiles; middle: DARK{1B} tiles; right: BACKUP+BRIGHT{1B}+DARK{1B} tiles).</p>\n"
    )
    html.write(
        "\t<p>We identify the Lya's with PRIORITY=3350 or (PRIORITY=3400 and ((Z>2.1) or (Z_QN>2.1 and IS_QSO_QN=1))).</p>\n"
    )
    html.write(
        "\t<p>Except for the Lya's we restrict to redrock DELTACHI2 > {} and ZWARN=0 spectra.</p>\n".format(
            dchi2min
        )
    )
    html.write(
        "\t<p>We use the {} reduction if available, the daily else.</p>\n".format(
            specprod_ref
        )
    )
    html.write("\t<p>Uniqueness identification is based on TARGETID.</p>\n")
    html.write(
        "\t<p>Colored lines are for unique spectra per target type, black lines are for unique spectra per spectral kind: Galactic (c*Z < 600 km/s), extra-Galactic (c*Z > 600 km/s), Lya (see above).</p>\n"
    )
    html.write(
        "\t<p>Note that there can be some overlap between targets, inside a program (e.g. ELG and QSO) or across programs (e.g. BGS and LRG, or standard stars).</p>\n"
    )

    # AR shutdowns
    start_nights, end_nights, comments = get_shutdowns(survey)
    html.write("\t<p></p>\n")
    html.write("\t<p>Shutdowns:</p>\n")
    for start_night, end_night, comment in zip(start_nights, end_nights, comments):
        tmp_start = datetime.strptime(str(start_night), "%Y%m%d")
        tmp_end = datetime.strptime(str(end_night), "%Y%m%d")
        ndays = (tmp_end - tmp_start).days
        html.write(
            "\t\t<p>- {} - {} ({} days): {}\n".format(
                start_night, end_night, ndays, comment,
            )
        )
    html.write("\t<p></p>\n")
    html.write("\t<br>\n")
    outpng = path_full2web(get_filename(outdir, survey, "nspec", ext="png"))
    txt = "<a href='{}'><img SRC='{}' width=100% height=auto></a>".format(
        outpng, outpng
    )
    html.write("\t\t<td> {} </td>\n".format(txt))
    html.write("\t</tr>\n")
    html.write("</div>\n")
    html.write("\n")

    # AR Focal plane state
    html.write(
        "<button type='button' class='collapsible'><strong>Focal plane state</strong> ({} nights)</button>\n".format(
            np.unique(e["NIGHT"]).size
        )
    )
    html.write("<div class='content'>\n")
    html.write("\t<br>\n")
    outmp4 = path_full2web(get_filename(outdir, survey, "fpstate", ext="mp4"))
    outpng = path_full2web(get_filename(outdir, survey, "fpstate", ext="png"))
    html.write(
        "\t<p>We show here the focal plane state as of {}.</p>\n".format(lastnight)
    )
    html.write(
        "\t<p>An animated movie since the start of the survey is available at this <a href='{}' target='external'> link.</p>\n".format(
            outmp4
        )
    )
    html.write(
        "\t\t<td> <a href='{}'><img SRC='{}' width=75% height=auto></a> </td>\n".format(
            outpng, outpng
        )
    )
    html.write("\t<tr>\n")
    html.write("</div>\n")
    html.write("\n")

    # AR Cumulative observing conditions
    html.write(
        "<button type='button' class='collapsible'><strong>Cumulative observing conditions</strong> ({} nights)</button>\n".format(
            np.unique(e["NIGHT"]).size
        )
    )
    html.write("<div class='content'>\n")
    html.write("\t<p>The last observing night is displayed in dashed lines.</p>\n")
    html.write("\t<br>\n")
    outpng = path_full2web(
        get_filename(outdir, None, "obsconds", case="cumulative", ext="png")
    )
    txt = "<a href='{}'><img SRC='{}' width=80% height=auto></a>".format(outpng, outpng)
    html.write("\t\t<td> {} </td>\n".format(txt))
    html.write("\t</tr>\n")
    html.write("</div>\n")
    html.write("\n")

    # AR Individual per-night html pages with exposure properties
    tmp_e = e.copy()
    sel = tmp_e["SKY_MAG_R_SPEC"] > 99
    tmp_e["SKY_MAG_R_SPEC"][sel] = 99.9
    write_html_perexp(outdir, survey, specprod, specprod_ref, tmp_e)

    # AR Per-night observing sequence, observing conditions, sframesky, tileqa, spacewatch
    html.write(
        "<button type='button' class='collapsible'><strong>Per-night products</strong> ({} nights)</button>\n".format(
            np.unique(e["NIGHT"]).size
        )
    )
    txts = get_pernight_menu(
        outdir, ["spacewatch", "skyseq", "obsconds", "exposures"], np.unique(obs_nights)
    )
    for txt in txts:
        html.write(txt)

    html.write("\n")

    # AR Expid / tileid finder
    """
    html.write(
        "<button type='button' class='collapsible'><strong>Expid - tileid finder</strong></button>\n"
    )
    html.write("<div class='content'>\n")
    html.write("\t</tr>\n")
    html.write("</div>\n")
    html.write("\n")
    """

    # AR used files
    html.write("\t<p></p>\n")
    html.write(
        "<button type='button' class='collapsible'><strong>Used files</strong></button>\n"
    )
    html.write("<div class='content'>\n")
    fns = get_fns(survey=survey, specprod=specprod)
    for fn, descr in zip(
        [
            fns["ops"]["tiles"],
            fns["spec"]["tiles"],
            fns["ops"]["status"],
            fns["spec"]["exps"],
            fns["spec"]["arxivdir"],
            fns["spec"]["cumuldir"],
        ],
        [
            "tiling file",
            "observed tiles",
            "spectroscopic status of each tile",
            "file with all exposures",
            "archive tile reductions",
            "cumulative tile reductions",
        ],
    ):
        html.write(
            "\t<p><a href='{}' target='external'> {} </a> : {}. </p>\n".format(
                path_full2web(fn),
                os.path.basename(fn),
                descr,
            )
        )
    html.write("\t</tr>\n")
    html.write("</div>\n")
    html.write("\n")

    # AR lines to make collapsing sections
    for collapsible in [
        "collapsible",
        "collapsible_sub",
        "collapsible_year",
        "collapsible_month",
    ] + [collapsible_names[program] for program in collapsible_names]:
        write_html_collapse_script(html, collapsible)

    # ADM html postamble for main page.
    write_html_today(html)
    html.write("</html></body>\n")
    html.close()

    log.info(
        "{}\tEND process_html (took {:.1f}s)".format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), time() - start
        )
    )


def path_full2web(fn):
    """
    Convert full path to web path (needs DESI_ROOT to be defined).

    Args:
        fn: filename full path (str)

    Returns:
        Web path (str)
    """
    return fn.replace(os.getenv("DESI_ROOT"), "https://data.desi.lbl.gov/desi")


def _javastring():
    """
    Return a string that embeds a date in a webpage.

    Notes:
        Credits to ADM (desitarget/QA.py).
    """
    js = textwrap.dedent(
        """
    <SCRIPT LANGUAGE="JavaScript">
    var months = new Array(13);
    months[1] = "January";
    months[2] = "February";
    months[3] = "March";
    months[4] = "April";
    months[5] = "May";
    months[6] = "June";
    months[7] = "July";
    months[8] = "August";
    months[9] = "September";
    months[10] = "October";
    months[11] = "November";
    months[12] = "December";
    var dateObj = new Date(document.lastModified)
    var lmonth = months[dateObj.getMonth() + 1]
    var date = dateObj.getDate()
    var fyear = dateObj.getYear()
    if (fyear < 2000)
    fyear = fyear + 1900
    if (date == 1 || date == 21 || date == 31)
    document.write(" " + lmonth + " " + date + "st, " + fyear)
    else if (date == 2 || date == 22)
    document.write(" " + lmonth + " " + date + "nd, " + fyear)
    else if (date == 3 || date == 23)
    document.write(" " + lmonth + " " + date + "rd, " + fyear)
    else
    document.write(" " + lmonth + " " + date + "th, " + fyear)
    </SCRIPT>
    """
    )
    return js


def write_html_today(html):
    """
    Write in an html object today's date.

    Args:
        html: html file object.
    """
    html.write(
        "<p style='font-size:1vw; text-align:right'><i>Last updated: {}</p></i>\n".format(
            _javastring()
        )
    )


# AR html per-night observing sequence, observing conditions, sframesky, tileqa, exposures
def write_html_night(html, outdir, case, e):
    """
    Write in an html object the per-night observing conditions section.

    Args:
        html: html file object
        outdir: output folder name (str)
        case: "skyseq", "obsconds" or "sframesky", "tileqa", "exposures" (str)
        e: Table with exposures (i.e. reading of exposures-daily.csv)
    """
    #
    if case == "skyseq":
        fmt = "png"
    elif case in ["sframesky", "tileqa"]:
        fmt = "pdf"
    elif case in ["exposures"]:
        fmt = "html"
    else:
        fmt = "mp4"
    print(case, fmt)
    #
    # year_word = "<span class='blue'><strong>YEAR</strong></span>"
    # month_word = "<span class='green'><strong>MONTH</strong></span>"
    # html.write(
    #    "\t<p>First click on the {} to access the list of the months for that year; ".format(year_word)+
    #    "then click on the {} to access the list of nights for that month; ".format(month_word)+
    #    "then click on the NIGHT to access the {} for that night.</a></p>\n".format(fmt)
    # )
    # AR per year
    years = np.unique(e["NIGHT"] // 10000)
    month_ncol = 10
    for year in years:
        isyear = e["NIGHT"] // 10000 == year
        # AR per month
        months = np.unique(e["NIGHT"][isyear] // 100)
        night_ncol = 10
        html.write(
            "\t<button type='button' class='collapsible_year'><strong>{}</strong> ({} nights)</button>\n".format(
                year,
                np.unique(e["NIGHT"][isyear]).size,
            )
        )
        html.write("\t<div class='content'>\n")
        month_col = 0
        for month in months:
            if case == "exposures":
                yyyymm = "{}{:02}".format(year, month)
            else:
                yyyymm = None
            nights = np.unique(e["NIGHT"][e["NIGHT"] // 100 == month])
            html.write(
                "\t\t<button style='margin-left:25px;' type='button' class='collapsible_month'><strong>{}</strong> ({} nights)</button>\n".format(
                    month,
                    nights.size,
                )
            )
            html.write("\t\t<div class='content'>\n")
            # AR building array
            html.write("\t\t\t<table style='margin-left:50px;' cellspacing='10px;'>\n")
            night_col = 0
            for night in nights:
                html.write(
                    "\t\t\t\t<td><a href='{}' target='external'> {} </td>\n".format(
                        path_full2web(
                            get_filename(
                                "{}{}".format(case, fmt),
                                outdir,
                                night=night,
                                yyyymm=yyyymm,
                            )
                        ),
                        night,
                    )
                )
                night_col += 1
                #
                if (night_col == night_ncol) | (night == nights[-1]):
                    html.write("\t\t\t\t</tr>\n")
                    night_col = 0
            html.write("\t\t\t</table>\n")
            html.write("\t\t</div>\n")
        html.write("\t</div>\n")
    html.write("\t</div>\n")
    html.write("\n")


# AR collapsible
def get_collapsible_names(survey):
    """
    CSS collapsible names used.

    Args:
        survey: survey name (str)

    Returns:
        collapsible_names: a dictionary with the infos (dict)
    """
    collapsible_names = {}

    programs, npassmaxs, program_strs = get_programs_npassmaxs(survey=survey)
    for program_str in program_strs:
        collapsible_names[program_str] = "collapsible_{}".format(program_str.lower())
        collapsible_names["sub{}".format(program_str)] = "collapsible_sub{}".format(
            program_str.lower()
        )

    return collapsible_names


# ADM html preamble
def write_html_preamble(html, title, cssfn):
    """
    Write the preamble information for the survey status page.

    Args:
        html: the html object
        title: the page title (str)
        cssfn: path to the *.css file (str)
    """
    html.write("<html><body>\n")
    html.write("<h1>{}</h1>\n".format(title))
    html.write("\n")
    #
    html.write("<head>\n")
    html.write("\t<meta charset='UTF-8'>\n")
    html.write("\t<meta http-equiv='X-UA-Compatible' content='IE=edge'>\n")
    html.write(
        "\t<meta name='viewport' content='width=device-width, initial-scale=1.0'>\n"
    )
    # https://stackoverflow.com/questions/49547/how-do-we-control-web-page-caching-across-all-browsers
    html.write(
        "\t<meta http-equiv='Cache-Control' content='no-cache, no-store, must-revalidate'>\n"
    )
    html.write("\t<meta http-equiv='Pragma' content='no-cache'>\n")
    html.write("\t<meta http-equiv='Expires' content='0'>\n")

    html.write("\t<link rel='stylesheet' href='{}'>\n".format(cssfn))
    html.write("</head>\n")
    html.write("\n")
    html.write("<body>\n")
    html.write("\n")


def create_perexp_dict_survey(
    outdir, survey, specprod, specprod_ref, e, fields, tilesfn=None
):
    """
    Create a dictionary with per-exposure informations, used for the survey status page.

    Args:
        outdir: output folder name (str)
        survey: survey name (str)
        specprod: spectroscopic production (e.g. daily) (str)
        specprod_ref: reference spectroscopic production (e.g. loa) (str)
        e: Table with exposures (i.e. reading of exposures-daily.csv)
        fields: (ordered) list of column header names (list)
        tilesfn (optional, defaults to tiles-{survey}.ecsv in $DESI_SURVEYOPS): path to tiles-{survey}.ecsv (str)

    Returns:
        expdict: dictionary with text-formatted values for keys to report in the table (dict)
        expids: np.array of exposures (float)
        nights: np.array of night of each exposure (float)

    Notes:
        Called by write_html_perexp().
        Outputs used by write_html_perexp_table().
    """
    # tilesfn
    if tilesfn is None:
        tilesfn = os.path.join(
            os.getenv("DESI_SURVEYOPS"), "ops", "tiles-{}.ecsv".format(survey)
        )
    # AR cutting e on the survey tiles
    tiles = Table.read(tilesfn)
    e = e[np.isin(e["TILEID"], tiles["TILEID"])]
    # AR redux path
    redux_path = path_full2web(
        os.path.join(os.getenv("DESI_ROOT"), "spectro", "redux", specprod)
    )
    #
    expdict = {}
    nexp = len(e)
    # AR point to specprod(=daily) as we deal here with per-night reduxs
    for field in fields:
        # AR tileid
        if field == "TILEID":
            expdict["TILEID"] = [
                "<a href='{}' target='external'> {}".format(
                    os.path.join(
                        redux_path,
                        "tiles",
                        "cumulative",
                        str(tileid),
                        str(night),
                        "tile-qa-{}-thru{}.png".format(tileid, night),
                    ),
                    tileid,
                )
                for tileid, night in zip(e["TILEID"], e["NIGHT"])
            ]
        # AR night
        if field == "NIGHT":
            expdict["NIGHT"] = [
                "<a href='{}' target='external'> {}".format(
                    os.path.join(redux_path, "exposures", "{}".format(night)), night
                )
                for night in e["NIGHT"]
            ]
        # AR expid
        if field == "EXPID":
            expdict["EXPID"] = [
                "<a href='{}' target='external'> {}".format(
                    os.path.join(
                        "exposures", "{}".format(night), "{:08}".format(expid)
                    ),
                    expid,
                )
                for expid, night in zip(e["EXPID"], e["NIGHT"])
            ]
        # AR nightwatch
        if field == "NIGHT":
            expdict["NIGHTWATCH"] = [
                "<a href='{}' target='external'> {}".format(
                    os.path.join(
                        "https://nightwatch.desi.lbl.gov/",
                        "{}".format(night),
                        "{:08}".format(expid),
                        "qa-summary-{:08}.html".format(expid),
                    ),
                    "Nightwatch",
                )
                for expid, night in zip(e["EXPID"], e["NIGHT"])
            ]
        # sframesky
        if field == "SFRAMESKY":
            expdict["SFRAMESKY"] = [
                "<a href='{}' target='external'> {}".format(
                    os.path.join(
                        redux_path,
                        "nightqa",
                        str(night),
                        "sframesky-{}.pdf".format(night),
                    ),
                    "sframesky",
                )
                for night in e["NIGHT"]
            ]
        # AR faprgrm
        if field == "FAPRGRM":
            expdict["FAPRGRM"] = e["FAPRGRM"].astype(str)
        # AR sky
        if field == "SKY_MAG_R_SPEC":
            expdict["SKY_MAG_R_SPEC"] = [
                "{:.1f}".format(_) for _ in e["SKY_MAG_R_SPEC"]
            ]
        # AR gfa quantities: airmass, transparency, seeing, fiber_fracflux
        if field in [
            "AIRMASS_GFA",
            "TRANSPARENCY_GFA",
            "SEEING_GFA",
            "FIBER_FRACFLUX_GFA",
        ]:
            expdict[field] = ["{:.2f}".format(_) for _ in e[field]]
        # AR exptime, efftime_spec
        if field in ["EXPTIME", "EFFTIME_SPEC"]:
            expdict[field] = ["{:.0f}".format(_) for _ in e[field]]
        # AR speed
        if field == "SPEED_ETC":
            expdict[field] = ["{:.2f}".format(_) for _ in get_speed(e, "etc")]
        if field == "SPEED_SPEC":
            expdict[field] = ["{:.2f}".format(_) for _ in get_speed(e, "spec")]
    #
    return expdict, e["EXPID"], e["NIGHT"]


# AR html per-exposure table
def write_html_perexp_table(
    outdir,
    survey,
    specprod,
    specprod_ref,
    ref_night,
    fields,
    expdict,
    expids,
    nights,
    txts=None,
):
    """
    Write an html file for one night, with table with exposures properties.

    Args:
        outdir: output folder name (str)
        survey: survey name (str)
        specprod: spectroscopic production (e.g. daily) (str)
        specprod_ref: reference spectroscopic production (e.g. loa) (str)
        ref_night: night to consider (int)
        fields: list of columns to be written (in the correct order) (list or numpy array)
        expdict: dictionary with at least fields entries (and each key is a list or a numpy array with N(exposures))
        expids: np.array of exposures (float)
        nights: np.array of night of each exposure (float)
        txts (optional, defaults to None): list of text strings to be printed before the table (list or numpy array)

    Notes:
        Called by write_html_perexp().
        expdict, expids, nights from create_perexp_dict_survey().
    """
    #
    htmlfn = get_filename(outdir, None, "exposures", night=ref_night, ext="html")
    css_basefn = os.path.basename(get_filename(outdir, survey, "html", ext="css"))
    html = open(htmlfn, "w")
    # ADM set up the html file and write preamble to it.
    write_html_preamble(
        html, "Exposures for {}".format(ref_night), "../{}".format(css_basefn)
    )

    if txts is not None:
        for txt in txts:
            html.write("<p>{}</p>\n".format(txt))
    html.write("<table>\n")
    # AR night header
    html.write("\t<tr>\n")
    html.write("\t</tr>\n")
    html.write("\t<tr>\n")
    # AR removing "_GFA" from table header names
    for field in fields:
        # AR hyperlink
        if field == fields[0]:
            txt = "<a id='exposures{}' href='#exposures{}'></a> {}".format(
                ref_night, ref_night, field.replace("_GFA", "")
            )
        elif field == "FIBER_FRACFLUX_GFA":
            txt = "FFRAC"
        elif field == "EFFTIME_SPEC":
            txt = "EFFTIME"
        elif field == "SKY_MAG_R_SPEC":
            txt = "SKY_R"
        else:
            txt = field.replace("_GFA", "")
        html.write("\t\t<th>{}</th>\n".format(txt))
    html.write("\t</tr>\n")
    html.write("\t<tr>\n")
    # AR writing table
    ii = np.where(nights == ref_night)[0]
    ii = ii[expids[ii].argsort()]
    for i in ii:
        for field in fields:
            html.write("\t\t<td> {} </td>\n".format(expdict[field][i]))
        html.write("\t</tr>\n")
    html.write("</table>\n")
    html.write("\n")
    html.write("</html></body>\n")
    html.close()


# AR html per-exposures (one html file per night)
def write_html_perexp(outdir, survey, specprod, specprod_ref, e, tilesfn=None):
    """
    Write an html file for each night, with exposures properties.

    Args:
        outdir: output folder name (str)
        survey: survey name (str)
        specprod: spectroscopic production (e.g. daily) (str)
        specprod_ref: reference spectroscopic production (e.g. loa) (str)
        survey: "main", "sv1" or "sv3" (string)
        e: Table with exposures (i.e. reading of exposures-daily.csv)
        tilesfn (optional, defaults to tiles-{survey}.ecsv in $DESI_SURVEYOPS): path to tiles-{survey}.ecsv (str)

    Notes:
        Calls create_perexp_dict_survey() and write_html_perexp_table().
    """

    fns = get_fns(survey=survey, specprod=specprod)
    # AR tilesfn
    if tilesfn is None:
        tilesfn = fns["ops"]["tiles"]

    # AR cut e on the survey tiles
    tiles = Table.read(tilesfn)
    e = e[np.isin(e["TILEID"], tiles["TILEID"])]
    #
    # AR prepare the table inputs
    # AR list of fields to be reported in the table, in the correct order
    if survey == "main":
        fields = [
            "EXPID",
            "NIGHT",
            "TILEID",
            "FAPRGRM",
            "NIGHTWATCH",
            "SFRAMESKY",
            "EXPTIME",
            "EFFTIME_SPEC",
            "AIRMASS_GFA",
            "TRANSPARENCY_GFA",
            "SEEING_GFA",
            "FIBER_FRACFLUX_GFA",
            "SKY_MAG_R_SPEC",
            "SPEED_ETC",
            "SPEED_SPEC",
        ]
    expdict, expids, nights = create_perexp_dict_survey(
        outdir, survey, specprod, specprod_ref, e, fields
    )
    # AR then create a table with all exposures
    txts = [
        "Column content:",
        "- EXPID: link to the https://data.desi.lbl.gov/desi/spectro/redux/daily/exposures/{NIGHT}/{EXPID} folder;",
        "- NIGHT: link to the https://data.desi.lbl.gov/desi/spectro/redux/daily/exposures/{NIGHT} folder;",
        "- TILEID: link to the https://data.desi.lbl.gov/desi/spectro/redux/daily/tiles/cumulative/{TILEID}/{NIGHT}/tile-qa-{TILEID}-thru{NIGHT}.png QA plot;",
        "- NIGHTWATCH: link to https://nightwatch.desi.lbl.gov/{NIGHT}/{EXPID}/qa-summary-{EXPID}.html;",
        "- SFRAMESKY: link to the https://data.desi.lbl.gov/desi/spectro/redux/daily/nightqa/{NIGHT}/sframesky-{NIGHT}.pdf file, with sframe images for sky fibers only.",
        "- FFRAC = FIBER_FRACFLUX: fraction of light in a 1.52 arcsec diameter fiber-sized aperture given the PSF shape, assuming that the PSF is perfectly aligned with the fiber (i.e. does not capture any astrometry/positioning errors).",
        "- AIRMASS, TRANSPARENCY, SEEING, FIBER_FRACFLUX: from GFA.",
        "- SKY_R = SKY_MAG_R_SPEC: spectroscopic sky, corrected for throughput, convolved with DECam r-band filter.",
        "- EFFTIME = EFFTIME_SPEC: based on TSNR2.",
        "- SPEED = EFFTIME * EBVFAC * AIRFAC / EXPTIME, with EBVFAC =  10.0 ** (2 * 2.165 * EBV / 2.5) and AIRFAC = AIRMASS ** 1.75.",
    ]
    for ref_night in np.unique(nights):
        write_html_perexp_table(
            outdir,
            survey,
            specprod,
            specprod_ref,
            ref_night,
            fields,
            expdict,
            expids,
            nights,
            txts=txts,
        )


def write_html_collapse_script(html, classname):
    """
    Write the required lines to have the collapsing sections working.

    Args:
        html: an html file object.
        classname: "collapsible" or "collapsible_month" (str)
    """
    html.write("<script>\n")
    html.write("var coll = document.getElementsByClassName('{}');\n".format(classname))
    html.write("var i;\n")
    html.write("\n")
    html.write("for (i = 0; i < coll.length; i++) {\n")
    html.write("\tcoll[i].addEventListener('click', function() {\n")
    html.write(
        "\t\tthis.classList.toggle('{}');\n".format(
            classname.replace("collapsible", "active")
        )
    )
    html.write("\t\tvar content = this.nextElementSibling;\n")
    html.write("\t\tif (content.style.display === 'block') {\n")
    html.write("\t\tcontent.style.display = 'none';\n")
    html.write("\t\t} else {\n")
    html.write("\t\t\tcontent.style.display = 'block';\n")
    html.write("\t\t}\n")
    html.write("\t});\n")
    html.write("}\n")
    html.write("</script>\n")


def get_pernight_menu(outdir, cases, nights):
    """
    Get the text to write in html to have the clickable menu.

    Args:
        outdir: output folder (str)
        cases: list "first level" cases (str)
        nights: list of all nights (str)

    Returns:
        txts: a list of strings to print (str)

    Notes:
        cases can be: ["spacewatch", "skyseq", "obsconds", "exposures"].
    """

    # AR night infos
    my_yyyys = np.array([str(night // 10000) for night in nights])
    my_mms = np.array(["{:02d}".format((night % 10000) // 100) for night in nights])
    my_dds = np.array(["{:02d}".format(night % 100) for night in nights])
    last_yyyy, last_mm, last_dd = my_yyyys[-1], my_mms[-1], my_dds[-1]

    # AR allowed cases
    allowed_cases = ["spacewatch", "skyseq", "obsconds", "exposures"]
    if not np.all(np.isin(cases, allowed_cases)):
        msg = "cases={} not all in allowed_cases={}".format(
            ",".join(cases), ",".join(allowed_cases)
        )
        log.error(msg)
        raise ValueError(msg)

    # AR initialize
    txts = [
        "<div class='content'>\n",
        "\t<p>\n",
        "\t\tProduct:\n",
        "\t\t<select id='product'>\n",
    ]
    # AR per-case box
    for case in cases:
        txts += [
            "\t\t\t<option value='{}'>{}</option>\n".format(case, case),
        ]

    # AR yyyy, mm dd boxes
    txts += [
        "\t\t</select>\n",
        "\t\t&nbsp\n",
        "\t\t<select name='yyyy' id='yyyy'>\n",
        "\t\t\t<option value='' selected='selected'>yyyy</option>\n",
        # "\t<option value='{}' selected='selected'>{}</option>\n".format(last_yyyy, last_yyyy),
        "\t\t</select>\n",
        "\t\t<select name='mm' id='mm'>\n",
        "\t\t\t<option value='' selected='selected'>mm</option>\n",
        # "\t<option value='{}' selected='selected'>{}</option>\n".format(last_mm, last_mm),
        "\t\t</select>\n",
        "\t\t<select name='dd' id='dd'>\n",
        "\t\t\t<option value='' selected='selected'>dd</option>\n",
        # "\t<option value='{}' selected='selected'>{}</option>\n".format(last_dd, last_dd),
        "\t\t</select>\n",
        "<script type='text/javascript'>\n",
        "var yyyyObject = {\n",
    ]
    for yyyy in np.unique(my_yyyys)[::-1]:
        txts += ["\t'{}': {{\n".format(yyyy)]
        yyyy_sel = my_yyyys == yyyy
        for mm in np.unique(my_mms[yyyy_sel]):
            yyyymm_sel = (yyyy_sel) & (my_mms == mm)
            dds = my_dds[yyyymm_sel]
            dds = ["'{}'".format(_) for _ in dds]
            txts += ["\t\t'{}': [{}],\n".format(mm, ", ".join(dds))]
        txts += ["\t},\n"]
    txts += ["}\n"]

    # AR now the function
    txts += [
        "window.onload = function() {\n",
        "\tvar yyyySel = document.getElementById('yyyy');\n",
        "\tvar mmSel = document.getElementById('mm');\n",
        "\tvar ddSel = document.getElementById('dd');\n",
        "\tfor (var x in yyyyObject) {\n",
        "\t\tyyyySel.options[yyyySel.options.length] = new Option(x, x);\n",
        "\t}\n",
        "\tyyyySel.onchange = function() {\n",
        "\tddSel.length = 1;\n",
        "\tmmSel.length = 1;\n",
        "\tvar mm = Object.keys(yyyyObject[yyyySel.value]);\n",
        "\tmm.sort();\n",
        "\tfor (var i = 0; i < mm.length; i++) {\n",
        # "\tfor (var i = mm.length; i > 0; i--) {\n",
        "\tmmSel.options[mmSel.options.length] = new Option(mm[i], mm[i]);\n",
        "\t\t}\n",
        "\t}\n",
        "\tmmSel.onchange = function() {\n",
        "\tddSel.length = 1;\n",
        "\tvar z = yyyyObject[yyyySel.value][this.value];\n",
        "\tfor (var i = 0; i < z.length; i++) {\n",
        # "\tfor (var i = z.length; i > 0; i--) {\n",
        "\t\tddSel.options[ddSel.options.length] = new Option(z[i], z[i]);\n",
        "\t\t}\n",
        "\t}\n",
        "}\n",
        "</script>\n",
    ]

    # AR now the action
    txts += [
        "<button onclick='getnight()' id='myButton' class='btn request-callback' > Go! </button>\n",
        "</p>\n",
        "<p id='link'></p>\n",
        "<p> <span class='output'></span> </p>\n",
        "<script type='text/javascript'>\n",
        "function getnight() {\n",
        "\tvar x;\n",
        "\tvar yyyy;\n",
        "\tvar mm;\n",
        "\tvar dd;\n",
        "\tvar ext;\n",
        "\tproduct = document.querySelector('#product').value;\n",
        "\tyyyy = document.querySelector('#yyyy').value;\n",
        "\tmm = document.querySelector('#mm').value;\n",
        "\tdd = document.querySelector('#dd').value;\n",
        "\tif (product == 'spacewatch'){\n",
        "\t ext = 'mp4';\n",
        "\t} else if (product == 'skyseq') {\n",
        "\text = 'png';\n",
        "\t} else if ((product == 'obsconds') || (product == 'sframesky') || (product == 'tileqa')){\n",
        "\text = 'pdf';\n",
        "\t} else {\n",
        "\text = 'html';\n",
        "\t}\n",
        "\tx = '{}/' + product + '/' + product + '-' + yyyy + mm + dd + '.' + ext\n".format(
            path_full2web(outdir)
        ),
        "document.getElementById('link').innerHTML = 'Opening ' + x;\n",
        "\twindow.open(x);\n",
        "}\n",
        "</script>\n",
    ]

    # AR close
    txts += [
        "\t</tr>\n",
        "</div>\n",
    ]

    return txts
