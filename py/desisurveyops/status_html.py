#!/usr/bin/env python

# AR general
import os
import textwrap
from datetime import datetime, timedelta
from time import time

# AR scientifical
import numpy as np

# AR astropy
from astropy.table import Table, vstack

# AR desisurveyops
from desisurveyops.status_utils import (
    get_filename,
    get_fns,
    get_obsdone_tiles,
    get_programs_passparams,
    get_shutdowns,
    get_history_tiles_infos,
    get_history_tiles_dir,
    get_backup_minefftime,
    get_speed,
)

# AR desiutil
from desiutil.log import get_logger

log = get_logger()


def process_html(
    outdir, survey, specprod, specprod_ref, programs, skip_passes, program_strs
):

    """
    Wrapper function to generate survey status html page.

    Args:
        outdir: output folder (str)
        survey: survey name (str)
        specprod: spectroscopic production (e.g. daily) (str)
        specprod_ref: reference spectroscopic production (e.g. loa) (str)
        programs: list of programs (str)
        skip_passes: passes to skip in each program (np.ndarray of ints)
        program_strs: list of program_strs (str)

    Notes:
        For (programs, skip_passes, program_strs), see desisurveyops.sky_utils.get_programs_passparams().
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
        get_history_tiles_dir(), os.path.basename(cssfn)
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

    # AR function names
    perprog_func_name = "perprog_func"
    pernight_func_name = "pernight_func"

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


    # AR we only display full tabs for BACKUP, BRIGHT1B, DARK1B to keep a minimal number of tabs
    # AR for other programs, the clickable menu will allow one to check any desired plot "on demand"
    # AR TODO: when pr488 will be merged, we will display: BACKUP, BRIGHT_RUN1B, DARK_RUN1B
    sel2show = np.isin(program_strs, ["BACKUP", "BRIGHT1B", "DARK1B"])

    # AR sky/zhist/qso/pending
    for program, skip_pass, program_str in zip(programs[sel2show], skip_passes[sel2show], program_strs[sel2show]):

        # AR have we already observed this program?
        sel = obs_progs == program
        # if npassmax is not None:
        #     fns = get_fns(survey=survey, specprod=specprod)
        #     fn = fns["ops"]["tiles"]
        #     t = Table.read(fn)
        #     t = t[t["PASS"] < npassmax]
        #     sel &= np.isin(obs_tiles, t["TILEID"])

        if skip_pass is not None:
            fn = fns["ops"]["tiles"]
            t = Table.read(fn)
            t = t[~np.isin(t["PASS"], skip_pass)]
            sel &= np.isin(obs_tiles, t["TILEID"])
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

        def _read_tiles(survey, night, rev, program, skip_pass, tilesdir):
            fn = get_filename(
                tilesdir, survey, "tiles", night=night, rev=rev, ext="ecsv"
            )
            t = Table.read(fn)
            sel = (t["IN_DESI"]) & (t["PROGRAM"] == program)
            if skip_pass is not None:
                sel &= ~np.isin(t["PASS"], skip_pass)
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
            t = _read_tiles(survey, night, rev, program, skip_pass, tilesdir)
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
                    skip_pass,
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
            # AR link to mp4 file (so that main-status.html hopefully is lighter)
            outmp4 = outpng.replace(".png", ".mp4")
            html.write(
                "\t<p>An animated movie since the start of the program is available at this <a href='{}' target='external'> link.</p>\n".format(
                    outmp4
                )
            )
            txt = "<a href='{}'><img SRC='{}' width=55% height=auto></a>".format(
                outpng, outpng
            )
            html.write("\t\t<td> {} </td>\n".format(txt))

            # AR zhist
            if program in ["DARK", "BRIGHT", "DARK1B", "BRIGHT1B"]:
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


    # AR per-program products
    html.write(
        "<button type='button' class='collapsible'><strong>Per-program plots/movies</strong></button>\n"
    )
    html.write("<div class='content'>\n")
    # AR retrieve button coding + function to call in window.onload
    button_cont_txts, perprog_onload_func_txts = get_perprog_menu(
        outdir,
        perprog_func_name,
        survey=survey,
    )
    # AR print button coding
    for txt in button_cont_txts:
        html.write(txt)

    # AR programs clarification
    html.write("\t<p>\n")
    html.write("\tPrograms are defined as tiles with:\n")
    html.write("\t<ul>\n")
    for program, skip_pass, program_str in zip(programs, skip_passes, program_strs):
        txt = "{}: ".format(program_str)
        if program_str in ["BRIGHT_RUN1B", "DARK_RUN1B"]:
            txt += "FAPRGRM/PROGRAM={}, or FAPRGRM/PROGRAM={} and observed with NIGHT>20260414 (200260414 is the end of Run1a)".format(
                program_str.replace("_RUN", ""), program_str.replace("_RUN1B", ""),
            )
        else:
            txt += "FAPRGRM/PROGRAM={}".format(program)
        if skip_pass is not None:
            txt += ", PASS!={}".format(",".join([str(_) for _ in skip_pass]))
        html.write("\t\t<li><p>{}.</p></li>\n".format(txt.lower()))
    html.write("\t</ul>\n")
    html.write("\t<br>\n")

    # AR plots clarification
    html.write("\t<p>We consider a tile observed if it has EFFTIME_SPEC > MINTFRAC * GOALTIME (except for BACKUP: EFFTIME_SPEC > 1.0s).</p>\n")
    html.write("\t<p>See e.g. the DARK section for a more detailed explanation for each plot.</p>\n")
    html.write("\t\t<p>Available plots/movies are:</p>\n")
    html.write("\t<ul>\n")
    html.write("\t\t<li><p>skymap-obs-ntile: sky map of observed tiles; the color-coding represents the number of tiles.</p></li>\n")
    html.write("\t\t<li><p>skymap-obs-fraccov: sky map of observed tiles; the color-coding represents the fraction of the final coverage.</p></li>\n")
    html.write("\t\t<li><p>skymap-complra: sky map of observed tiles; the color-coding represents the (weighted) completeness per R.A. slice of 30 deg.</p></li>\n")
    html.write("\t\t<li><p>skymap-compllst: sky map of observed tiles; the color-coding represents the (weighted) completeness per LST slice of 30 deg.</p></li>\n")
    html.write("\t\t<li><p>skymap-pending (for BRIGHT* and DARK* programs only): sky map of pending tiles, identified from tiles-main.ecsv with: (IN_DESI=True) & ((STATUS == 'obsstart') | (STATUS == 'obsend').</p></li>\n")
    html.write("\t\t<li><p>zhist (for BRIGHT* and DARK* programs only): redshift histogram.</p></li>\n")
    html.write("\t\t<li><p>qso: QSO/Lya diagnosis.</p></li>\n")
    html.write("\t</ul>\n")

    # AR close
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
    html.write("<div class='content'>\n")
    # AR retrieve button coding + function to call in window.onload
    button_cont_txts, pernight_onload_func_txts = get_pernight_menu(
        outdir,
        ["spacewatch", "skyseq", "obsconds", "exposures"],
        np.unique(obs_nights),
        pernight_func_name,
    )
    # AR print button coding
    for txt in button_cont_txts:
        html.write(txt)
    # AR close
    html.write("</div>\n")
    html.write("\n")


    # AR window.onload: defined functions to use the per-program and per-night buttons
    for txt in perprog_onload_func_txts:
        html.write(txt)
    for txt in pernight_onload_func_txts:
        html.write(txt)
    html.write("<script type='text/javascript'>\n")
    html.write("window.onload = function() {\n")
    # AR order apparently matters... need to put pernight_func_name first
    html.write("\t{}();\n".format(pernight_func_name))
    html.write("\t{}();\n".format(perprog_func_name))
    html.write("}\n")
    html.write("</script>\n")


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
    prog_collapsibles = np.unique(
        [collapsible_names[program] for program in collapsible_names]
    ).tolist()
    for collapsible in [
        "collapsible",
        "collapsible_sub",
    ] + prog_collapsibles:
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
            datetime.now().astimezone().strftime("%H:%M %Z, %b %-d, %Y")
        )
    )


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

    # AR to simplify files, only define 3 categories of collapsibles: BACKUP, BRIGHT, DARK
    _, _, program_strs = get_programs_passparams(survey=survey)
    for program_str in program_strs:
        progshort = None
        for progtest in ["BACKUP", "BRIGHT", "DARK"]:
            if program_str[:len(progtest)] == progtest:
                progshort = progtest
        if progshort is None:
            raise ValueError("Unexpected program_str = {}".format(program_str))
        collapsible_names[program_str] = "collapsible_{}".format(progshort.lower())
        collapsible_names["sub{}".format(program_str)] = "collapsible_sub{}".format(
            progshort.lower()
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


def get_perprog_menu(outdir, function_name, survey="main", n_indent=1):
    """
    Get the text to write in html to have the per-program clickable menu.

    Args:
        outdir: output folder (str)
        function_name: name of the defined function (str)
        survey (optional, defaults to main): survey (str)
        n_indent (optional, defaults to 1): indentation level (int)

    Returns:
        button_cont_txts: a list of strings to print to define the buttons content (list of str)
        onload_func_txts: a list of strings to print to define the function (in the window.onload call)
    """

    program_strs = ["BACKUP", "BRIGHT", "BRIGHT4PASS", "BRIGHT1B", "BRIGHT1BNOPASS5", "DARK", "DARK1B"]

    # AR (name, case, quant, [fmts]) ; last element is a list of formats
    items = {}
    for program_str in program_strs:
        items[program_str] = [
            ("skymap", "obs", "ntile", ["png", "mp4"]),
            ("skymap", "obs", "fraccov", ["png", "mp4"]),
            ("skymap", "complra", None, ["png"]),
            ("skymap", "compllst", None, ["png"]),
        ]
        if (program_str[:6] == "BRIGHT") | (program_str[:4] == "DARK"):
            items[program_str] += [
                ("skymap", "pending", None, ["png"]),
                ("zhist", None, None, ["png"]),
            ]
        if program_str[:4] == "DARK":
             items[program_str] += [
                ("qso", None, None, ["png"]) ,
            ]

    # AR initialize
    button_cont_txts = [
        "<p>Pick a png/mp4 file to display:",
    ]

    # AR button: per-case box
    for name in ["program", "file", "format"]:
        button_cont_txts += [
            "<select name='{}' id='{}'>".format(name, name),
            "\t<option value='' selected='selected'>{}</option>".format(name),
            "</select>",
        ]
    # AR button: "go" box
    button_cont_txts += [
        "<button onclick='getprog()' id='myProgButton' class='btn request-callback' > Go! </button>",
        "</p>",
        "<p id='proglink'></p>",
        "<p> <span class='output'></span> </p>",
    ]

    # AR button: dictionary with content
    button_cont_txts += [
        "<script type='text/javascript'>",
        "var programObject = {",
    ]
    for program_str in program_strs:
        button_cont_txts += ["\t'{}': {{".format(program_str)]
        for item in items[program_str]:
            button_cont_txts += [
                "\t\t'{}': {},".format(
                    "-".join([_ for _ in item[:-1] if _ is not None]),
                    str(item[-1]),
                )
            ]
        button_cont_txts += ["\t},"]
    button_cont_txts += ["}"]

    # AR button: function
    button_cont_txts += [
        "function getprog() {",
        "\tvar x;",
        "\tvar link;",
        "\tvar program;",
        "\tvar file;",
        "\tvar format;",
        "\tprogram = document.querySelector('#program').value;",
        "\tfile = document.querySelector('#file').value;",
        "\tformat = document.querySelector('#format').value;",
        "\tx = program + '-' + file + '.' + format;",
    ]
    started = None
    for program_str in program_strs:
        for item in items[program_str]:
            # print(program_str, item)
            name, case, quant, fmts = item
            file = "-".join([_ for _ in item[:-1] if _ is not None])
            for fmt in fmts:
                if started is None:
                    cond_str = "if"
                    started = 1
                else:
                    cond_str = "} else if"
                button_cont_txts += [
                    "\t" + cond_str + " (x == '" + program_str + "-" + file +"." + fmt + "') {",
                    "\t\tlink = '" + path_full2web(get_filename(outdir, survey, name, program_str=program_str, case=case, quant=quant, ext=fmt)) + "';",
                ]
    button_cont_txts += [
        "\t} else {",
        "\t1==1",
        "\t}",
        "document.getElementById('proglink').innerHTML = 'Opening ' + link;",
        "\twindow.open(link);",
        "}",
        "</script>",
    ]

    # AR button: function
    onload_func_txts = [
        "<script type='text/javascript'>",
        "function " + function_name + "() {",
        "var programSel = document.getElementById('program');",
        "var fileSel = document.getElementById('file');",
        "var formatSel = document.getElementById('format');",
        "for (var x in programObject) {",
        "\tprogramSel.options[programSel.options.length] = new Option(x, x);",
        "}",
        "programSel.onchange = function() {",
        "fileSel.length = 1;",
        "formatSel.length = 1;",
        "var file = Object.keys(programObject[programSel.value]);",
        "for (var i = 0; i < file.length; i++) {",
        "fileSel.options[fileSel.options.length] = new Option(file[i], file[i]);",
        "\t}",
        "}",
        "fileSel.onchange = function() {",
        "formatSel.length = 1;",
        "var z = programObject[programSel.value][this.value];",
        "for (var i = 0; i < z.length; i++) {",
        "\tformatSel.options[formatSel.options.length] = new Option(z[i], z[i]);",
        "\t}",
        "}",
        "}",
        "</script>",
    ]

    # AR add indents + end of line
    indent_str = "".join(["\t" for _ in range(n_indent)])
    button_cont_txts = ["{}{}\n".format(indent_str, _) for _ in button_cont_txts]
    onload_func_txts = ["{}{}\n".format(indent_str, _) for _ in onload_func_txts]

    return button_cont_txts, onload_func_txts


def get_pernight_menu(outdir, cases, nights, function_name, n_indent=1):
    """
    Get the text to write in html to have the clickable menu.

    Args:
        outdir: output folder (str)
        cases: list "first level" cases (str)
        nights: list of all nights (str)
        function_name: name of the defined function (str)
        n_indent (optional, defaults to 1): indentation level (int)

    Returns:
        button_cont_txts: a list of strings to print to define the buttons content (list of str)
        onload_func_txts: a list of strings to print to define the function (in the window.onload call)

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

    # AR button: initialize
    button_cont_txts = [
        "<p>Pick a product to display:",
        "<select id='product'>",
    ]
    # AR button: per-case box
    for case in cases:
        button_cont_txts += [
            "\t<option value='{}'>{}</option>".format(case, case),
        ]
    # AR button: yyyy, mm dd boxes
    button_cont_txts += [
        "</select>",
        "&nbsp",
        "<select name='yyyy' id='yyyy'>",
        "\t<option value='' selected='selected'>yyyy</option>",
        "</select>",
        "<select name='mm' id='mm'>",
        "\t<option value='' selected='selected'>mm</option>",
        "</select>",
        "<select name='dd' id='dd'>",
        "\t<option value='' selected='selected'>dd</option>",
        "</select>",
    ]
    # AR button: "go" box
    button_cont_txts += [
        "<button onclick='getnight()' id='myNightButton' class='btn request-callback' > Go! </button>",
        "</p>",
        "<p id='nightlink'></p>",
        "<p> <span class='output'></span> </p>",
    ]

    # AR button: dictionary with content
    button_cont_txts += [
        "<script type='text/javascript'>",
        "var yyyyObject = {",
    ]
    for yyyy in np.unique(my_yyyys)[::-1]:
        button_cont_txts += ["\t'{}': {{".format(yyyy)]
        yyyy_sel = my_yyyys == yyyy
        for mm in np.unique(my_mms[yyyy_sel]):
            yyyymm_sel = (yyyy_sel) & (my_mms == mm)
            dds = my_dds[yyyymm_sel]
            dds = ["'{}'".format(_) for _ in dds]
            button_cont_txts += ["'{}': [{}],".format(mm, ", ".join(dds))]
        button_cont_txts += ["},"]
    button_cont_txts += ["}"]

    # AR button: function
    button_cont_txts += [
        "function getnight() {",
        "var yyyy;",
        "var mm;",
        "var dd;",
        "var ext;",
        "var link;",
        "product = document.querySelector('#product').value;",
        "yyyy = document.querySelector('#yyyy').value;",
        "mm = document.querySelector('#mm').value;",
        "dd = document.querySelector('#dd').value;",
        "if (product == 'spacewatch'){",
        " ext = 'mp4';",
        "} else if (product == 'skyseq') {",
        "ext = 'png';",
        "} else if ((product == 'obsconds') || (product == 'sframesky') || (product == 'tileqa')) {",
        "ext = 'pdf';",
        "} else {",
        "ext = 'html';",
        "}",
        "link = '{}/' + product + '/' + product + '-' + yyyy + mm + dd + '.' + ext".format(
            path_full2web(outdir)
        ),
        "document.getElementById('nightlink').innerHTML = 'Opening ' + link;",
        "window.open(link);",
        "}",
        "</script>",
    ]

    # AR function to be loaded in window.onload
    onload_func_txts = [
        "<script type='text/javascript'>",
        "function " + function_name + "() {",
        "var yyyySel = document.getElementById('yyyy');",
        "var mmSel = document.getElementById('mm');",
        "var ddSel = document.getElementById('dd');",
        "for (var x in yyyyObject) {",
        "\tyyyySel.options[yyyySel.options.length] = new Option(x, x);",
        "}",
        "yyyySel.onchange = function() {",
        "ddSel.length = 1;",
        "mmSel.length = 1;",
        "var mm = Object.keys(yyyyObject[yyyySel.value]);",
        "mm.sort();",
        "for (var i = 0; i < mm.length; i++) {",
        "mmSel.options[mmSel.options.length] = new Option(mm[i], mm[i]);",
        "\t}",
        "}",
        "mmSel.onchange = function() {",
        "ddSel.length = 1;",
        "var z = yyyyObject[yyyySel.value][this.value];",
        "for (var i = 0; i < z.length; i++) {",
        "\tddSel.options[ddSel.options.length] = new Option(z[i], z[i]);",
        "\t}",
        "}",
        "}",
        "</script>",
    ]

    # AR add indents + end of line
    indent_str = "".join(["\t" for _ in range(n_indent)])
    button_cont_txts = ["{}{}\n".format(indent_str, _) for _ in button_cont_txts]
    onload_func_txts = ["{}{}\n".format(indent_str, _) for _ in onload_func_txts]

    return button_cont_txts, onload_func_txts
