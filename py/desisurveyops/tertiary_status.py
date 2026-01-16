#!/usr/bin/env python

import os
from glob import glob
import tempfile
import multiprocessing
import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import gridspec
from PIL import Image
from desispec.tile_qa_plot import get_quantz_cmap, make_tile_qa_plot
from desispec.night_qa import cameras, petals, _read_sframesky
from desisurveyops.status_utils import get_speed
from desisurveyops.status_html import write_html_today, write_html_collapse_script
from desiutil.log import get_logger

log = get_logger()

# AR to be updated with new prognums
def get_prognum_desc(prognum=None):
    myd = {
        1: {"TARGETS": "High-density z<1", "FIELD": "COSMOS"},
        2: {"TARGETS": "High-density z<1", "FIELD": "XMM-LSS"},
        3: {"TARGETS": "M31", "FIELD": "M31"},
        4: {"TARGETS": "M31 (metal poor halo)", "FIELD": "M31"},
        # AR calibration fields
        5: {"TARGETS": "Calibration", "FIELD": "XMM-LSS"},
        6: {"TARGETS": "Calibration", "FIELD": "XMM-LSS"},
        7: {"TARGETS": "Calibration", "FIELD": "COSMOS"},
        8: {"TARGETS": "Calibration", "FIELD": "COSMOS"},
        9: {"TARGETS": "Calibration", "FIELD": "MBHB1"},
        10: {"TARGETS": "Calibration", "FIELD": "MBHB1"},
        11: {"TARGETS": "Calibration", "FIELD": "GAMA15"},
        12: {"TARGETS": "Calibration", "FIELD": "GAMA15"},
        #
        13: {"TARGETS": "M31", "FIELD": "M31"},
        14: {"TARGETS": "M31", "FIELD": "M31"},
        15: {"TARGETS": "LBG/LAE", "FIELD": "XMM-LSS"},
        16: {"TARGETS": "ELG-bkp-4pet", "FIELD": "-"},
        17: {"TARGETS": "GD1-bkp-4pet", "FIELD": "-"},
        18: {"TARGETS": "LAE ODIN/WIRO", "FIELD": "XMM-LSS"},
        19: {"TARGETS": "DirCal", "FIELD": "XMM-LSS"},
        20: {"TARGETS": "GD1", "FIELD": "GD1"},
        21: {"TARGETS": "ELG-bkp-5pet", "FIELD": "-"},
        22: {"TARGETS": "M81-M82-bkp-5pet", "FIELD": "M81-M82"},
        23: {"TARGETS": "4-in-1", "FIELD": "COSMOS"},
        24: {"TARGETS": "BOOTES-III", "FIELD": "BOOTES-III"},
        25: {"TARGETS": "BOOTES-III", "FIELD": "BOOTES-III"},
        26: {"TARGETS": "LBG/LAE/ELG", "FIELD": "COSMOS"},
        27: {"TARGETS": "DirCal-bkp-4pet", "FIELD": "RA,DEC=243,43"},
        28: {"TARGETS": "UrsaMinor-bkp-4pet", "FIELD": "UrsaMinor"},
        29: {"TARGETS": "GD1-bkp-4pet", "FIELD": "GD1 Stream"},
        30: {"TARGETS": "Pal5", "FIELD": "Pal5"},
        31: {"TARGETS": "Pal5", "FIELD": "Pal5"},
        32: {"TARGETS": "SN-M101", "FIELD": "M101"},
        33: {"TARGETS": "UrsaMinor", "FIELD": "UrsaMinor"},
        34: {"TARGETS": "NGC4993", "FIELD": "NGC4993"},
        35: {"TARGETS": "NGC4993", "FIELD": "NGC4993"},
        36: {"TARGETS": "M92", "FIELD": "M92"},
        37: {"TARGETS": "LBG", "FIELD": "COSMOS"},
        38: {"TARGETS": "ELG", "FIELD": "COSMOS"},
        39: {"TARGETS": "Spare", "FIELD": "COSMOS"},
        40: {"TARGETS": "NGC4993", "FIELD": "NGC4993"},
        41: {"TARGETS": "CFIS_LYA", "FIELD": "RA,DEC=350,1.5"},
        42: {"TARGETS": "LOWDEC", "FIELD": "-38 DEC -23"},
        43: {"TARGETS": "LOWDEC", "FIELD": "-38 DEC -23"},
        44: {"TARGETS": "HIZ_IBIS", "FIELD": "XMM-LSS"},
        45: {"TARGETS": "M31", "FIELD": "M31"},
        46: {"TARGETS": "PMTEST", "FIELD": "DEC=-31,-36"},
        47: {"TARGETS": "LAE/LBG/DWARF", "FIELD": "COSMOS"},
        48: {"TARGETS": "XLG", "FIELD": "XMM-LSS"},
        49: {"TARGETS": "LAE/QSO", "FIELD": "XMM-LSS"},
    }
    if prognum is None:
        return myd
    else:
        return myd[prognum]


def get_tertiary_folder(prognum):
    mydir = os.path.join(
        os.getenv("DESI_ROOT"),
        "survey",
        "fiberassign",
        "special",
        "tertiary",
        "{:04d}".format(prognum),
    )
    return mydir


# AR latest spectro prod
# TODO AR is there an automated way to do that?
# AR note: does not really matters here, as we usually use
# AR    the tertiary-status page to build from the daily
# AR    to monitor the latest observations
def get_ref_prod():
    return "loa"


def get_prognum_latest_prod(prognum):

    # AR tileids
    fns = sorted(
        glob(
            os.path.join(
                get_tertiary_folder(prognum), "08?", "fiberassign-??????.fits.gz"
            )
        )
    )
    tileids = [fits.getheader(fn, 0)["TILEID"] for fn in fns]

    # AR is this prognum in the latest prod?
    # AR if not go for daily
    ref_prod = get_ref_prod()
    exps = read_exposures(ref_prod, prognum=prognum)
    if len(exps) > 0:
        return ref_prod
    else:
        return "daily"


# AR for some bright programs, we set GOALTIME=600
# AR    but we want 3x obs., ie GOALTIME=1800
# def get_prognums_inflate_goaltime_by_three():
#    return np.array([13, 20, 24, 30])
# AR for some bright programs, we set GOALTIME=600
# AR    but we want 3x obs., ie GOALTIME=1800
# AR for some dark programs, we set GOALTIME=1000
# AR    but we want 4x obs., ie GOALTIME=4000
# AR could need to be updated for new prognums
def get_goaltime_factor_inflate(prognum):
    if prognum in [13, 20, 24, 30, 48]:
        factor = 3.0
    # AR forgot to change GOALTIME=1000 to GOALTIME=8000 for 83577...
    elif prognum in [47]:
        factor = 8.0
    # AR tertiary49: GOALTIME=3600 but we finally requested 4000...
    elif prognum in [49]:
        factor = 10.0 / 9.0
    # AR default value
    else:
        factor = 1.0
    return factor


# AR exposures to discard
def get_black_expids():
    return np.array(
        [
            # pal5/bright obs. taken in dark conditions
            # ([desi-milkyway 3612] 6/15/23)
            185031,
            185032,
            185033,
            185034,
            185035,
            185036,
        ]
    )


# AR removes efftime_spec=0 exposures..
def read_exposures(prod, prognum=None):
    fn = os.path.join(
        os.getenv("DESI_ROOT"),
        "spectro",
        "redux",
        prod,
        "exposures-{}.csv".format(prod),
    )
    d = Table.read(fn)
    sel = d["EFFTIME_SPEC"] > 0
    sel &= ~np.in1d(d["EXPID"], get_black_expids())

    if prognum is not None:
        sel &= d["FAPRGRM"] == "tertiary{}".format(prognum)

    d = d[sel]

    return d


def get_overview_table(prognums):

    nprog = len(prognums)

    myd = {
        key: ["-" for _ in range(nprog)]
        for key in [
            "PROGNUM",
            "TARGETS",
            "FIELD",
            "OBSCONDS",
            "SBPROF",
            "GOALTIME",
            "TILEID_MIN",
            "TILEID_MAX",
            "NTILES",
            "NIGHT_MIN",
            "NIGHT_MAX",
        ]
    }
    myd["PROGNUM"] = prognums

    # AR tileids, obsonds, goaltime, night_min, night_max
    for i, prognum in enumerate(prognums):

        mydir = get_tertiary_folder(prognum)

        fn = os.path.join(mydir, "tertiary-tiles-{:04d}.ecsv".format(prognum))
        d = Table.read(fn)
        myd["TILEID_MIN"][i], myd["TILEID_MAX"][i], myd["NTILES"][i] = (
            d["TILEID"].min(),
            d["TILEID"].max(),
            len(d),
        )

        fn = os.path.join(mydir, "tertiary-targets-{:04d}.fits".format(prognum))
        hdr = fits.getheader(fn, "TARGETS")
        for key in ["OBSCONDS", "SBPROF", "GOALTIME"]:
            val = hdr[key]
            if key == "GOALTIME":
                val *= get_goaltime_factor_inflate(prognum)
            myd[key][i] = val

        myprod = get_prognum_latest_prod(prognum)
        e = read_exposures(myprod, prognum=prognum)
        sel = e["FAPRGRM"] == "tertiary{}".format(prognum)
        if sel.sum() > 0:
            myd["NIGHT_MIN"][i], myd["NIGHT_MAX"][i] = (
                e["NIGHT"][sel].min(),
                e["NIGHT"][sel].max(),
            )
    # AR targets, field
    for key in ["TARGETS", "FIELD"]:
        myd[key] = [get_prognum_desc(prognum=prognum)[key] for prognum in prognums]

    d = Table()
    for key in myd:
        d[key] = myd[key]

    return d


def write_html_init(html, outcss):

    html.write("<html><body>\n")
    html.write("<h1>Tertiary programs overview page </h1>\n")
    html.write("\n")

    html.write("<head>\n")
    html.write("\t<meta charset='UTF-8'>\n")
    html.write("\t<meta http-equiv='X-UA-Compatible' content='IE=edge'>\n")
    html.write(
        "\t<meta name='viewport' content='width=device-width, initial-scale=1.0'>\n"
    )
    html.write("\t<link rel='stylesheet' href='{}'>\n".format(os.path.basename(outcss)))
    html.write("</head>\n")
    html.write("\n")
    html.write("<body>\n")
    html.write("\n")
    html.write("<script src='sort-table.js'></script>\n")
    html.write("<link rel='stylesheet' href='sort-table.css'>\n")
    html.write("\n")


def write_html_overview(html, prognums):
    html.write(
        "\t<button style='margin-left:25px;' typ='button' class='{}'><strong>Tertiary programs: overview</strong></button>\n".format(
            "collapsible_even",
        )
    )
    html.write("\t<div class='content'>\n")
    html.write("\t<br>\n")

    d = get_overview_table(prognums)

    # AR header
    # html.write("\t<table>\n")
    html.write("\t<table class='js-sort-table'>\n")
    for key in d.colnames:
        if isinstance(d[key][0], np.number):
            keyclass = "js-sort-number"
        else:
            keyclass = "js-sort-string"
        html.write("\t\t\t\t<th class={}>{}</th>\n".format(keyclass, key))
    html.write("\t\t\t</tr>\n")
    for i in range(len(d)):
        color = "k"
        for key in d.colnames:
            html.write("\t\t<td style='color:{};'> {} </td>".format(color, d[key][i]))
        html.write("\t</tr>\n")
    html.write("\t</table>\n")
    html.write("\n")
    html.write("\t</div>\n")
    html.write("\n")


def make_sky_plot(prognum, prod, outpng):

    d = read_exposures(prod, prognum=None)
    sel = d["FAPRGRM"] == "tertiary{}".format(prognum)
    label = "Tertiary{} ({} exposures over {} tiles over {} nights)".format(
        prognum,
        sel.sum(),
        np.unique(d["TILEID"][sel]).size,
        np.unique(d["NIGHT"][sel]).size,
    )

    clim = (0.5, 1.5)
    cmap = get_quantz_cmap(matplotlib.cm.jet, 11, 0, 1)
    fig, ax = plt.subplots()
    ax.scatter(
        d["SKY_MAG_R_SPEC"],
        d["SKY_MAG_G_SPEC"] - d["SKY_MAG_R_SPEC"],
        c=d["EFFTIME_SPEC"] / d["EFFTIME_ETC"],
        zorder=0,
        s=1,
        alpha=0.1,
        cmap=cmap,
        vmin=clim[0],
        vmax=clim[1],
    )
    sc = ax.scatter(
        d["SKY_MAG_R_SPEC"][sel],
        (d["SKY_MAG_G_SPEC"] - d["SKY_MAG_R_SPEC"])[sel],
        c=(d["EFFTIME_SPEC"] / d["EFFTIME_ETC"])[sel],
        ec="k",
        zorder=1,
        s=20,
        alpha=0.7,
        cmap=cmap,
        vmin=clim[0],
        vmax=clim[1],
        label=label,
    )
    ax.set_title("{} -- as of NIGHT={}".format(prod, d["NIGHT"].max()))
    ax.set_xlabel("SKY_MAG_R_SPEC")
    ax.set_ylabel("SKY_MAG_G_SPEC - SKY_MAG_R_SPEC")
    ax.set_xlim(17, 22)
    ax.set_ylim(-0.5, 1.5)
    ax.grid()
    ax.legend(loc=2)
    #
    cbar = plt.colorbar(sc)
    cbar.set_label("EFFTIME_SPEC / EFFTIME_ETC")
    cbar.mappable.set_clim(clim)
    #
    plt.savefig(outpng, bbox_inches="tight")
    plt.close()


def make_speed_plot(prognum, prod, outpng):

    d = read_exposures(prod, prognum=prognum)
    d["SPEED_ETC"] = get_speed(d, "etc")
    d["SPEED_SPEC"] = get_speed(d, "spec")

    label = "Tertiary{} ({} exposures over {} tiles over {} nights)".format(
        prognum, len(d), np.unique(d["TILEID"]).size, np.unique(d["NIGHT"]).size
    )

    bins = np.linspace(0, 3, 31)

    fig, ax = plt.subplots()
    _ = ax.hist(
        d["SPEED_ETC"],
        bins=bins,
        density=True,
        histtype="step",
        color="r",
        alpha=0.5,
        label="ETC; {}".format(label),
    )
    _ = ax.hist(
        d["SPEED_SPEC"],
        bins=bins,
        density=True,
        histtype="stepfilled",
        color="b",
        alpha=0.5,
        label="SPEC; {}".format(label),
    )
    ax.axvline(0.4, color="k", ls="--")
    ax.set_title("{} -- as of NIGHT={}".format(prod, d["NIGHT"].max()))
    ax.set_xlabel("Survey speed (ETC or SPEC)")
    ax.set_ylabel("Normalized counts")
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(0, 4)
    ax.grid()
    ax.legend(loc=2)
    plt.savefig(outpng, bbox_inches="tight")
    plt.close()


# AR https://github.com/desihub/desispec/blob/eb9f86e93c81c11792363121408231bc45b370c7/py/desispec/night_qa.py#L878-L954
# AR copying that function, but need a minor re-writing because of the night argument...
def make_sframesky_pdf(prognum, prod, outpdf, numproc):
    #
    specprod_dir = os.path.join(os.getenv("DESI_ROOT"), "spectro", "redux", prod)
    d = read_exposures(prod, prognum=prognum)
    #
    myargs = []
    for expid, night in zip(d["EXPID"], d["NIGHT"]):
        myargs.append(
            [
                night,
                specprod_dir,
                expid,
            ]
        )
    # AR launching pool
    pool = multiprocessing.Pool(processes=numproc)
    with pool:
        mydicts = pool.starmap(_read_sframesky, myargs)
    # AR creating pdf (+ removing temporary files)
    cmap = matplotlib.cm.Greys_r
    with PdfPages(outpdf) as pdf:
        for mydict in mydicts:
            if mydict is not None:
                fig = plt.figure(figsize=(20, 10))
                gs = gridspec.GridSpec(len(cameras), 1, hspace=0.025)
                clim = (-100, 100)
                xlim = (0, 3000)
                for ic, camera in enumerate(cameras):
                    ax = plt.subplot(gs[ic])
                    nsky = 0
                    if "flux" in mydict[camera]:
                        nsky = mydict[camera]["flux"].shape[0]
                        im = ax.imshow(
                            mydict[camera]["flux"],
                            cmap=cmap,
                            vmin=clim[0],
                            vmax=clim[1],
                            zorder=0,
                        )
                        # AR overlay in transparent pixels with ivar=0
                        # AR a bit obscure why I need to add +1 in xs, ys...
                        # AR probably some indexing convention in imshow()
                        xys = np.argwhere(mydict[camera]["nullivar"])
                        xs, ys = 1 + xys[:, 0], 1 + xys[:, 1]
                        ax.scatter(
                            ys, xs, c="g", s=0.1, alpha=0.1, zorder=1, rasterized=True
                        )
                        for petal in petals:
                            ii = np.where(mydict[camera]["petals"] == petal)[0]
                            if len(ii) > 0:
                                ax.plot(
                                    [0, mydict[camera]["flux"].shape[1]],
                                    [ii.min(), ii.min()],
                                    color="r",
                                    lw=1,
                                    zorder=1,
                                )
                                ax.text(
                                    10,
                                    ii.mean(),
                                    "{}".format(petal),
                                    color="r",
                                    fontsize=10,
                                    va="center",
                                )
                        ax.set_ylim(0, mydict[cameras[0]]["flux"].shape[0])
                        if ic == 1:
                            p = ax.get_position().get_points().flatten()
                            cax = fig.add_axes(
                                [
                                    p[0] + 0.85 * (p[2] - p[0]),
                                    p[1],
                                    0.01 * (p[2] - p[0]),
                                    1.0 * (p[3] - p[1]),
                                ]
                            )
                            cbar = plt.colorbar(
                                im,
                                cax=cax,
                                orientation="vertical",
                                ticklocation="right",
                                pad=0,
                                extend="both",
                            )
                            cbar.set_label("FLUX [{}]".format(mydict["flux_unit"]))
                            cbar.mappable.set_clim(clim)
                    ax.text(
                        0.99,
                        0.92,
                        "CAMERA={}".format(camera),
                        color="k",
                        fontsize=15,
                        fontweight="bold",
                        ha="right",
                        transform=ax.transAxes,
                    )
                    if ic == 0:
                        ax.set_title(
                            "EXPID={:08d}  NIGHT={}  TILEID={}  {} SKY fibers".format(
                                mydict["expid"], mydict["night"], mydict["tileid"], nsky
                            )
                        )
                    ax.set_xlim(xlim)
                    if ic == 2:
                        ax.set_xlabel("WAVELENGTH direction")
                    ax.set_yticklabels([])
                    ax.set_ylabel("FIBER direction")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close()


def make_tileqa_pdf(prognum, prod, outpdf, numproc):

    specprod_dir = os.path.join(os.getenv("DESI_ROOT"), "spectro", "redux", prod)
    d = read_exposures(prod, prognum=prognum)

    # AR d is sorted by increasing expid, so np.unique() is picking the first expid for each tile
    tileids, ii = np.unique(d["TILEID"], return_index=True)
    lastnights = np.array([d["NIGHT"][d["TILEID"] == tileid][-1] for tileid in tileids])

    tmpoutdir = tempfile.mkdtemp()
    myargs = []
    outpngs = []
    for tileid, lastnight in zip(tileids, lastnights):
        in_tileqafits = os.path.join(
            specprod_dir,
            "tiles",
            "cumulative",
            str(tileid),
            str(lastnight),
            "tile-qa-{}-thru{}.fits".format(tileid, lastnight),
        )
        if not os.path.isfile(in_tileqafits):
            print("WARNING : missing {} => skipping it".format(in_tileqafits))
            continue
        outpng = os.path.join(
            tmpoutdir, "tile-qa-{}-thru{}.png".format(tileid, lastnight)
        )
        tileqafits = outpng.replace(".png", ".fits")
        os.system("cp {} {}".format(in_tileqafits, tileqafits))
        outpngs.append(outpng)
        myargs.append([tileqafits, specprod_dir])
    # AR launching pool
    pool = multiprocessing.Pool(processes=numproc)
    with pool:
        mydicts = pool.starmap(make_tile_qa_plot, myargs)
    # AR remove the copied tileqafits
    for myarg in myargs:
        os.remove(myarg[0])
    # AR per-night pdf tileqa
    with PdfPages(outpdf) as pdf:
        for outpng in outpngs:
            fig, ax = plt.subplots()
            img = Image.open(outpng)
            ax.imshow(img, origin="upper")
            ax.axis("off")
            pdf.savefig(fig, bbox_inches="tight", dpi=300)
            plt.close()


def write_html_plot(html, outdir, outpng, width):

    # AR width : e.g. "30%"
    # webpng = outpng.replace("/global/cfs/cdirs", "https://data.desi.lbl.gov")
    # AR we use relative path w.r.t. outdir
    webpng = outpng.replace("{}{}".format(outdir, os.path.sep), "")
    # html.write("<br>\n")
    # html.write("<h2>Sky mag control plot</h2>\n")

    if outpng[-4:] == ".png":
        txt = "<a href='{}'><img SRC='{}' width={} height=auto></a>".format(
            webpng, webpng, width
        )
    elif outpng[-4:] == ".pdf":
        txt = "<iframe src='{}' width=100% height=100%></iframe>\n".format(webpng)
    else:
        raise ValueError("outpng={} => unexpected extension".format(outpng))
    html.write("\t<td> {} </td>\n".format(txt))
    html.write("\t</tr>\n")


def write_html_exptable(html, prognum, prod):
    d = read_exposures(prod)
    d["SPEED_ETC"] = get_speed(d, "etc")
    d["SPEED_SPEC"] = get_speed(d, "spec")
    # AR cut on faprgrm
    sel = d["FAPRGRM"] == "tertiary{}".format(prognum)
    d = d[sel]

    # AR round
    d["EXPTIME"] = d["EXPTIME"].round(0)
    d["EFFTIME_SPEC"] = d["EFFTIME_SPEC"].round(0)
    d["SEEING_GFA"] = d["SEEING_GFA"].round(2)
    d["AIRMASS"] = d["AIRMASS"].round(2)
    d["SPEED_ETC"] = d["SPEED_ETC"].round(2)
    d["SPEED_SPEC"] = d["SPEED_SPEC"].round(2)
    d["SKY_MAG_R_SPEC"] = d["SKY_MAG_R_SPEC"].round(1)
    d["SKY_MAG_GR_SPEC"] = (d["SKY_MAG_G_SPEC"] - d["SKY_MAG_R_SPEC"]).round(1)
    # AR rename
    d["SEEING_GFA"].name = "SEEING"
    d["SKY_MAG_R_SPEC"].name = "SKY_R"
    d["SKY_MAG_GR_SPEC"].name = "SKY_GR"

    # AR html
    keys = [
        "NIGHT",
        "EXPID",
        "TILEID",
        "EXPTIME",
        "SKY_R",
        "SKY_GR",
        "SEEING",
        "AIRMASS",
        "EFFTIME_ETC",
        "EFFTIME_SPEC",
        "SPEED_ETC",
        "SPEED_SPEC",
    ]
    # AR header
    # html.write("\t<table>\n")
    html.write("\t<table class='js-sort-table'>\n")
    for key in keys:
        if isinstance(d[key][0], np.number):
            keyclass = "js-sort-number"
        else:
            keyclass = "js-sort-string"
        html.write("\t\t\t\t<th class={}>{}</th>\n".format(keyclass, key))
        # html.write("\t\t\t\t<th>{}</th>\n".format(key))
    html.write("\t\t\t</tr>\n")
    for i in range(len(d)):
        color = "k"
        for key in keys:
            html.write("\t\t<td style='color:{};'> {} </td>".format(color, d[key][i]))
        html.write("\t</tr>\n")
        # if i % 10 == 0:
        #     for key in keys:
        #         html.write("\t\t\t\t<th>{}</th>\n".format(key))
        #     html.write("\t\t\t</tr>\n")
    html.write("</table>\n")
    html.write("\n")


def get_efftime_min(prognum, prod):
    fn = os.path.join(
        os.getenv("DESI_ROOT"), "spectro", "redux", prod, "tiles-{}.csv".format(prod)
    )
    d = Table.read(fn)
    d = d[d["FAPRGRM"] == "tertiary{}".format(prognum)]
    assert np.unique(d["GOALTIME"] * d["MINTFRAC"]).size == 1
    efftime_min = (d["GOALTIME"] * d["MINTFRAC"])[0]
    efftime_min *= get_goaltime_factor_inflate(prognum)
    print("tertiary{}: efftime_min = {}s".format(prognum, efftime_min))
    return efftime_min


def write_html_tileidtable(html, prognum, prod, efftime_min):
    keys = [
        "TILEID",
        "TILERA",
        "TILEDEC",
        "LASTNIGHT",
        "NEXP",
        "EFFTIME_ETC",
        "EFFTIME_SPEC",
    ]
    # working from exposures-{prod}.csv, not tiles-{prod}.csv, as we discard some exposures
    e = read_exposures(prod, prognum=prognum)
    d = Table()
    d["TILEID"] = np.unique(e["TILEID"])
    d["TILERA"], d["TILEDEC"], d["EFFTIME_ETC"], d["EFFTIME_SPEC"] = 0.0, 0.0, 0.0, 0.0
    d["LASTNIGHT"], d["NEXP"] = 0, 0
    d.keep_columns(keys)
    for i, tileid in enumerate(d["TILEID"]):
        sel = e["TILEID"] == tileid
        d["TILERA"][i], d["TILEDEC"][i] = e["TILERA"][sel][0], e["TILEDEC"][sel][0]
        d["LASTNIGHT"][i] = e["NIGHT"][sel].max()
        d["NEXP"][i] = sel.sum()
        d["EFFTIME_ETC"][i] = e["EFFTIME_ETC"][sel].sum()
        d["EFFTIME_SPEC"][i] = e["EFFTIME_SPEC"][sel].sum()
    d["EFFTIME_ETC"] = d["EFFTIME_ETC"].round(1)
    d["EFFTIME_SPEC"] = d["EFFTIME_SPEC"].round(1)
    # designed but unobserved tiles
    # fn = os.path.join(
    #    os.getenv("DESI_ROOT"), "spectro", "redux", prod, "tiles-{}.csv".format(prod)
    # )
    # d = Table.read(fn)
    # d = d[d["FAPRGRM"] == "tertiary{}".format(prognum)]
    # d.keep_columns(keys)
    #
    mydir = get_tertiary_folder(prognum)
    fn = os.path.join(mydir, "tertiary-tiles-{:04d}.ecsv".format(prognum))
    tiles = Table.read(fn)
    sel = ~np.in1d(tiles["TILEID"], d["TILEID"])
    tiles = tiles[sel]
    tiles["RA"].name, tiles["DEC"].name = "TILERA", "TILEDEC"
    d2 = Table()
    for key in keys:
        if key in ["TILEID", "TILERA", "TILEDEC"]:
            d2[key] = tiles[key]
        else:
            d2[key] = np.zeros_like(d[key], shape=(len(tiles),))
    d = vstack([d, d2])
    d = d[d["TILEID"].argsort()]

    # AR header
    # html.write("\t<table>\n")
    html.write("\t<table class='js-sort-table'>\n")
    for key in keys:
        if isinstance(d[key][0], np.number):
            keyclass = "js-sort-number"
        else:
            keyclass = "js-sort-string"
        html.write("\t\t\t\t<th class={}>{}</th>\n".format(keyclass, key))
        # html.write("\t\t\t\t<th>{}</th>\n".format(key))
    html.write("\t\t\t</tr>\n")
    for i in range(len(d)):
        if d["EFFTIME_SPEC"][i] > efftime_min:
            color = "green"
        else:
            color = "red"
        for key in keys:
            html.write("\t\t<td style='color:{};'> {} </td>".format(color, d[key][i]))
        html.write("\t</tr>\n")
    html.write("</table>\n")
    html.write("\n")


def write_html_prognum(html, prognum, outdir, is_odd, numproc, html_only):
    if is_odd:
        collapsible, collapsible_sub = "collapsible_odd", "collapsible_odd_sub"
    else:
        collapsible, collapsible_sub = "collapsible_even", "collapsible_even_sub"

    html.write(
        "\t<button style='margin-left:25px;' typ='button' class='{}'><strong>Tertiary{}: {}</strong></button>\n".format(
            collapsible, prognum, get_prognum_desc(prognum=prognum)["TARGETS"]
        )
    )
    html.write("\t<div class='content'>\n")
    html.write("\t<br>\n")

    # AR isobs?
    prod = get_prognum_latest_prod(prognum)
    d = read_exposures(prod)
    sel = d["FAPRGRM"] == "tertiary{}".format(prognum)
    isobs = sel.sum() > 0

    if isobs:
        # AR tileid table
        efftime_min = get_efftime_min(prognum, prod)
        html.write(
            "\t<button style='margin-left:25px;' typ='button' class='{}'><strong>Tertiary{}: per-tileid table</strong></button>\n".format(
                collapsible_sub,
                prognum,
            )
        )
        html.write("\t<div class='content'>\n")
        html.write("\t<br>\n")
        html.write(
            "\t<p>A tile is done (displayed in green) if EFFTIME_SPEC > {}s; else it is displayed in red.</p>\n".format(
                efftime_min
            )
        )
        html.write("\t<br>\n")
        write_html_tileidtable(html, prognum, prod, efftime_min)
        html.write("\t</div>\n")
        html.write("\n")

        # AR exposure table
        html.write(
            "\t<button style='margin-left:25px;' typ='button' class='{}'><strong>Tertiary{}: per-exposure table</strong></button>\n".format(
                collapsible_sub,
                prognum,
            )
        )
        html.write("\t<div class='content'>\n")
        html.write("\t<br>\n")
        write_html_exptable(html, prognum, prod)
        html.write("\t</div>\n")
        html.write("\n")

        # AR sky mag, speed control plot, sframesky, tileqa
        for case, func, width in zip(
            ["sky", "speed", "sframesky", "tileqa"],
            [make_sky_plot, make_speed_plot, make_sframesky_pdf, make_tileqa_pdf],
            ["30%", "30%", None, None],
        ):
            outpng = os.path.join(
                outdir, case, "{}-tertiary{}.png".format(case, prognum)
            )
            if case in ["sframesky", "tileqa"]:
                outpng = outpng.replace(".png", ".pdf")
                if not html_only:
                    func(prognum, prod, outpng, numproc)
                collapsible_txt = "Tertiary{}: {} pdf".format(prognum, case)
            else:
                if not html_only:
                    func(prognum, prod, outpng)
                collapsible_txt = "Tertiary{}: {} control plot".format(prognum, case)

            html.write(
                "\t<button style='margin-left:25px;' typ='button' class='{}'><strong>{}</strong></button>\n".format(
                    collapsible_sub,
                    collapsible_txt,
                )
            )
            html.write("\t<div class='content'>\n")
            html.write("\t<br>\n")
            write_html_plot(html, outdir, outpng, width)
            html.write("\t</div>\n")
            html.write("\n")

    html.write("\t</div>\n")
    html.write("\t</tr>\n")
    html.write("\n")


def write_html_close(html):

    for collapsible in [
        "collapsible_odd",
        "collapsible_odd_sub",
        "collapsible_even",
        "collapsible_even_sub",
    ]:
        write_html_collapse_script(html, collapsible)

    # ADM html postamble for main page.
    write_html_today(html)
    html.write("</html></body>\n")
    html.close()
