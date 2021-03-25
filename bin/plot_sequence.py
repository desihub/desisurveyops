#!/usr/bin/env python
  
import os, sys
import numpy as np
from astropy.io import fits
from glob import glob
import argparse
import matplotlib.pyplot as plt
from astropy.time import Time
import desisurveyops.utilities as utils

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                        description="""Generate detailed timing plot between two times,  
                        e.g. try ./plot_night.py -n 20201221""") 

parser.add_argument('-n','--night', type=str, default="20201215", required=False,
                    help='Night to analyze')

parser.add_argument('-t1','--starttime', type=float, default=None, required=True,
                    help='Start time for plot')

parser.add_argument('-t2','--stoptime', type=float, default=None, required=True,
                    help='Stop time for plot')

parser.add_argument('-o','--outplot', type=str, default=None, required=False,
                    help='Ouptput file name (plot-night.png)')

parser.add_argument('-v','--verbose', action='store_true', default=False, 
                    required=False,
                    help='Provide verbose output?')

args = parser.parse_args()
night = args.night
t1 = args.starttime
t2 = args.stoptime

outdir = utils.set_outdir()
utils.plot_defaults() 

if args.outplot is None:
    outplot = os.path.join(outdir, "plot-" + night + ".png") 
else: 
    outplot = os.path.join(outdir, args.outplot+".png") 

# Compute total science and guiding time in hours
twibeg_mjd, twiend_mjd = utils.get_twilights(int(night))
startdate = int(twibeg_mjd) 
twibeg_hours = 24.*(twibeg_mjd - startdate)
twiend_hours = 24.*(twiend_mjd - startdate)
twitot_hours = twiend_hours - twibeg_hours

# Get dome telemetry
dome = utils.get_dometelemetry(night) 
dometime = np.array([ (t - startdate)*24 for t in dome['dome_timestamp']])

if args.verbose:
    print("Finished dome telemetry query")

# Get spectrograph CCD telemetry
spec_ccds = utils.get_ccdtelemetry(night)
specccdtime = (spec_ccds['time_recorded'] - startdate)*24

if args.verbose:
    print("Finished CCD telemetry query")

# Telescope telemetry
tel = utils.get_teltelemetry(night)
teltime = (tel['time_recorded'] - startdate)*24

if args.verbose:
    print("Finished telescope telemetry query")

# Create the plot 
barheight = 0.10
y_sky = 0.5
y_guide = 0.3
y_science = 0.1
y2 = 1

# Get all expids from the night
expidlist = utils.getexpidlist(night)

# Get spec data:
spec_start, spec_width, spec_expid = utils.get_scidata(expidlist, night)

# Narrow down the list of expids to the specified time range
mask = np.asarray(spec_start) > t1 - 0.5
mask = mask*(np.asarray(spec_start) < t2 + 0.5)
expidlist = np.asarray(spec_expid)[mask]

# Get obs data for this time range:
science_start, science_width, dither_start, dither_width, guide_start, guide_width = utils.calc_obstimes(night)

science_start = np.asarray(spec_start)[mask]
science_width = np.asarray(spec_width)[mask]
sky_start, sky_width = utils.get_skydata(expidlist, night)
guide_start, guide_width = utils.get_guidedata(expidlist, night)
fvc_start, fvc_width = utils.get_fvcdata(expidlist, night)

fig, ax1 = plt.subplots(figsize=(18,8))
ax1.plot(teltime, 100*tel['tracking'], label="tracking")
ax1.plot(teltime, 95*tel['mirror_ready'], label="mirror_ready")
ax1.plot(teltime, 90*tel['dome_inposition'], 'b--', label="dome_inposition")
ax1.plot(specccdtime, 105*spec_ccds['ccd_idle'], 'r:', label="ccd_idle (spectros)")

ax1.plot(teltime, tel['slew_timer'], 'k:', label="slew_timer [s]")
ax1.set_xlabel('UT Time [hours]')
ax1.set_ylabel('Slew Timer')

ax1.set_xlim(t1, t2)
ax1.set_ylim(-5, 160)
ax1.legend(loc='upper right')

ax2 = ax1.twinx()
ax2.barh(y_science, science_width, barheight, science_start, label="Spec Exposure", align='center', color='red', alpha=0.5)
ax2.barh(y_guide, guide_width, barheight, guide_start, label="Guide Exposure", align='center', color='gray', alpha=0.25)
ax2.barh(y_science, fvc_width, barheight, fvc_start, label="FVC Exposure", align='center', color='black', alpha=0.5)
ax2.barh(y_sky, sky_width, barheight, sky_start, label="Sky Exposure", align='center', color='blue', alpha=0.25)
for i in range(len(spec_start)):
    if mask[i] and spec_start[i] >= t1 and spec_start[i] <= t2:
        ax2.text(spec_start[i], y_science, spec_expid[i], va='center')
ax2.set_ylim(0, y2)
ax2.set_yticks([])
ax2.legend(loc='upper left')

plt.savefig(outplot, bbox_inches="tight")
if args.verbose: 
    print("Wrote", outplot)

