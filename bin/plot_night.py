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
                        description="""Generate nightly observing efficiency plot,  
                        e.g. try ./plot_night.py -n 20201221""") 

parser.add_argument('-n','--night', type=str, default="20201215", required=False,
                    help='Night to analyze')

parser.add_argument('-c','--clobber', action='store_true', default=False, 
                    required=False,
                    help='Clobber (overwrite) output if it already exists?')

parser.add_argument('-v','--verbose', action='store_true', default=False, 
                    required=False,
                    help='Provide verbose output?')

args = parser.parse_args()
night = args.night

outdir = utils.set_outdir()
utils.plot_defaults() 
outplot = os.path.join(outdir, "nightstats" + night + ".png") 

# See if the nightly data exist, and generate them if they do not:
specfilename = "specdata" + str(night) + ".json"
guidefilename = "guidedata" + str(night) + ".json"
specdatafile = os.path.join(outdir, 'NightlyData', specfilename)
guidedatafile = os.path.join(outdir, 'NightlyData', guidefilename)

if os.path.isfile(specdatafile) and not args.clobber:
    specdata = utils.read_json(specdatafile)
else: 
    print("Note: {} not found ... creating it".format(specdatafile))
    utils.calc_expidlist(night)
    specdata = read_json(specdatafile, verbose=args.verbose, clobber=args.clobber)

if os.path.isfile(guidedatafile) and not args.clobber:
    guidedata = utils.read_json(guidedatafile)
else: 
    print("Note: {} not found ... creating it".format(guidedatafile))
    utils.calc_guidelist(night)
    guidedata = read_json(guidedatafile, verbose=args.verbose, clobber=args.clobber)

twibeg_mjd, twiend_mjd = utils.get_twilights(int(night))
startdate = int(twibeg_mjd) 


# Calculate the start and duration for the science observations:
science_start = []
science_width = []
for item in specdata: 
    if specdata[item]['OBSTYPE'] == 'SCIENCE' and specdata[item]['FLAVOR'] == 'science' and 'Dither' not in specdata[item]['PROGRAM'] and specdata[item]['DOMSHUTU'] == 'open' and specdata[item]['PMCOVER'] == 'open':
        science_start.append( (specdata[item]['DATE-OBS'] - startdate)*24. )
        science_width.append( specdata[item]['EXPTIME']/3600. )

# Separately account for time spent on dither tests
dither_start = []
dither_width = []
for item in specdata:
    if specdata[item]['OBSTYPE'] == 'SCIENCE' and specdata[item]['FLAVOR'] == 'science' and 'Dither' in specdata[item]['PROGRAM']:
        dither_start.append( (specdata[item]['DATE-OBS'] - startdate)*24. )
        dither_width.append( specdata[item]['EXPTIME']/3600. )

# Times for guiding: 
guide_start = []
guide_width = []
for item in guidedata:
    # if guidedata[item]['OBSTYPE'] == 'SCIENCE' and guidedata[item]['FLAVOR'] == 'science' and guidedata[item]['PMCOVER'] == 'open' and guidedata[item]['DOMSHUTU'] == 'open':
    if guidedata[item]['OBSTYPE'] == 'SCIENCE' and guidedata[item]['FLAVOR'] == 'science' and guidedata[item]['DOMSHUTU'] == 'open':
        guide_start.append( (guidedata[item]['GUIDE-START'] - startdate)*24. )
        guide_width.append( (guidedata[item]['GUIDE-STOP'] - guidedata[item]['GUIDE-START'])*24. )

# Get guider fwhm data to plot
guide = utils.get_guidetelemetry(night) 
guidetime = (Time(guide['time_recorded']).mjd - startdate)*24

# Compute total science and guiding time in hours
twibeg_hours = 24.*(twibeg_mjd - startdate)
twiend_hours = 24.*(twiend_mjd - startdate)
twitot_hours = twiend_hours - twibeg_hours

# Get dome telemetry
dome = utils.get_dometelemetry(night) 
dometime = np.array([ (t - startdate)*24 for t in dome['dome_timestamp']])
dome_open = np.array([t for t in dome['shutter_upper']], dtype=bool)
firstopen, lastclose, fractwiopen, twiopen_hours = utils.calc_domevals(night)

# Calculate the total science time and guide time between twilights:
science_hours = utils.get_obstime(science_start, science_width, twibeg_hours, twiend_hours) 
guide_hours = utils.get_obstime(guide_start, guide_width, twibeg_hours, twiend_hours) 

# Percent of time dome was open between twilights with science, guide exposures
if twiopen_hours > 0.:
    science_fraction = science_hours/twiopen_hours
    guide_fraction = guide_hours/twiopen_hours
else:
    science_fraction = 0.
    guide_fraction = 0.

# Total number of each type of observing block
science_num = len(science_width)
guide_num = len(guide_width)

if args.verbose:
    print("Total hours between twilights: {0:.1f}".format(twiopen_hours))
    print("  Open: {0:.1f}%".format(100.*fractwiopen))
    print("  Science: {0:.1f}% ({1})".format(100.*science_fraction, science_num))
    print("  Guide: {0:.1f}% ({1})".format(100.*guide_fraction, guide_num))

# Create the figure
fig, ax1 = plt.subplots(figsize=(14,4))
barheight = 0.10

y_guide = 0.25
y_science = 0.75

ymin = -0.05
ymax = 1.25

# Draw time sequence for spectroscopy, guiding
ax1.barh(y_science, science_width, 2*barheight, science_start, align='center', color='b')
ax1.barh(y_science, dither_width, 2*barheight, dither_start, align='center', color='c')
ax1.barh(y_guide, guide_width, barheight, guide_start, align='center', color='b')

# Mark twilight times
ax1.plot([twibeg_hours, twibeg_hours], [ymin, ymax], 'k--')
ax1.plot([twiend_hours, twiend_hours], [ymin, ymax], 'k--')

# Populate in case no data were obtained in each category
xlab = float(twibeg_hours - 4.)
if len(science_start) == 0:
    science_start.append(xlab + 4.)
    science_width.append(0.)

if len(guide_start) == 0:
    guide_start.append(xlab + 4.)
    guide_width.append(0.)

# Default label locations
y1, y2 = ax1.get_ylim()
ylab1 = y1 + 0.9*(y2-y1)
ylab2 = y1 + 0.8*(y2-y1)
ylab3 = y1 + 0.7*(y2-y1)
ylab4 = y1 + 0.6*(y2-y1)
lab1 = "Night: {0:.1f} hrs".format(twitot_hours)
lab2 = "Open: {0:.1f}%".format(100.*fractwiopen)
lab3 = "Science: {0:.1f}% ({1})".format(100.*science_fraction, science_num)
lab4 = "Guide: {0:.1f}% ({1})".format(100.*guide_fraction, guide_num)
ax1.text(xlab, ylab1, lab1, va='center')
ax1.text(xlab, ylab2, lab2, va='center')
ax1.text(xlab, ylab3, lab3, va='center')
ax1.text(xlab, ylab4, lab4, va='center')

# Dome status
ax1.plot(dometime, dome['shutter_upper'], ':')

# Adjust limits
ax1.set_ylim(ymin, ymax)   
ax1.set_xlim(twibeg_hours - 4.5, twiend_hours + 3)
# night_t = t - 1
night_t = Time(twibeg_mjd-1., format='mjd') 
ax1.set_title(night_t.strftime('%Y/%m/%d'))
ax1.set_xlabel('UT [hours]')
ax1.set_yticks([])

# Add points for seeing
ax2 = ax1.twinx()
ax2.set_ylabel("Seeing [arcsec]")
ax2.scatter(guidetime, guide['seeing'], c='gray') #, marker='o') 
y1, y2 = ax2.get_ylim()
ax2.set_ylim(0, y2) 

plt.savefig(outplot, bbox_inches="tight")
if args.verbose: 
    print("Wrote", outplot)

