#!/usr/bin/env python
  
import os, sys
import numpy as np
from astropy.io import fits
import argparse
import matplotlib.pyplot as plt
from astropy.time import Time
import desisurveyops.utilities as utils

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                        description="""Generate summary plot of observing efficiency""")

parser.add_argument('-f','--filename', type=str, default="ObservingEfficiency.png",
                    required=False, help='Output file name (default nightlystats.fits)')

parser.add_argument('-s','--summaryfile', type=str, default="nightlystats.fits",
                    required=False, help='Summary file name (default nightlystats.fits)')

parser.add_argument('-c','--clobber', action='store_true', default=False, 
                    required=False,
                    help='Clobber (overwrite) output if it already exists?')

parser.add_argument('-v','--verbose', action='store_true', default=False, 
                    required=False,
                    help='Provide verbose output?')

args = parser.parse_args()

utils.plot_defaults()

datadir = utils.set_datadir()
outdir = utils.set_outdir()

# See if the output file already exists and clobber=True

outplot = os.path.join(outdir, "ObservingEfficiency.png")
if os.path.isfile(outplot) and not args.clobber: 
    print("Error: {} already exists and clobber = False".format(outplot))

# Read in the file created by "update_summary.py" 
if args.summaryfile is None:
    summaryfile = os.path.join(outdir, args.summaryfile)
else:
    summaryfile = os.path.join(outdir, "nightlystats.fits")

if os.path.isfile(summaryfile):
    hdu = fits.open(summaryfile)
else:
    print("Error: {} not found".format(summaryfile))
    exit


# Create datetime array for plots
dates = np.array([Time(t, format='mjd').datetime for t in hdu[1].data['TWIBEG']])


# Compute interexposure values
interexp_min = []
interexp_med = []
for i in range(len(hdu[1].data['NIGHT'])):
    interexp = utils.calc_interexp(hdu[1].data['NIGHT'][i])
    if len(interexp) > 1:
        interexp_min.append(np.min(interexp))
        interexp_med.append(np.median(interexp))
    else:
        interexp_min.append(-60.)
        interexp_med.append(-60.)

# Create the plot
fig, axarr = plt.subplots(2, 1, figsize=(14,9), sharex=True)
axarr[0].set_ylabel("Interexposure Time [s]")
axarr[0].plot(dates, interexp_min, 'ko', fillstyle='none', label='Minimum')
axarr[0].plot(dates, interexp_med, 'b^', label="Median")
axarr[0].set_ylim(0, 900)
axarr[0].legend()
#axarr[1].plot(hdu[1].data['TWIBEG'], hdu[1].data['DOMEFRAC'], 'o', label="Dome Open Fraction")
#axarr[1].plot(hdu[1].data['TWIBEG'], hdu[1].data['SCTWFRAC'], 'o', label="Observing Efficiency")
axarr[1].plot(dates, hdu[1].data['DOMEFRAC'], 'ko', fillstyle='none', label="Dome Open")
axarr[1].plot(dates, hdu[1].data['SCTWFRAC'], 'b^', label="Observing Efficiency")
axarr[1].set_xlabel("Night")
axarr[1].set_ylabel("Fraction")
axarr[1].legend()
plt.tight_layout()
plt.savefig(outplot, bbox_inches="tight")
if args.verbose: 
    print("Wrote", outplot)

