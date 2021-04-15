import os, sys
import numpy as np
from astropy.table import QTable
import matplotlib.pyplot as plt
import ephem
import desisurveyops.utils as utils
import desisurveyops.obsdata as od


def init_summary(night, outfilename, clobber=False, verbose=False):
    '''
    Create an empty fits table will contain summary information for each night.

    Parameters
    ----------
    night : int
        night in form 20210320 for 20 March 2021
    clobber : bool
        overwrite existing file
    verbose : bool
        provide verbose output

    Returns
    -------
    none
    '''

    outdir = utils.get_outdir()
    outputfile = os.path.join(outdir, outfilename)

    if os.path.isfile(outputfile) and not clobber:
        print("init_summary(): {} already exists and clobber=False".format(outputfile))
        return

    t = QTable(calc_row(night), 
    names=('NIGHT', 'TWIBEG', 'TWIEND', 'DOMEOPEN', 'DOMECLOSE', 'DOMEFRAC', 'DOMEHRS', 'SCSTART', 'SCSTOP', 'SCTWTOT', 'SCTOT', 'SCTWFRAC'),
    meta={'Name': 'Nightly Efficiency Summary Table'})

    t.write(outputfile, overwrite=clobber)

    if verbose:
        print("Initialized summary table {}".format(outputfile))

def calc_row(night):
    '''
    Calculate information about observing efficiency for a single night

    Parameters
    ----------
    night : int
        night in form 20210320 for 20 March 2021
    verbose : bool
        provide verbose output

    Returns
    -------
    rowdata : list
        info to update row of summary table for the night
    '''

    twibeg_mjd, twiend_mjd = utils.get_twilights(night) 
    firstopen, lastclose, fractwiopen, twiopen_hrs = od.calc_domevals(night)
    science_first, science_last, scitwi_hrs, scitot_hrs = od.calc_science(night) 
    print(twiopen_hrs)
    if twiopen_hrs > 0.: 
        fracscitwi = scitwi_hrs/twiopen_hrs
    else:
        fracscitwi = 0.

    # names=('NIGHT', 'TWIBEG', 'TWIEND', 'DOMEOPEN', 'DOMECLOSE', 'DOMEFRAC', 'DOMEHRS', 'SCSTART', 'SCSTOP', 'SCTWTOT', 'SCTOT', 'SCTWFRAC'),
    c0 = np.array([night], dtype=np.int32)
    c1 = np.array([twibeg_mjd], dtype=float)
    c2 = np.array([twiend_mjd], dtype=float)
    c3 = np.array([firstopen], dtype=float)
    c4 = np.array([lastclose], dtype=float)
    c5 = np.array([fractwiopen], dtype=float)
    c6 = np.array([twiopen_hrs], dtype=float)
    c7 = np.array([science_first], dtype=float)
    c8 = np.array([science_last], dtype=float)
    c9 = np.array([scitwi_hrs], dtype=float)
    c10 = np.array([scitot_hrs], dtype=float)
    c11 = np.array([fracscitwi], dtype=float)

    newrow = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11]

    return newrow


def update_table(night, outfilename, clobber=False, verbose=False): 
    '''
    Update the summary table with data for 'night' 

    Parameters
    ----------
    night : int
        night in form 20210320 for 20 March 2021
    clobber : bool
        overwrite existing data
    verbose : bool
        provide verbose output

    Returns
    -------
    none
    '''

    outdir = utils.get_outdir()
    outputfile = os.path.join(outdir, outfilename)

    t = QTable.read(outputfile)

    if int(night) in t['NIGHT']:
        print("Found {} in table".format(night))
        if not clobber:
            print("Data for {} already in table and clobber=False".format(night))
            return
        else:
            print("clobber=True")
            indx = np.where(t['NIGHT'] == night)[0][0]
            t.remove_row(indx)

    newrow = calc_row(night)
    t.add_row(newrow)
    t.write(outputfile, overwrite=True)

    if verbose:
        print("Added new row: ", newrow)


