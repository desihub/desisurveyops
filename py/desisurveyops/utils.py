#!/usr/bin/env python
  
import os, sys
import numpy as np
from astropy.io import fits
from glob import glob
import matplotlib.pyplot as plt
from astropy.time import Time
import json
import ephem
import psycopg2
import pandas as pd

def get_outdir(verbose=False): 
    '''
    Set the root directory for the output and a NightlyData/ subdirectory

    Parameters
    ----------
    verbose : bool
        Print verbose output? 

    Returns
    -------
    outdir : string
        root output directory 
    '''

    try:
        outdir = os.environ['DESINIGHTSTATS']
    except KeyError:
        print('  Error: Did not find environment variable DESINIGHTSTATS for output directory') 
        raise RuntimeError

    if not os.path.isdir(outdir): 
        try: 
            os.makedirs(outdir) 
        except PermissionError:
            raise RuntimeError("Could not create directory {}".format(outdir))
            exit(1)


    # Directory for expid info per night
    nightlydir = os.path.join(outdir, 'NightlyData')

    if not os.path.isdir(nightlydir): 
        try: 
            os.makedirs(nightlydir) 
        except PermissionError:
            raise RuntimeError("Could not create directory {}".format(nightlydir))
            exit(1)

    if verbose: 
        print("Output directory set to {}".format(outdir))

    return outdir


def get_rawdatadir(verbose=False): 
    '''
    Set the root directory for the raw data

    Parameters
    ----------
    verbose : bool
        Print verbose output? 

    Returns
    -------
    outdir : string
        root output directory 
    '''

    try: 
        datadir = os.environ['DESI_SPECTRO_DATA']
        # Note this is /global/cfs/cdirs/desi/spectro/data/ at NERSC
    except KeyError: 
        print('  Error: Did not find environment variable DESI_SPECTRO_DATA')
        raise RuntimeError

    if not os.path.isdir(datadir): 
        print("Error: root data directory {} not found".format(datadir))
        raise RuntimeError

    if verbose: 
        print("Raw data directory set to {}".format(datadir))

    return datadir

        
def read_json(filename: str):
    '''
    Read in a json file

    Parameters
    ----------
    filename : string
        File to read in and convert to dictionary

    Returns
    -------
    output : dict
        file as dictionary

    '''

    with open(filename) as fp:
        return json.load(fp)


def plot_defaults():
    '''
    Load reasonable matplotlib font sizes

    Parameters
    ----------
    none

    Returns
    -------
    none
    '''

    # matplotlib settings 
    SMALL_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 18

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('lines', linewidth=2)
    plt.rc('axes', linewidth=2)
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def find_dateobs(hhh):
    '''
    Delve deep into hdu to find keyword

    '''
    try:
        val = hhh['GUIDE0T'].data[0]['DATE-OBS']
    except KeyError:
        try:
            val = hhh['GUIDE0'].data[0]['DATE-OBS']
        except IndexError:
            try:
                val = hhh['GUIDE0'].header['DATE-OBS']
            except KeyError:
                print("KeyError")
    return val

def get_twilights(night, alt=-18., verbose=False):
    '''
    Calculate and return 18 deg twilight values for the start and end of the night.

    Parameters
    ----------
    night : int
        night in form 20210320 for 20 March 2021
    verbose : bool
        print verbose output? 

    Returns
    -------
    twibeg : float
        time of twilight in MJD at beginning of the night
    twiend : float
        time of twilight in MJD at the end of the night
    '''

    night = int(night)
    nightstr = str(night)
    timestr = "{0}-{1}-{2}T00:00:00.0".format(nightstr[:4], nightstr[4:6], nightstr[6:8])
    t = Time(timestr, format='isot', scale='utc') + 1

    # Set observatory lat,long to calculate twilight
    desi = ephem.Observer()
    desi.lon = '-111.59989'
    desi.lat = '31.96403'
    desi.elev = 2097.
    desi.date = t.strftime('%Y/%m/%d 7:00')

    # Calculate twilight times
    desi.horizon = str(alt)
    beg_twilight=desi.previous_setting(ephem.Sun(), use_center=True) # End astro twilight
    end_twilight=desi.next_rising(ephem.Sun(), use_center=True) # Begin astro twilight
    twibeg = Time( beg_twilight.datetime(), format='datetime').mjd
    twiend = Time( end_twilight.datetime(), format='datetime').mjd

    if verbose: 
        print("Evening twilight: ", beg_twilight, "UT") 
        print("Morning twilight: ", end_twilight, "UT") 

    return twibeg, twiend


def getexpidlist(night):
    '''
    Get a list of all expids from a given night

    Parameters
    ----------
    night : int
        night in form 20210320 for 20 March 2021

    Returns
    -------
    expidlist : list
        list of expids
    '''

    outdir = get_outdir()
    datadir = get_rawdatadir()
    nightdir = os.path.join(datadir, str(night))

    expfiles = glob(nightdir + "/*/desi*")

    expidlist = []
    for i in range(len(expfiles)):
        i1 = expfiles[i].rfind('desi-') + 5
        i2 = expfiles[i].rfind('.fits')
        expidlist.append(expfiles[i][i1:i2].lstrip('0'))

    return expidlist


