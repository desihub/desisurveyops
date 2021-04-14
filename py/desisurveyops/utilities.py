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

def set_outdir(verbose=False): 
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
        outdir = '/global/cfs/cdirs/desi/users/martini/NightStats/'

    if not os.path.isdir(outdir): 
        os.mkdir(outdir) 

    # Directory for expid info per night
    nightlydir = os.path.join(outdir, 'NightlyData')

    if not os.path.isdir(nightlydir): 
        os.mkdir(nightlydir) 

    if verbose: 
        print("Output directory set to {}".format(outdir))

    return outdir


def set_datadir(verbose=False): 
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

    datadir = '/global/cfs/cdirs/desi/spectro/data/'

    if not os.path.isdir(datadir): 
        print("Error: root data directory {} not found".format(datadir))
        exit

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


def calc_sciencelist(night, clobber=False, verbose=False): 
    '''
    Create json files with start times, exptimes, and other information 
    for spectra for a single night. Outputs have the form specdata20201224.json. 

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

    # Output directory
    outdir = set_outdir()

    # Directory with raw data
    datadir = set_datadir()
    nightdir = os.path.join(datadir, str(night)) 

    # Names for output json file: 
    filename = "specdata" + str(night) + ".json"
    specdatafile = os.path.join(outdir, 'NightlyData', filename) 

    # See if json file for this night already exists: 
    if not os.path.isfile(specdatafile) or clobber: 
        # List of all directories and expids: 
        expdirs = glob(nightdir + "/*")
        expids = expdirs.copy()
        for i in range(len(expdirs)):
            expids[i] = expdirs[i][expdirs[i].find('000')::]

        # Print warning if there are no data
        if len(expids) == 0: 
            print("Warning: No observations found") 

        # Get all spec observations: 
        specdata = {}
        for expid in expids: 
            tmpdir = os.path.join(nightdir, expid)
            scifiles = (glob(tmpdir + "/desi*"))
            if len(scifiles) > 0:  
               hhh = fits.open(scifiles[0])
               if hhh[1].header['DATE-OBS'] == None:
                   continue
               specdata[expid] = {}
               specdata[expid]['DATE-OBS'] = Time(hhh[1].header['DATE-OBS']).mjd
               specdata[expid]['OBSTYPE'] = hhh[1].header['OBSTYPE']
               specdata[expid]['FLAVOR'] = hhh[1].header['FLAVOR']
               specdata[expid]['PROGRAM'] = hhh[1].header['PROGRAM']
               specdata[expid]['EXPTIME'] = hhh[1].header['EXPTIME']
               try: 
                   specdata[expid]['DOMSHUTU'] = hhh[1].header['DOMSHUTU']
               except KeyError: 
                   specdata[expid]['DOMSHUTU'] = 'None'
               try: 
                   specdata[expid]['PMCOVER'] = hhh[1].header['PMCOVER']
               except KeyError: 
                   specdata[expid]['PMCOVER'] = 'None'
        with open(specdatafile, 'w') as fp:
            json.dump(specdata, fp) 

        if verbose: 
            print("Wrote", specdatafile) 


def calc_guidelist(night, clobber=False, verbose=False): 
    '''
    Create json files with start times, exptimes, and other information 
    about guiding for a single night. Outputs have the form guidedata20201224.json. 

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

    outdir = set_outdir()
    datadir = set_datadir()
    nightdir = os.path.join(datadir, str(night)) 
    twibeg_mjd, twiend_mjd = get_twilights(int(night))
        
    # Names for output json file: 
    filename = "guidedata" + str(night) + ".json"
    guidedatafile = os.path.join(outdir, 'NightlyData', filename) 


    # See if json file for this night already exists: 
    if not os.path.isfile(guidedatafile) or clobber: 
        # List of all directories and expids: 
        expdirs = glob(nightdir + "/*")
        expids = expdirs.copy()
        for i in range(len(expdirs)):
            expids[i] = expdirs[i][expdirs[i].find('000')::]

        # Print warning if there are no data
        if len(expids) == 0: 
            print("Warning: No observations found") 

        guidedata = {}
        for expid in expids:
            # tmpdir = os.path.join(nightdir, expid, "guide-" + expid + ".fits.fz")
            # guidefiles = (glob(tmpdir + "/guide-" + expid + ".fits.fz"))
            # guidefile = tmpdir + "/guide-" + expid + ".fits.fz"
            guidefile = os.path.join(nightdir, expid, "guide-" + expid + ".fits.fz")
            #if len(guidefiles) > 0:
            if os.path.isfile(guidefile):
                try:
                    hhh = fits.open(guidefile)
                except OSError: 
                    if verbose: 
                        print("OSError with {}".format(guidefile))
                    continue
                guidedata[expid] = {}
                t1 = Time( find_dateobs(hhh) ).mjd
                try: 
                    t2 = Time(hhh['GUIDE0T'].data[-1]['DATE-OBS']).mjd
                except KeyError:
                    t2 = t1  # Need to fix this
                guidedata[expid]['GUIDE-START'] = t1
                guidedata[expid]['GUIDE-STOP'] = t2
                try:
                    guidedata[expid]['DOMSHUTU'] = hhh[0].header['DOMSHUTU']
                except KeyError:
                    guidedata[expid]['DOMSHUTU'] = 'None'
                try:
                    guidedata[expid]['PMCOVER'] = hhh[0].header['PMCOVER']
                except KeyError:
                    guidedata[expid]['PMCOVER'] = 'UNKNOWN'
                try:
                    guidedata[expid]['OBSTYPE'] = hhh[0].header['OBSTYPE']
                except KeyError:
                    guidedata[expid]['OBSTYPE'] = 'None'
                try:
                    guidedata[expid]['FLAVOR'] = hhh[0].header['FLAVOR']
                except KeyError:
                    guidedata[expid]['FLAVOR'] = 'None'
                try:
                    guidedata[expid]['EXPTIME'] = hhh[0].header['EXPTIME']
                except KeyError:
                    guidedata[expid]['EXPTIME'] = 0.
        with open(guidedatafile, 'w') as fp:
            json.dump(guidedata, fp)

        if verbose: 
            print("Wrote", guidedatafile) 


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

def get_twilights(night, verbose=False):
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

    # Calculate astronomical twilight times
    desi.horizon = '-18'
    beg_twilight=desi.previous_setting(ephem.Sun(), use_center=True) # End astro twilight
    end_twilight=desi.next_rising(ephem.Sun(), use_center=True) # Begin astro twilight
    twibeg = Time( beg_twilight.datetime(), format='datetime').mjd
    twiend = Time( end_twilight.datetime(), format='datetime').mjd

    if verbose: 
        print("Evening twilight: ", beg_twilight, "UT") 
        print("Morning twilight: ", end_twilight, "UT") 

    return twibeg, twiend


def get_dometelemetry(night, verbose=False):
    '''
    Retrieve dome telemetry data for a given night.

    Parameters
    ----------
    night : int
        night in form 20210320 for 20 March 2021
    verbose : bool
        print verbose output?

    Returns
    -------
    dome : record array
        dome telemetry stream for night
    '''

    # connect to telemetry database
    conn = psycopg2.connect(host="db.replicator.dev-cattle.stable.spin.nersc.org", port="60042", database="desi_dev", user="desi_reader")

    # set query limits as 3 hours before/after twilight at beginning/end of night
    twi1, twi2 = get_twilights(night)
    twibeg = Time( Time(twi1, format='mjd'), format='datetime')
    twiend = Time( Time(twi2, format='mjd'), format='datetime')
    query_start = twibeg - (3./24.)
    query_stop = twiend + (3./24.)

    # query for dome data
    domedf = pd.read_sql_query(f"SELECT dome_timestamp,shutter_upper,mirror_cover FROM environmentmonitor_dome WHERE time_recorded >= '{query_start}' AND time_recorded < '{query_stop}'", conn)
    dome = domedf.to_records()

    # convert dome_timestamp to a Time object in MJD
    dome['dome_timestamp'] = np.array([Time(t).mjd for t in dome['dome_timestamp']])

    return dome

def get_ccdtelemetry(night, verbose=False):
    '''
    Retrieve spectrograph ccd telemetry data for a given night.

    Parameters
    ----------
    night : int
        night in form 20210320 for 20 March 2021
    verbose : bool
        print verbose output?

    Returns
    -------
    ccd : record array
        ccd telemetry stream for night
    '''

    # connect to telemetry database
    conn = psycopg2.connect(host="db.replicator.dev-cattle.stable.spin.nersc.org", port="60042", database="desi_dev", user="desi_reader")

    # set query limits as 3 hours before/after twilight at beginning/end of night
    twi1, twi2 = get_twilights(night)
    twibeg = Time( Time(twi1, format='mjd'), format='datetime')
    twiend = Time( Time(twi2, format='mjd'), format='datetime')
    query_start = twibeg - (3./24.)
    query_stop = twiend + (3./24.)

    specdf = pd.read_sql_query(f"SELECT * FROM spectrographs_ccds WHERE time_recorded >= '{query_start}' AND time_recorded < '{query_stop}'", conn)
    spec_ccds = specdf.to_records()

    # Convert dome_timestamp to a Time object in MJD
    spec_ccds['time_recorded'] = np.array([Time(t).mjd for t in spec_ccds['time_recorded']])

    return spec_ccds

def get_teltelemetry(night, verbose=False):
    '''
    Retrieve telescope telemetry data for a given night.

    Parameters
    ----------
    night : int
        night in form 20210320 for 20 March 2021
    verbose : bool
        print verbose output?

    Returns
    -------
    ccd : record array
        ccd telemetry stream for night
    '''

    # connect to telemetry database
    conn = psycopg2.connect(host="db.replicator.dev-cattle.stable.spin.nersc.org", port="60042", database="desi_dev", user="desi_reader")

    # set query limits as 3 hours before/after twilight at beginning/end of night
    twi1, twi2 = get_twilights(night)
    twibeg = Time( Time(twi1, format='mjd'), format='datetime')
    twiend = Time( Time(twi2, format='mjd'), format='datetime')
    query_start = twibeg - (3./24.)
    query_stop = twiend + (3./24.)

    # Telescope data:
    teldf = pd.read_sql_query(f"SELECT * FROM tcs_info WHERE time_recorded >= '{query_start}' AND time_recorded < '{query_stop}'", conn)
    tel = teldf.to_records()

    # Convert dome_timestamp to a Time object in MJD
    tel['time_recorded'] = np.array([Time(t).mjd for t in tel['time_recorded']])

    return tel


def get_guidetelemetry(night, verbose=False):
    '''
    Retrieve guider telemetry data for a given night.

    Parameters
    ----------
    night : int
        night in form 20210320 for 20 March 2021
    verbose : bool
        print verbose output?

    Returns
    -------
    guide : record array
        guide telemetry stream for night
    '''

    # connect to telemetry database
    conn = psycopg2.connect(host="db.replicator.dev-cattle.stable.spin.nersc.org", port="60042", database="desi_dev", user="desi_reader")

    # set query limits as 3 hours before/after twilight at beginning/end of night
    twi1, twi2 = get_twilights(night)
    twibeg = Time( Time(twi1, format='mjd'), format='datetime')
    twiend = Time( Time(twi2, format='mjd'), format='datetime')
    query_start = twibeg - (3./24.)
    query_stop = twiend + (3./24.)

    # query for guider data
    guidedf = pd.read_sql_query(f"SELECT time_recorded,seeing,meanx,meany FROM guider_summary WHERE time_recorded >= '{query_start}' AND time_recorded < '{query_stop}'", conn)
    guide = guidedf.to_records()

    return guide


def calc_domevals(night, verbose=False):
    '''
    Calculate basic dome parameters for a given night.

    Parameters
    ----------
    night : int
        night in form 20210320 for 20 March 2021
    verbose : bool
        print verbose output?

    Returns
    -------
    dome_firstopen : float
        first time the dome opened (MJD)
    dome_lastclose : float
        last time dome closed (MJD)
    fractwiopen : float
        fraction of the time the dome was open between 18 deg twilights 
    twiopen_hrs : float
        hours the dome was open between 18 deg twilights
    '''

    twibeg_mjd, twiend_mjd = get_twilights(int(night))
    startdate = int(twibeg_mjd)
    twibeg_hours = 24.*(twibeg_mjd - startdate)
    twiend_hours = 24.*(twiend_mjd - startdate)
    twitot_hours = twiend_hours - twibeg_hours

    # Get dome status:
    dome = get_dometelemetry(night)
    dometime = np.array([ (t - startdate)*24 for t in dome['dome_timestamp']])
    dome_open = np.array([t for t in dome['shutter_upper']], dtype=bool)

    # Compute total time with dome open between twilights:
    nmask = dometime >= twibeg_hours
    nmask = nmask*(dometime <= twiend_hours) # time between twilights
    dmask = nmask*dome_open # dome open
    open_fraction = np.sum(dmask)/np.sum(nmask)
    fractwiopen = open_fraction

    # Total time dome was open between twilights
    twiopen_hrs = open_fraction*twitot_hours

    # get times the dome first opened and last closed 
    # shutter_upper == 1 means dome was open
    open_indices = np.where(dome_open)[0]

    if verbose: 
        print("len(open_indices) = ", len(open_indices))

    # check if dome didn't open at all
    if len(open_indices) == 0:
        return 0., 0., 0., 0.

    # time stamps when dome first opened and last closed 
    firstopen = dome['dome_timestamp'][open_indices[0]]
    lastclose = dome['dome_timestamp'][open_indices[-1]]

    if verbose: 
        print("MJD for dome first open, last close:", firstopen, lastclose)
        print("UT for dome first open, last close:", Time(firstopen, format='mjd').datetime, Time(lastclose, format='mjd').datetime )
        print("Dome open: {0:.3f} Dome close: {1:.3f} Fraction of twilight with dome open = {2:.2f}".format(firstopen, lastclose, fractwiopen) )

    return firstopen, lastclose, fractwiopen, twiopen_hrs


def calc_obstimes(night, verbose=False, clobber=False):
    '''
    Calculate start and lengths of science, dither, and guide exposures 

    Parameters
    ----------
    night : int
        night in form 20210320 for 20 March 2021
    verbose : bool
        print verbose output?
    clobber : bool
        overwrite existing file

    Returns
    -------
    science_start : float
        start times of science exposures in UT hours
    science_width : float
        length of science exposures in UT hours
    dither_start : float
        start times of dither exposures in UT hours
    dither_width : float
        length of dither exposures in UT hours
    guide_start : float
        start times of guide exposures in UT hours
    guide_width : float
        length of guide exposures in UT hours
    '''

    outdir = set_outdir()

    # Read in the data
    specfilename = "specdata" + str(night) + ".json"
    specdatafile = os.path.join(outdir, 'NightlyData', specfilename)
    if os.path.isfile(specdatafile) and not clobber:
        specdata = read_json(specdatafile)
    else:
        print("Note: {} not found ... creating it".format(specdatafile))
        calc_sciencelist(night)
        specdata = read_json(specdatafile)

    guidefilename = "guidedata" + str(night) + ".json"
    guidedatafile = os.path.join(outdir, 'NightlyData', guidefilename)
    if os.path.isfile(guidedatafile) and not clobber:
        guidedata = read_json(guidedatafile)
    else:
        print("Note: {} not found ... creating it".format(guidedatafile))
        calc_guidelist(night)
        guidedata = read_json(guidedatafile)

    twibeg_mjd, twiend_mjd = get_twilights(int(night))
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
    
    return science_start, science_width, dither_start, dither_width, guide_start, guide_width

def get_totobs(start, length, twibeg_hours, twiend_hours, verbose=False): 
    '''
    Calculate the total observing time between twilights:

    Parameters
    ----------
    start : float
        start times of exposures in UT hours
    length : float
        length of exposures in UT hours
    twibeg_hours : float
        time of twilight at the beginning of the night in UT hours
    twiend_hours : float
        time of twilight at the end of the night in UT hours
    verbose : bool
        print verbose output?

    Returns
    -------
    obshours : float
        total observing hours between twilights 
    '''

    obshours = 0.
    for i in range(len(start)):
        t1 = start[i]
        t2 = t1 + length[i]
        if t1 <= twibeg_hours and t2 >= twibeg_hours:
            obshours += t2-twibeg_hours
        elif t1 >= twibeg_hours and t2 <= twiend_hours:
            obshours += t2-t1
        elif t1 <= twiend_hours and t2 >= twiend_hours:
            obshours += t1-twiend_hours

    return obshours


def calc_science(night, verbose=False):
    '''
    Calculate quantities about science exposures for a given night


    Parameters
    ----------
    night : int
        night in form 20210320 for 20 March 2021
    verbose : bool
        print verbose output?

    Returns
    -------
    science_first : float
        start time of first spectroscopic exposure in MJD
    science_last : float
        end time of last spectroscopic exposure in MJD
    scitwi_hours : float
        number of hours of spectroscopic exposures between twilights in UT hours 
    scitot_hours : float
        total number of hours of spectroscopic exposures in UT hours (not limited by twilight)
    '''

    twibeg_mjd, twiend_mjd = get_twilights(int(night))
    startdate = int(twibeg_mjd)
    twibeg_hours = 24.*(twibeg_mjd - startdate)
    twiend_hours = 24.*(twiend_mjd - startdate)
    twitot_hours = twiend_hours - twibeg_hours

    science_start, science_width, dither_start, dither_width, guide_start, guide_width = calc_obstimes(night)

    # Science hours between twilights 
    scitwi_hours = get_totobs(science_start, science_width, twibeg_hours, twiend_hours)

    # Calculate MJD of start of first science exposure:
    if len(science_start) == 0:
        return 0., 0., 0., 0.
    science_first = min(science_start)/24. + startdate
    indx = np.where( np.asarray(science_start) == max(science_start))[0][0]
    science_last = (science_start[indx] + science_width[indx])/24. + startdate

    # Science hours total
    scitot_hours = np.sum(science_width) 

    # Open shutter time between twilights
    fracscitwi = scitwi_hours/(twiend_hours - twibeg_hours)

    return science_first, science_last, scitwi_hours, scitot_hours

def get_skydata(expidlist, night, verbose=False):
    '''
    Determine start and lengths of all sky frames associated with expidlist

    Parameters
    ----------
    expidlist : list
        list of expids 
    night : int
        night in form 20210320 for 20 March 2021
    verbose : bool
        print verbose output?

    Returns
    -------
    sky_start : list
        start times of each sky exposure in UT hours
    sky_width : list
        lengths of each sky exposure in UT hours
    '''

    datadir = set_datadir()
    nightdir = os.path.join(datadir, str(night)) 
    twibeg_mjd, twiend_mjd = get_twilights(int(night))
    startdate = int(twibeg_mjd)

    if verbose:
        print(night, startdate, twibeg_mjd, nightdir)

    sky_start = []
    sky_width = []
    for expid in expidlist:
        tmpdir = os.path.join(nightdir, expid.zfill(8))
        skyfiles = (glob(tmpdir + "/sky-" + expid.zfill(8) + ".fits.fz"))
        if len(skyfiles) > 0:
            shh = fits.open(skyfiles[0])
            for i in range(len(shh['SKYCAM0T'].data)):
                tt = Time(shh['SKYCAM0T'].data[i]['DATE-OBS']).mjd
                sky_start.append((tt - startdate)*24 )
                sky_width.append(shh['SKYCAM0T'].data[i]['EXPTIME']/3600.)
    return sky_start, sky_width


def get_guidedata(expidlist, night, acqonly=False, startonly=False):
    '''
    Determine start and lengths of all sky frames associated with expidlist

    Parameters
    ----------
    expidlist : list
        list of expids 
    night : int
        night in form 20210320 for 20 March 2021
    acqonly : bool 
        report only the times for the acquisition images
    startonly : bool 
        report only the start of guiding

    Returns
    -------
    guide_start : list
        start times of each guide exposure in UT hours
    guide_width : list
        lengths of each guide exposure in UT hours
    '''

    datadir = set_datadir()
    nightdir = os.path.join(datadir, str(night)) 
    twibeg_mjd, twiend_mjd = get_twilights(int(night))
    startdate = int(twibeg_mjd)

    guide_start = []
    guide_width = []
    for expid in expidlist:
        tmpdir = os.path.join(nightdir, expid.zfill(8))
        guidefiles = (glob(tmpdir + "/guide-" + expid.zfill(8) + ".fits.fz"))
        if len(guidefiles) > 0:
            ghh = fits.open(guidefiles[0])
            if acqonly:
                tt = Time(ghh['GUIDE0T'].data[0]['DATE-OBS']).mjd
                guide_start.append((tt - startdate)*24 )
                guide_width.append(ghh['GUIDE0T'].data[0]['EXPTIME']/3600.)
            elif startonly:
                tt = Time(ghh['GUIDE0T'].data[1]['DATE-OBS']).mjd
                guide_start.append((tt - startdate)*24 )
                guide_width.append(ghh['GUIDE0T'].data[1]['EXPTIME']/3600.)
            else:
                for i in range(len(ghh['GUIDE0T'].data)):
                    tt = Time(ghh['GUIDE0T'].data[i]['DATE-OBS']).mjd
                    guide_start.append((tt - startdate)*24 )
                    guide_width.append(ghh['GUIDE0T'].data[i]['EXPTIME']/3600.)
    return guide_start, guide_width

def get_fvcdata(expidlist, night, verbose=False):
    '''
    Determine start and lengths of all FVC frames associated with expidlist

    Parameters
    ----------
    expidlist : list
        list of expids 
    night : int
        night in form 20210320 for 20 March 2021

    Returns
    -------
    fvc_start : list
        start times of each FVC exposure in UT hours
    fvc_width : list
        lengths of each FVC exposure in UT hours
    '''

    datadir = set_datadir()
    nightdir = os.path.join(datadir, str(night)) 
    twibeg_mjd, twiend_mjd = get_twilights(int(night))
    startdate = int(twibeg_mjd)

    fvc_start = []
    fvc_width = []
    for expid in expidlist:
        tmpdir = os.path.join(nightdir, expid.zfill(8))
        fvcfiles = (glob(tmpdir + "/fvc-" + expid.zfill(8) + ".fits.fz"))
        if len(fvcfiles) > 0:
            if verbose:
                print(fvcfiles[0])
            fhh = fits.open(fvcfiles[0])
            tt = Time(fhh['F0000'].header['DATE']).mjd
            fvc_start.append((tt - startdate)*24 )
            fvc_width.append(fhh['F0000'].header['EXPTIME']/3600.)
            try:   # Splits do not have a second FVC image
                tt = Time(fhh['F0001'].header['DATE']).mjd
                fvc_start.append((tt - startdate)*24 )
                fvc_width.append(fhh['F0001'].header['EXPTIME']/3600.)
            except KeyError: 
                continue
    return fvc_start, fvc_width

def get_scidata(expidlist, night):
    '''
    Determine start and lengths of all science frames associated with expidlist

    Parameters
    ----------
    expidlist : list
        list of expids 
    night : int
        night in form 20210320 for 20 March 2021

    Returns
    -------
    spec_start : list
        start times of each spec exposure in UT hours
    spec_width : list
        lengths of each spec exposure in UT hours
    spec_expid : list
        expid of each spec exposure
    '''

    datadir = set_datadir()
    nightdir = os.path.join(datadir, str(night)) 
    twibeg_mjd, twiend_mjd = get_twilights(int(night))
    startdate = int(twibeg_mjd)

    spec_start = []
    spec_width = []
    spec_expid = []

    # Identify expids that corresponds to science exposures:
    for expid in expidlist:
        tmpdir = os.path.join(nightdir, expid.zfill(8))
        scifiles = (glob(tmpdir + "/desi*"))
        if len(scifiles) > 0:
            hhh = fits.open(scifiles[0])
            tt = Time(''.join(hhh['SPECTCONS'].data['DATE-OBS'][0])).mjd
            try:
                flavor = hhh[1].header['FLAVOR']
                program = hhh[1].header['PROGRAM']
                obstype = hhh[1].header['OBSTYPE']
                exptime = hhh[1].header['EXPTIME']
            except:
                continue
            if obstype == 'SCIENCE' and flavor == 'science' and 'Dither' not in program and exptime > 10.:
                #spec_expids.append(expid)
                spec_start.append((tt - startdate)*24 )
                spec_width.append(hhh['SPEC'].header['EXPTIME']/3600.)
                spec_expid.append(expid)

    return spec_start, spec_width, spec_expid


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

    outdir = set_outdir()
    datadir = set_datadir()
    nightdir = os.path.join(datadir, str(night))

    expfiles = glob(nightdir + "/*/desi*")

    expidlist = []
    for i in range(len(expfiles)):
        i1 = expfiles[i].rfind('desi-') + 5
        i2 = expfiles[i].rfind('.fits')
        expidlist.append(expfiles[i][i1:i2].lstrip('0'))

    return expidlist


def calc_interexp(night, minexptime=300., clobber=False, verbose=False):
    '''
    Calculate the interexposure times

    Parameters
    ----------
    night : int
        night in form 20210320 for 20 March 2021
    minexptime : float
        minimum science exposure time to consider in calculation
    verbose : bool
        print verbose output?
    clobber : bool
        overwrite existing file

    Returns
    -------
    interexp : 2-d float array
        columns are DATE-OBS in MJD, DATE-OBS in UT hrs, EXPTIME in s, EXPID, interexposure time in s
    '''

    # Read in the data
    outdir = set_outdir()
    specfilename = "specdata" + str(night) + ".json"
    specdatafile = os.path.join(outdir, 'NightlyData', specfilename)
    if os.path.isfile(specdatafile) and not clobber:
        specdata = read_json(specdatafile)
    else:
        print("Note: {} not found ... creating it".format(specdatafile))
        calc_sciencelist(night)
        specdata = read_json(specdatafile)

    # Take about the dict
    dateobs_mjd = []
    exptime = []
    expid = []
    for i, key in enumerate(specdata.keys()):
        if specdata[key]['EXPTIME'] > minexptime:
            dateobs_mjd.append(specdata[key]['DATE-OBS'])
            exptime.append(specdata[key]['EXPTIME'])
            expid.append(key)

    twibeg_mjd, twiend_mjd = get_twilights(int(night))
    startdate = int(twibeg_mjd)

    scidata = np.zeros([len(dateobs_mjd), 5])
    for i in range(len(dateobs_mjd)):
        dateobs_hrs = (dateobs_mjd[i] - startdate)*24.
        scidata[i] = np.asarray([dateobs_mjd[i], dateobs_hrs, exptime[i], expid[i], 0.])
            
    # Sort by expid:
    columnIndex = 3
    sortdata = scidata[scidata[:,columnIndex].argsort()]

    # Calculate interexposure time between each successive pair
    secperday = 24.*3600.
    interexp = []
    for i in range(len(sortdata) - 1):
        sortdata[i][4] = secperday*(sortdata[i+1][0] - sortdata[i][0] - sortdata[i][1]/secperday)
        # interexp.append(secperday*(sortdata[i+1][0] - sortdata[i][0] - sortdata[i][1]/secperday ))

    return sortdata[:-1]
