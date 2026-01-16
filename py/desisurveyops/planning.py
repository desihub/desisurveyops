"""This module includes a simple code to perform field and tile planning on a given night. The utilities available are essentially equivalent to STARALT (https://astro.ing.iac.es/staralt/) but with a few additional customized options useful for DESI."""

import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import ephem
import pytz

from datetime import datetime, timedelta

from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.coordinates import solar_system_ephemeris, get_body

from astropy.table import Table
from astropy.time import Time

def get_ephemerides(observing_date, offset=7*u.hour, obs_lon='-111.6', obs_lat='31.96333333', obs_elev=2120):
    """
    Get ephemerides for a site on some particular observing date. Returns sunset, 6, 10, 12 deg twighlight, and sunrise times.

    Parameters
    ----------
    observing_date: astropy.Time
        The date at the start of the night of observing.
    offset: astropy.Quantity
        A time offset from UTC. obs (MST) is 7 hr behind UTC.
    obs_lon: str
        Observatory longitude in decimal degrees.
    obs_lat: str
        Observatory latitude in decimal degrees.
    obs_elev: float
        Observatory elevation (meters a.s.l.).

    Returns
    -------
    sunset: astropy.Time
        Local time of sunset.
    dusk_06deg: astropy.Time
        Local time of 6 deg twilight.
    dusk_10deg: astropy.Time
        Local time of 10 deg twilight.
    dusk_12deg: astropy.Time
        Local time of 12 deg twilight (start of night).
    dawn_12deg: astropy.Time
        Local time of 12 deg twilight (end of night).
    sunrise: astropy.Time
        Local time of sunrise.
    """
    obs = ephem.Observer()
    obs.date = (observing_date + offset).to_datetime()
    obs.lon = obs_lon
    obs.lat = obs_lat
    obs.elev = obs_elev

    sun = ephem.Sun()

    sunset = Time(obs.previous_setting(sun).datetime()) - offset
    sunrise = Time(obs.next_rising(sun, use_center=True).datetime()) - offset

    obs.horizon = '-6:00'
    dusk_06deg = Time(obs.previous_setting(sun, use_center=True).datetime()) - offset
    dawn_06deg = Time(obs.next_rising(sun, use_center=True).datetime()) - offset

    obs.horizon = '-10:00'
    dusk_10deg = Time(obs.previous_setting(sun, use_center=True).datetime()) - offset
    dawn_10deg = Time(obs.next_rising(sun, use_center=True).datetime()) - offset

    obs.horizon = '-12:00'
    dusk_12deg = Time(obs.previous_setting(sun, use_center=True).datetime()) - offset
    dawn_12deg = Time(obs.next_rising(sun, use_center=True).datetime()) - offset

    return sunset, dusk_06deg, dusk_10deg, dusk_12deg, dawn_12deg, sunrise

def alt2am(alt):
    """Compute airmass given an altitude, using the Kasten and Young formula (Appl. Opt. 28:4735, 1989).
Parameters ---------
    alt : float or ndarray
        Altitude(s) in decimal degrees.

    Returns
    -------
    am : float or ndarray
        Airmass(es).
    """
    am = np.full_like(alt, np.inf, dtype=float)
    select = alt >= 0
    z = 90. - alt[select]
    am[select] = 1. / (np.cos(np.radians(z)) + 0.50572*(6.07995 + 90 - z)**-1.6364)
    return am

def am2alt(am):
    """Inversion of the altitude-airmass calculation using linear interpolation.

    Parameters
    ---------
    am : float or ndarray
        Airmass(es).

    Returns
    -------
    alt : float or ndarray
        Altitude(s) in decimal degrees.
    """
    _alt = np.arange(90, -1, -1)
    _am = alt2am(_alt)
    return np.interp(am, _am, _alt)

def plot_transit_kpno(night, fields_name, fields_ra, fields_dec, airmass_limit=None, highlight_fields=None, verbose=False):
    """Plot transits of one or more fields.

    Parameters
    ----------
    night: str
        Observing date in format YYYY-MM-DD.
    fields_name: list or ndarray
        Field names; could be object names, numerical tile IDs, etc.
    fields_ra: list or ndarray
        List of field RAs in decimal degrees.
    fields_dec: list or ndarray
        List of field declinations in decimal degrees.
    airmass_limit: None or float
        Compute total time of a field above some airmass.
    highight_fields: list
        List of fields to highlight in the plot (fields not listed will be grayed out).

    Returns
    -------
    fig: matplotlib.Figure
        Figure object for additional plotting/manipulation.
    """
    local_obs_date = Time(f'{night} 23:59')
    sunset, dusk06, dusk10, dusk12, dawn12, sunrise = get_ephemerides(local_obs_date)

    #- Set up the observations
    kpno = EarthLocation.of_site('Kitt Peak')
    observing_date = local_obs_date + 7*u.hour
    time_grid = observing_date + np.arange(-8*60, 8*60+1, 1) * u.minute
    local_time_grid = time_grid - 7*u.hour
    altaz = AltAz(location=kpno, obstime=time_grid)
    frame_kpno = AltAz(obstime=time_grid, location=kpno)

    #- Get the moon position
    moon = get_body('moon', time_grid, kpno, ephemeris='jpl')
    moon_altaz = moon.transform_to(frame_kpno)

    #- Plot the field transits
    fig, ax = plt.subplots(1,1, figsize=(11,8.5))
    ax.plot(local_time_grid.datetime, moon_altaz.alt.degree, ls='--', lw=2, color='k', label='Moon')
    
    lstyles = ['solid', 'dashed', 'dashdot', 'dotted']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']

    p = None
    for k, (field, ra, dec) in enumerate(zip(fields_name, fields_ra, fields_dec)):
        c = SkyCoord(ra=ra, dec=dec, unit='deg')
        c_altaz = c.transform_to(altaz)
        airmass = alt2am(c_altaz.alt.degree)
    
        i = np.argmax(c_altaz.alt)
        t_transit = local_time_grid[i]
        if verbose:
            print(f'{field} local transit time: {t_transit} (MJD: {Time(t_transit).mjd})')
    
        #- Time above airmass 1.5 or specified airmass limit.
        isvalid = airmass > 0
        amlimit = 1.5 if airmass_limit is None else airmass_limit
        isobservable = airmass[isvalid] <= amlimit
        if not np.any(isobservable):
            if verbose:
                print(f'Field of {field} is never at airmass > {amlimit}.\n')
            continue
        
        t0 = local_time_grid[isvalid][isobservable][0]
        t1 = local_time_grid[isvalid][isobservable][-1]
        imax = np.argmax(c_altaz.alt.degree)
        if verbose:
            print(f'Minimum airmass: {airmass[imax]:.2f}')
            print(f'Time above airmass {amlimit}: {t0} -> {t1} ({(t1-t0).sec/60:.1f} min)\n')

        obs_t1 = Time(t1 + 7*u.hour, scale='utc', location=kpno)
        if verbose:
            print(obs_t1.sidereal_time('mean'))
    
        moonsep = np.mean(c_altaz.separation(moon_altaz)).value
        fieldinfo = rf'{field}, Moon: {moonsep:.1f}$^\circ$'
        if k == 3:
            fieldinfo += '\n'
        if p is not None:
            thecolor = colors[(k//len(lstyles)) % len(colors)]
            width=2
            if highlight_fields is not None:
                if not field in highlight_fields:
                    thecolor='silver'
                    width = 1
                    
            p = ax.plot(local_time_grid.datetime, c_altaz.alt.degree,
                        ls=lstyles[k % len(lstyles)],
                        # color=p[0].get_color(),
                        color=thecolor,
                        lw=width,
                        label=fieldinfo)
        else:
            thecolor = colors[(k//len(lstyles)) % len(colors)]
            width=2
            if highlight_fields is not None:
                if not field in highlight_fields:
                    thecolor='silver'
                    width=1

            p = ax.plot(local_time_grid.datetime, c_altaz.alt.degree,
                        ls=lstyles[k % len(lstyles)],
                        color=thecolor,
                        lw=width,
                        label=fieldinfo)
    
    ax.set(xlabel='date/time [MST]',
           ylabel='altitude [deg]',
           xlim= (local_time_grid.datetime[0], local_time_grid.datetime[-1]),
           ylim=(0, 90),
           title=f'KPNO: night of {"-".join([f"{local_obs_date.ymdhms[i]:02d}" for i in np.arange(3)])}')
    
    ax.fill_betweenx((0,90), dusk12.datetime, dawn12.datetime, color='navy', alpha=0.1)
    
    ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left', borderaxespad=0., fontsize=10)
    
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    
    xfmt = mpl.dates.DateFormatter('%m-%d %H:%M')
    ax.xaxis.set_major_formatter(xfmt)
    
    xloc = mpl.dates.HourLocator()
    ax.xaxis.set_major_locator(xloc)
    
    ax2 = ax.secondary_yaxis('right', functions=(alt2am, am2alt))
    ax2.set(yticks=[1,1.25,1.5,2,3,5], ylabel='airmass')
    for am in [1.5,2,3]:
        ax.axhline(am2alt(am), zorder=-10, lw=1, ls='--', color='k', alpha=0.5)
    
    ax.grid(ls=':')
    fig.subplots_adjust(top=0.945, right=0.675, bottom=0.145, left=0.075)
    
    return fig

def plot_tiles_transit_kpno(night, filename, airmass_limit=None, highlight_tiles=None, verbose=False):
    """Plot transits of tiles from a tiles file.

    Parameters
    ----------
    night: str
        Observing date in format YYYY-MM-DD.
    filename: str
        Path to tiles file, in eCSV format.
    airmass_limit: None or float
        Compute total time of a field above some airmass.
    highight_tiles: list
        List of TILEIDs to highlight in the plot (other tiles will be grayed out).

    Returns
    -------
    fig: matplotlib.Figure
        Figure object for additional plotting/manipulation.
    """
    tiles = Table.read(filename)
    
    return plot_transit_kpno(night,
                             fields_name=tiles['TILEID'].value,
                             fields_ra=tiles['RA'].value,
                             fields_dec=tiles['DEC'].value,
                             airmass_limit=airmass_limit,
                             highlight_fields=highlight_tiles)

def main():
    """Plot transits for a tile eCSV file or field name(s) with RA(s) and Dec(s).

    Example usage with a tiles file:
        python planning.py -n 2026-01-30 -t tiles-082918-082930.ecsv -a 1.5

    Example usage with a list of fields:
        python planning.py -n 2026-01-15 --fields XMM-LSS COSMOS GAMA-15 --ra 35.7 150.1 215.7 --dec -4.75 2.182 -0.7

    Example highlighting one field out of several:
        python planning.py -n 2026-01-15 --fields XMM-LSS COSMOS GAMA-15 --ra 35.7 150.1 215.7 --dec -4.75 2.182 -0.7 --highlight XMM-LSS
    """
    from argparse import ArgumentParser

    p = ArgumentParser(description='Field and tile planning for DESI')
    p.add_argument('-n', '--night', type=str,
                   default=datetime.today().strftime('%Y-%m-%d'),
                   help='Night in format YYYY-MM-DD')
    p.add_argument('-t', '--tilefile', type=str,
                   default=None,
                   help='Tiles file in eCSV format')
    p.add_argument('-f', '--fields', type=str, nargs='*',
                   help='Field name(s)')
    p.add_argument('-r', '--ra', type=float, nargs='*',
                   help='Field RA(s)')
    p.add_argument('-d', '--dec', type=float, nargs='*',
                   help='Field Dec(s)')
    p.add_argument('-a', '--airmass-limit', type=float,
                   default=3.,
                   help='Airmass threshold for computing transits')
    p.add_argument('--highlight', type=str, nargs='*',
                   help='List of field/tile names to highlight in the plot')
    p.add_argument('-v', '--verbose', action='store_true',
                   help='Verbose output')
    args = p.parse_args()

    #- Manual check for compatible options (could also be done with a subcommand)
    if args.tilefile and (args.fields or args.ra or args.dec):
        raise SystemExit(f'{os.path.basename(__file__)}: Tile filename option incompatible a list of field names and RA,Dec')

    #- Manual check for compatible fields, RA, Dec
    if any(v is not None for v in (args.fields, args.ra, args.dec)):
        if None not in (args.fields, args.ra, args.dec):
            if len(args.fields) == len(args.ra) == len(args.dec):
                fig = plot_transit_kpno(args.night, args.fields, args.ra, args.dec, args.airmass_limit, args.highlight, args.verbose)
            else:
                raise SystemExit(f'{os.path.basename(__file__)}: Number of fields, RAs, and Decs must be the same')
        else:
            raise SystemExit(f'{os.path.basename(__file__)}: Field(s) must include name(s), RA(s), and Dec(s)')
    else:
        if args.tilefile is None:
            raise SystemExit(f'{os.path.basename(__file__)}: Specify a tile filename or a list of field names and RA,Dec')

        fig = plot_tiles_transit_kpno(args.night, args.tilefile,  args.airmass_limit, args.highlight, args.verbose)

    plt.show()

if __name__ == '__main__':
    main()
