various notes about desi_survey_status:

- ffmpeg:
    - as of Jun. 2025, the code crashes with /usr/bin/ffmpeg
    - so the user has to add to its PATH the path to a "working" version
    - example of a "working" version: $DESI_ROOT/users/raichoor/main-status/ffmpeg-git-20220910-amd64-static
- if running for the first time:
    - pick a folder to work in (OUTDIR)
    - launch on a node the script:
         desi_main_status --outdir $OUTDIR --numproc 256
      as of Jun. 2025, this first run takes 3 hours, as all the main survey tiles have to be "ingested"
      to have an idea, he approximate timing of each step is:
        skymap      35min
        qso         75min
        obsconds    10min
        fpstate     5min
        zhist       5min
        nspec       20min
        skyseq      1min
        spacewatch  35min
        html        1min
    - as of Jun. 2025, the total folder size is ~50G:
        510M    ./zhist
        9.4G    ./skymap
        12M     ./tiles
        2.2M    ./qso
        16G     ./nspec
        340M    ./fpstate
        9.1G    ./obsconds
        28M     ./exposures
        935M    ./skyseq
        14G     ./spacewatch

- in "usual" mode:
    - simply run: desi_main_status --outdir $OUTDIR --numproc 256
    - a "usual" run for a night takes like 20-30 minutes

- tiles file:
    - the "history" of the tiles files is stored in desisurveyops/data/tiles-YYYYMMDD-rev?????.ecsv files
    - this need to be updated in the future (e.g. for BRIGHT1B)
    - related function to also edit: desisurveyops.status_utils.get_history_tiles_infos()

- sky goal maps:
    - to prevent the need to update a fits file each time tiles-main.ecsv is edited,
    I ve removed those files, and now compute these maps "on-the-fly"; it is pretty fast
    (like 2mins for all programs), so it is ok

- nspec:
    - zmtl files are split per year
    - nspec.ecsv has to be a single file, to properly handle duplicates
    - compute all of them takes ~10min
    - numbers are computed from loa if before 20240409, from daily otherwise

- gfa:
    - I have removed any dependency on the gfa file; so this code is now blind to it,
    and will not catch any missing gfa file

- as of Jun. 1st, 2025, the files that live in some users area are:
    desi nominal 14k deg2 footprint: $DESI_ROOT/users/raichoor/desi-14k-footprint/desi-14k-footprint-dark.ecsv
    ephemerids (for moon): $DESI_ROOT/users/ameisner/GFA/gfa_reduce_etc/gfa_ephemeris.fits
  we probably want to move those to more official places (e.g. $DESI_ROOT/survey/)

