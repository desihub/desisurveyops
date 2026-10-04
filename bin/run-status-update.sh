#!/bin/bash

source /global/cfs/cdirs/desi/software/desi_environment.sh main
rootdir=/global/homes/b/brookluo/desihub/mydesi/desisurveyops
export PYTHONPATH=$rootdir/py:$PYTHONPATH
$rootdir/bin/desi_tertiary_status --prognum $(cat $rootdir/bin/tertiary-progress.txt)  --outdir /global/cfs/cdirs/desicollab/users/brookluo/tertiary-status --numproc 4
