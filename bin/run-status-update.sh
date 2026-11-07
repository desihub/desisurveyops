#!/bin/bash

source /global/cfs/cdirs/desi/software/desi_environment.sh 25.3
rootdir=/global/homes/b/brookluo/desihub/mydesi/desisurveyops
export PYTHONPATH=$rootdir/py:$PYTHONPATH
set -x
$rootdir/bin/desi_tertiary_status --prognums $(cat $rootdir/bin/tertiary-progress.txt)  --outdir /global/cfs/cdirs/desicollab/users/brookluo/tertiary-status --numproc 4
set +x
