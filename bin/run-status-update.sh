#!/bin/bash

source /global/cfs/cdirs/desi/software/desi_environment.sh main
export PYTHONPATH=/global/homes/b/brookluo/desihub/desisurveyops/py:$PYTHONPATH
/global/homes/b/brookluo/desihub/desisurveyops/bin/desi_tertiary_status --prognum $(cat /global/u1/b/brookluo/desihub/mydesi/desisurveyops/bin/tertiary-progress.txt)  --outdir /global/cfs/cdirs/desicollab/users/brookluo/tertiary-status --numproc 4
