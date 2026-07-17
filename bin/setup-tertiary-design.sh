#!/bin/bash
source /global/cfs/cdirs/desi/software/desi_environment.sh main
module load fiberassign/5.8.0
if [[ $1 == "local" ]]; then
    export PYTHONPATH=/global/u1/b/brookluo/desihub/mydesi/desisurveyops/py:$PYTHONPATH
    export PATH=/global/u1/b/brookluo/desihub/mydesi/desisurveyops/bin:$PATH
elif [[ $1 == "desiproc" ]]; then
    export PYTHONPATH=/pscratch/sd/b/brookluo/mydesi/desisurveyops/py:$PYTHONPATH
    export PATH=/pscratch/sd/b/brookluo/mydesi/desisurveyops/bin:$PATH
elif [[ $1 == "rsync" ]]; then
    rsync -rv --exclude "__pycache__" --exclude ".git" "/global/homes/b/brookluo/desihub/mydesi/desisurveyops/" /pscratch/sd/b/brookluo/mydesi/desisurveyops/
else
    echo "Usage: source setup-tertiary-design.sh [local|desiproc|rsync]"
fi
export DESIMODEL=/global/common/software/desi/perlmutter/desiconda/current/code/desimodel/main
export SKYHEALPIXS_DIR=/global/cfs/cdirs/desi/target/skyhealpixs/v1
export SKYBRICKS_DIR=/global/cfs/cdirs/desi/target/skybricks/v3
