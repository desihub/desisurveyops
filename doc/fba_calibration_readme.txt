# pick a prognum and a TILEID
PROGNUM=5
TILEID=83000  # the tileid to design; for calibration, we design them one-by-one
CHECKER=AR    # checker name for the ToO*ecsv CHECKER column; your initials


# ====================================================
# environment + various settings

PROGNUMPAD=`echo $PROGNUM | awk '{printf("%04d\n", $1)}'`
TILEIDPAD=`echo $TILEID | awk '{printf("%06d\n", $1)}'`

# load the environment and set variables
source /global/cfs/cdirs/desi/software/desi_environment.sh 22.2
module load desisurveyops
module swap fiberassign/5.7.0
export DESIMODEL=/global/common/software/desi/$NERSC_HOST/desiconda/current/code/desimodel/main
export SKYHEALPIXS_DIR=$DESI_ROOT/target/skyhealpixs/v1

# pick the latest rundate
#   (as we design the tiles one-by-one, months/years apart)
FPSTATEFN=`ls $DESIMODEL/data/focalplane/desi-state_*ecsv | tail -n 1`
RUNDATE=`tail -n 1 $FPSTATEFN | awk '{print $1}'`
echo "RUNDATE = $RUNDATE"

# desitarget catalogs version ! for standards only !
# calibration tiles are either BRIGHT or DARK, so use 1.1.1
STD_DTVER=1.1.1

# go to the official folder for fiberassign design
# (if designing the first tile for PROGNUM, create this folder beforehand)
# (if you want to run a test, just point TERTIARY_DESIGN_DIR to your test folder;
#   note that you will probably need to copy there some files)
TERTIARY_DESIGN_DIR=$DESI_ROOT/survey/fiberassign/special/tertiary/$PROGNUMPAD
# ====================================================


# ====================================================
# if designing the first tile for PROGNUM:

# first generate tiles/priorities/targets
desi_fba_calibration_inputs --prognum $PROGNUM --targdir $TERTIARY_DESIGN_DIR --steps tiles,priorities,targets --checker $CHECKER

# verify that the three files have been created
ls -lrth $TERTIARY_DESIGN_DIR/tertiary-{tiles,priorities,targets}-$PROGNUMPAD.*
# ====================================================



# ====================================================
# then create for TILEID:
# - the ToO-$PROGNUM-$TILEID.{ecsv,log} files
# - the fiberassign-$TILEIDPAD.{fits.gz,png,log} files
# - if the tile has already been designed+svn-committed, add --forcetileid to the desi_fba_tertiary_wrapper call
#   ! use with caution !
# - if you want to run a test in a test TERTIARY_DESIGN_DIR folder, add --custom_too_development
#   to the desi_fba_tertiary_wrapper call)
# - because of being too cautious in fiberassign/5.7.0, ToO files from outside $DESI_SURVEYOPS/
#       are not accepted; hence we need to run with --custom_too_development anyway...
FADIR=$DESI_TARGET/fiberassign/tiles/trunk # folder to parse for previous tileids
desi_fba_tertiary_wrapper --prognum $PROGNUM --targdir $TERTIARY_DESIGN_DIR --rundate $RUNDATE --std_dtver $STD_DTVER --add_main_too --only_tileid $TILEID --custom_too_development

# verify that the files have been created
ls -tlrh $TERTIARY_DESIGN_DIR/ToO-$PROGNUM-$TILEID.{ecsv,log}
ls -tlrh $TERTIARY_DESIGN_DIR/${TILEIDPAD:0:3}/fiberassign-$TILEIDPAD.{fits.gz,png,log}
# ====================================================



# ====================================================
# now svn-commit
#
# first update your local checkouts
#
# set path to your up-to-date checkout of surveyops
# note that you just need to checkout the tertiary folder (the mtl folder is large!)
# export MYSURVEYOPS=your_surveyops_checkout
# if you do not have a checkout, run this from $MYSURVEYOPS:
#   svn --username your_wiki_username co https://desi.lbl.gov/trac/browser/data/surveyops/trunk/tertiary
cd $MYSURVEYOPS/tertiary
svn up

# if designing the first tile for PROGNUM:
cd $MYSURVEYOPS
mkdir tertiary/$PROGNUMPAD                                                                                                                                                  
cp -p $TERTIARY_DESIGN_DIR/tertiary-{tiles,priorities,targets}-$PROGNUMPAD.* tertiary/$PROGNUMPAD/
svn add tertiary/$PROGNUMPAD
svn commit tertiary/$PROGNUMPAD -m "Adding tertiary$PROGNUM tiles, priorities, targets files"

# svn-commit the TILEID ToO files
cd $MYSURVEYOPS
cp -p $TERTIARY_DESIGN_DIR/ToO-$PROGNUM-$TILEID.{ecsv,log} tertiary/$PROGNUMPAD/
svn add tertiary/$PROGNUMPAD/ToO-$PROGNUMPAD-$TILEIDPAD.*
svn commit tertiary/$PROGNUMPAD -m "Adding ToO-$PROGNUMPAD-$TILEIDPAD ecsv and log"

# set path to your up-to-date-checkout of tiles
# note that you just need to checkout the 083/ folder (the rest is large!)
# export MYTILES=your_tiles_checkout
# if you do not have a checkout, run this from $MYTILES:
#   svn --username your_wiki_username co https://desi.lbl.gov/trac/browser/data/tiles/trunk/083
cd $MYTILES/${TILEIDPAD:0:3}
svn up
cp -p $TERTIARY_DESIGN_DIR/${TILEIDPAD:0:3}/fiberassign-$TILEIDPAD.{fits.gz,png,log} .
svn add fiberassign-$TILEIDPAD.{fits.gz,png,log}
svn commit fiberassign-$TILEIDPAD.{fits.gz,png,log} -m "add calibration tile ($TILEID from tertiary$PROGNUM)"
# ====================================================

