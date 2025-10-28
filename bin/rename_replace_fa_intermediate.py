#!/usr/bin/env python

import os
import fitsio
import numpy as np
from desiutil.log import get_logger
from argparse import ArgumentParser

log = get_logger()

intermediate_cases = ["tiles", "sky", "targ", "scnd", "gfa"]
default_fbadir = os.path.join(os.getenv("DESI_ROOT"), "survey", "fiberassign", "main")


def parse():
    parser = ArgumentParser(
        description="This script renames existing fiberassign intermediary files for a set of specified tiles, and replaces these files with new versions from a specififed location."
    )
    parser.add_argument(
        "--cases",
        help="csv list of cases of files to check (default={})".format(
            ",".join(intermediate_cases)
        ),
        type=str,
        default=",".join(intermediate_cases),
    )
    parser.add_argument(
        "--fbadir",
        help="folder with the fiberassign intermediate products (default={})".format(
            default_fbadir
        ),
        type=str,
        default=default_fbadir,
    )
    parser.add_argument(
        "--rpldir",
        help = "folder with the replacement fiberassign intermediate products",
        type = str,
        default = "/global/cfs/cdirs/desi/users/lucasnap/validate-last-night-reruns-iss376/fiberassign"
    )
    parser.add_argument(
        "--runcommands",
        help="pass to actually run the rename/repleace commands rather than just output them",
        action="store_true"
    )
    parser.add_argument(
        "--intiles",
        help="Location of csv file containing list of all tiles for which to rename/replace intermediary files",
        type = str,
        default = "/global/cfs/cdirs/desi/users/lucasnap/validate-last-night-reruns-iss376/replace_tiles.csv"
    )
    parser.add_argument(
        "--rplsuffix",
        help="Suffix to add to original files before moving new files. (default is -rpl)",
        type=str,
        default="-rpl"
    )

    args = parser.parse_args()
    assert np.all(np.in1d(args.cases.split(","), intermediate_cases))
    for kwargs in args._get_kwargs():
        log.info("{}\t{}".format(kwargs[0], kwargs[1]))
    return args

def main():

    args = parse()

    tiles = np.genfromtxt(args.intiles,delimiter=',',dtype=int)

    #list to hold strings of rename/replace commands
    rpl_commands = []
    
    #first lets check that all of the necessary files exist in both fbadir and rpldir locations
    #and while doing so we can save the commands we'll need later to move/rename the files
    for tile in tiles:
        ts = str(tile).zfill(6)
        for case in args.cases.split(","):
            fba_file = os.path.join(args.fbadir,ts[:3],"{}-{}.fits".format(ts,case))
            rpl_file = os.path.join(args.rpldir,ts[:3],"{}-{}.fits".format(ts,case))
            
            if not os.path.exists(fba_file):
                log.info("Original File:{} does not exist. Please investigate before continuing.".format(fba_file))
                raise FileNotFoundError()
            if not os.path.exists(rpl_file):
                log.info("Replacement File:{} does not exist. Please investigate before continuing.".format(rpl_file))
                raise FileNotFoundError()

            rpl_commands.append("mv {} {}".format(fba_file,fba_file.replace(".fits","-{}.fits".format(args.rplsuffix))))
            rpl_commands.append("mv {} {}".format(rpl_file,fba_file))
                
    log.info("All original/replacement files exist. Outputting rename/replace commands.")

    #print and optionally execute the rename/replace commands
    for command in rpl_commands:
        print(command)
        if args.runcommands:
            os.system(command)

            
if __name__ == "__main__":
    main()

  