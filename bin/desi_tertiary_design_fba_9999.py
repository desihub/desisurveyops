from typing import List, Dict
import numpy as np
from pathlib import Path
from astropy.table import Table, vstack
from desisurveyops.fba_tertiary_design_io import (
    assert_environ_settings,
    assert_files,
    assert_tertiary_settings,
    create_targets_assign,
    create_tiles_table,
    creates_priority_table,
    finalize_target_table,
    get_fn,
    get_tile_centers_rosette,
    match_coord,
    plot_targets_assign,
    print_samples_overlap,
    read_yaml,
    subsample_targets_avail,
    TertiaryTileDesignBase
)
from desiutil.log import get_logger

logger = get_logger()


class TertiaryTileDesign(TertiaryTileDesignBase):
    """Tertiary Tile Design for XMM High-z (prognum)

    This class implements the specific tertiary tile design logic for the XMM High-z program.
    It inherits from TertiaryTileDesignBase and overrides necessary methods to perform the design.

    Attributes:
        yamlfp (str): The path to the yaml file containing the tertiary design configuration.
    """

    def __init__(self, yamlfp: str):
        self.yamlfp = yamlfp
        self.settings = read_yaml(yamlfp)["settings"]
        self.samples = read_yaml(yamlfp)["samples"]
        self.rootdir = Path(self.settings["targdir"])

    def create_tiles(self, outfp: str):
        field_ra, field_dec = 35.75, -4.75
        rundate = self.settings["rundate"]
        obsconds = self.settings["obsconds"]
        tileids = np.arange(
            self.settings["tileid_start"],
            self.settings["tileid_start"] + self.settings["ntile"],
            dtype=int,
        )
        tab = create_tiles_table(tileids, field_ra, field_dec, obsconds)
        tab.write(outfp)

    def create_priorities(self, outfp: str):
        tab = creates_priority_table(self.yamlfp)
        tab.pprint_all()
        for sample in np.unique(tab["TERTIARY_TARGET"]):
            sel = tab["TERTIARY_TARGET"] == sample
            logger.info("priorites for {}: {}".format(sample, tab["PRIORITY"][sel].tolist()))
        tab.write(outfp)


    def create_targets(self, outfp: str):
        
        def merge_target_catalogs(catdir: Path, samples: Dict[str, Dict]):
            
            sample_names = np.array(list(samples.keys()))
            priorities = np.array([samples[name]["PRIORITY_INIT"] for name in sample_names])
            ngoals = np.array([samples[name]["NGOAL"] for name in sample_names])

            order = priorities.argsort()[::-1]
            sample_names = sample_names[order]
            priorities = priorities[order]
            ngoals = ngoals[order]

            tables = []
            logger.info("reading targets from %s", self.rootdir)
            for sample, prio, ngoal in zip(sample_names, priorities, ngoals):
                sample_cfg = self.samples[sample]
                fn = self.rootdir / sample_cfg["FN"]
                logger.info("reading %s (%s)", sample, fn)
                cat = Table.read(fn)

                if "RA" not in cat.colnames and "ra" in cat.colnames:
                    cat["RA"] = cat["ra"]
                if "DEC" not in cat.colnames and "dec" in cat.colnames:
                    cat["DEC"] = cat["dec"]
                for col in ["RA", "DEC"]:
                    if col not in cat.colnames:
                        raise ValueError(f"Missing {col} column in {fn}")
                for col, default in [("PMRA", 0.0), ("PMDEC", 0.0), ("REF_EPOCH", 2015.5)]:
                    if col not in cat.colnames:
                        cat[col] = default

                merged = Table()
                for col in ["RA", "DEC", "PMRA", "PMDEC", "REF_EPOCH"]:
                    merged[col] = cat[col]
                merged["TERTIARY_TARGET"] = sample
                merged["CHECKER"] = sample_cfg["CHECKER"]
                merged["PRIORITY_INIT"] = int(prio)
                merged["NGOAL"] = int(ngoal)
                tables.append(merged)

            if len(tables) == 0:
                raise RuntimeError("No targets loaded from yaml samples")

            targets = vstack(tables)
            targets.sort("PRIORITY_INIT", reverse=True)
            logger.info("assembled %d targets across %d sample(s)", len(targets), len(tables))
            return targets

        targets = finalize_target_table(targets, self.yamlfp)
        targets.write(outfp)
        