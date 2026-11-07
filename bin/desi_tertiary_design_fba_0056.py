from pathlib import Path
import numpy as np
from astropy.table import Table
from desisurveyops.fba_tertiary_design_io import (
    finalize_target_table,
    read_yaml,
    TertiaryTileDesignBase,
)
from desiutil.log import get_logger

logger = get_logger()


class TertiaryTileDesign(TertiaryTileDesignBase):
    """Tertiary tile design for GW cosmology (prognum 0056).

    Tiles and priorities are already written in targdir. This design only
    rebuilds the targets file: it reads the existing catalog, drops TARGETID,
    and lets finalize_target_table assign encode_targetid values for this
    prognum.
    """

    def __init__(self, yamlfp: str):
        self.yamlfp = yamlfp
        self.settings = read_yaml(yamlfp)["settings"]
        self.samples = read_yaml(yamlfp)["samples"]
        self.rootdir = Path(self.settings["targdir"])

    def create_tiles(self, outfp: str):
        logger.info("0056 tiles already defined; leaving %s unchanged", outfp)

    def create_priorities(self, outfp: str):
        logger.info("0056 priorities already defined; leaving %s unchanged", outfp)

    def create_targets(self, outfp: str):
        fns = {cfg["FN"] for cfg in self.samples.values()}
        if len(fns) != 1:
            raise ValueError(
                "0056 expects every sample to point at the same input catalog"
            )
        fn = self.rootdir / fns.pop()
        logger.info("reading targets from %s", fn)
        targets = Table.read(fn)
        for col in targets.colnames:
            if targets[col].dtype.kind == "S":
                targets[col] = targets[col].astype(str)

        catalog_samples = set(np.unique(targets["TERTIARY_TARGET"]))
        missing = catalog_samples - set(self.samples)
        if missing:
            raise ValueError(
                "catalog TERTIARY_TARGET values missing from yaml: {}".format(
                    sorted(missing)
                )
            )

        if "TARGETID" in targets.colnames:
            logger.info("dropping %d input TARGETID values", len(targets))
            targets.remove_column("TARGETID")
        if "SUBPRIORITY" in targets.colnames:
            targets.remove_column("SUBPRIORITY")

        targets = finalize_target_table(targets, self.yamlfp)
        targets.meta["PROGNUM"] = self.settings["prognum"]
        logger.info(
            "assigned %d TARGETID values for prognum %s",
            len(targets),
            self.settings["prognum"],
        )
        targets.write(outfp, overwrite=True)
