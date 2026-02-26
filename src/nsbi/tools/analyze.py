#!/usr/bin/env python3
import argparse
import json
import os
import pathlib

import numpy as np
import pandas as pd

from nsbi.physics.analysis import wwlvlv, zz2l2v, zz4l
from nsbi.physics.simulation import mcfm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge MCFM event CSVs from multiple processes.")
    parser.add_argument("samples", type=str, help="List of process names to merge")

    parser.add_argument(
        "--data-dir",
        required=False,
        default="data/",
        help="Physics normalized & analyzed events output",
    )
    parser.add_argument(
        "--xsec-json", required=False, default="xsec.json", help="List of process names to merge"
    )
    parser.add_argument(
        "--events-csv",
        required=False,
        default="events_*.csv",
        help="List of process names to merge",
    )
    parser.add_argument(
        "--analyzed-csv",
        required=False,
        default="analyzed.csv",
        help="Physics normalized & analyzed events output",
    )

    args = parser.parse_args()

    with open(args.samples) as f:
        samples = json.load(f)

    print("----------" * 8)

    for analysis, processes in samples.items():
        if analysis == "zz4l":
            analyze = zz4l.analyze
        elif analysis == "zz2l2v":
            analyze = zz2l2v.analyze
        elif analysis == "wwlvlv":
            analyze = wwlvlv.analyze

        with open(os.path.join(args.data_dir, analysis, args.xsec_json)) as f:
            xsec = json.load(f)

        for process in processes:
            print("Analysis :", os.path.join(analysis, process))

            events = mcfm.from_csv(
                cross_section=np.prod(xsec[process]),
                file_path=os.path.join(args.data_dir, analysis, process, args.events_csv),
                ignore_negative_weights=False,
            )

            events_analyzed = os.path.join(args.data_dir, analysis, args.analyzed_csv)

            events_analyzed = analyze(events)

            # Combine dataframes, ignoring index
            df_analyzed = pd.concat(
                [
                    events_analyzed.kinematics.reset_index(drop=True),
                    events_analyzed.components.reset_index(drop=True),
                    events_analyzed.weights.reset_index(drop=True),
                ],
                axis=1,
            )

            output_dir = os.path.join(args.data_dir, analysis, process)
            pathlib.Path(output_dir).mkdir(exist_ok=True, parents=True)
            df_analyzed.to_csv(os.path.join(output_dir, args.analyzed_csv), index=False)

            print("----------" * 8)
