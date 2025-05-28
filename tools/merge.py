import os
import json
import argparse
import numpy as np
import pandas as pd

from physics.simulation import mcfm


def main(data_dir, processes, events_csv, xsec_json, output):
    # Load cross sections
    with open(os.path.join(data_dir, xsec_json), 'r') as f:
        xsec = json.load(f)

    # Get cross sections for specified processes
    cross_sections = [xsec[proc] for proc in processes]

    # Load and stack events
    events_individual = [
        mcfm.from_csv(
            cross_section=np.prod(xsec[proc]),
            file_path=os.path.join(data_dir, proc, events_csv),
            n_rows = 10_000
        )
        for proc in processes
    ]
    events_merged = mcfm.stack(*events_individual)

    # Combine dataframes, ignoring index
    df_merged = pd.concat(
        [
            events_merged.kinematics.reset_index(drop=True),
            events_merged.components.reset_index(drop=True),
            events_merged.weights.reset_index(drop=True)
        ],
        axis=1
    )

    # Ensure output directory exists
    output_dir = os.path.join(data_dir, output)
    os.makedirs(output_dir, exist_ok=True)

    # Write to CSV
    df_merged.to_csv(os.path.join(output_dir, 'events.csv'), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge MCFM event CSVs from multiple processes.")
    parser.add_argument('--data-dir', required=True, help='Directory containing process subfolders and xsec.json')
    parser.add_argument('--events-csv', required=False, default='events_*.csv', help='List of process names to merge')
    parser.add_argument('--xsec-json', required=False, default='xsec.json', help='List of process names to merge')
    parser.add_argument('--processes', required=True, nargs='+', help='List of process names to merge')
    parser.add_argument('--output', required=True, help='Name of the output merged process')

    args = parser.parse_args()
    main(data_dir=args.data_dir, processes=args.processes, events_csv=args.events_csv, xsec_json=args.xsec_json, output=args.output)
