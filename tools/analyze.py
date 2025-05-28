import os
import json
import argparse
import numpy as np
import pandas as pd

from physics.simulation import mcfm

from physics.analysis import zz4l, zz2l2v, wwlvlv

def main(process_name, analysis_name, xsec_json, events_csv, output):

    if analysis == 'zz4l':
        analyze = zz4l.analyze
    elif analysis == 'zz2l2v':
        analyze = zz2l2v.analyze
    elif analysis == 'wwlvlv':
        analyze = wwlvlv.analyze

    # Load cross sections
    with open(xsec_json, 'r') as f:
        xsec = json.load(f)

    # Load and stack events
    events = mcfm.from_csv(
        cross_section = np.prod(xsec[process_name]),
        file_path=events_csv
    )

    events_analyzed = analyze(events)

    # Combine dataframes, ignoring index
    df_analyzed = pd.concat(
        [
            events_analyzed.kinematics.reset_index(drop=True),
            events_analyzed.components.reset_index(drop=True),
            events_analyzed.weights.reset_index(drop=True)
        ]
        axis=1
    )

    # Ensure output directory exists
    output_dir = os.path.join(data_dir, output)
    os.makedirs(output_dir, exist_ok=True)

    # Write to CSV
    df_analyzed.to_csv(os.path.join(output_dir, 'events.csv'), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge MCFM event CSVs from multiple processes.")
    parser.add_argument('process', required=True, type=str, help='List of process names to merge')
    parser.add_argument('analysis', required=True, type=str, help='List of process names to merge')
    parser.add_argument('--xsec', required=False, default='xsec.json', help='List of process names to merge')
    parser.add_argument('--events', required=False, default='events_*.csv', help='List of process names to merge')
    parser.add_argument('--output', required=False, default='events_analyzed.csv', help='Physics normalized & analyzed events output')

    args = parser.parse_args()
    main(analysis, process, xsec, events, output)
