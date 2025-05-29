import os
import re
import pickle
import numpy as np

from models.taylr import TAYLR

def load_results(output_dir, coeffs):
    """
    Load the results from the TAYLR runs.
    Args:
        output_dir (str): Directory where the TAYLR runs are stored.
        coeffs (list): List of (n, m, l) triplets used in the TAYLR runs.
    Returns:
        events_test (list): List of test events.
        scaler_x (object): Scaler for the input features.
        models (3D list): Nested list [n][m][l] of models or None.
        scalers_y (3D list): Nested list [n][m][l] of scalers or None.
    """
    def folder_name(n, m, l):
        return f'taylr_c6_{n}_ct_{m}_cg_{l}'

    # Determine shape of the 3D arrays
    max_n = max(c[0] for c in coeffs) + 1
    max_m = max(c[1] for c in coeffs) + 1
    max_l = max(c[2] for c in coeffs) + 1

    # Initialize 3D arrays filled with None
    models = [[[None for _ in range(max_l)] for _ in range(max_m)] for _ in range(max_n)]
    scalers_y = [[[None for _ in range(max_l)] for _ in range(max_m)] for _ in range(max_n)]

    # Load common files from the first run
    first_dir = os.path.join(output_dir, folder_name(*coeffs[0]))
    with open(os.path.join(first_dir, 'events_test.pkl'), 'rb') as f:
        events_test = pickle.load(f)
    with open(os.path.join(first_dir, 'scaler_X.pkl'), 'rb') as f:
        scaler_x = pickle.load(f)

    # pattern for matching checkpoint filenames, e.g., 'epoch=3-step=100.ckpt'
    ckpt_pattern = re.compile(r"epoch=(\d+).*\.ckpt")

    for n, m, l in coeffs:
        taylr_dir = os.path.join(output_dir, folder_name(n, m, l))

        # Load scaler_y
        with open(os.path.join(taylr_dir, 'scaler_y.pkl'), 'rb') as f:
            scaler_y = pickle.load(f)

        logs_dir = os.path.join(taylr_dir, 'lightning_logs')

        versions = [d for d in os.listdir(logs_dir) if d.startswith('version_') and d.split('_')[-1].isdigit()]
        if not versions:
            raise FileNotFoundError(f"No version directories found in {logs_dir}")
        latest_version = max(versions, key=lambda v: int(v.split('_')[-1]))

        checkpoint_dir = os.path.join(logs_dir, latest_version, 'checkpoints')
        all_ckpts = [f for f in os.listdir(checkpoint_dir) if ckpt_pattern.match(f)]
        if not all_ckpts:
            raise FileNotFoundError(f"No matching checkpoint files found in {checkpoint_dir}")
        all_ckpts.sort(key=lambda f: int(ckpt_pattern.match(f).group(1)))
        latest_ckpt = os.path.join(checkpoint_dir, all_ckpts[-1])

        # Load model
        model = TAYLR.load_from_checkpoint(checkpoint_path=latest_ckpt)

        # Assign to proper location
        models[n][m][l] = model
        scalers_y[n][m][l] = scaler_y

    return events_test, scaler_x, models, scalers_y