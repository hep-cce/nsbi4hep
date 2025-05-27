import os, re, pickle

from ..taylr import TAYLR

def load_results(output_dir, coeffs):
    """
    Load the results from the taylr runs.
    Args:
        output_dir (str): Directory where the taylr runs are stored.
        coeffs (list): List of [c6, ct, cg] triplets used in the taylr runs.
    Returns:
        events_test (list): List of test events.
        scaler_x (object): Scaler for the input features.
        models (list): List of loaded TAYLR models.
        scalers_y (list): List of scalers for the output features.
    """
    def folder_name(c6, ct, cg):
        return f'taylr_c6_{c6}_ct_{ct}_cg_{cg}'

    # testing events and feature scaler are the same for all runs
    first_dir = os.path.join(output_dir, folder_name(*coeffs[0]))
    with open(os.path.join(first_dir, 'events_test.pkl'), 'rb') as f:
        events_test = pickle.load(f)
    with open(os.path.join(first_dir, 'scaler_X.pkl'), 'rb') as f:
        scaler_x = pickle.load(f)

    models = []
    scalers_y = []

    for c in coeffs:
        taylr_dir = os.path.join(output_dir, folder_name(*c))

        # Load scaler_y
        with open(os.path.join(taylr_dir, 'scaler_y.pkl'), 'rb') as f:
            scaler_y = pickle.load(f)
        scalers_y.append(scaler_y)

        # Use earliest version directory
        logs_dir = os.path.join(taylr_dir, 'lightning_logs')
        versions = [d for d in os.listdir(logs_dir) if d.startswith('version_') and d.split('_')[-1].isdigit()]
        if not versions:
            raise FileNotFoundError(f"No version directories found in {logs_dir}")
        earliest_version = min(versions, key=lambda v: int(v.split('_')[-1]))

        # Use earliest checkpoint
        checkpoint_dir = os.path.join(logs_dir, earliest_version, 'checkpoints')
        all_ckpts = [f for f in os.listdir(checkpoint_dir) if ckpt_pattern.match(f)]
        if not all_ckpts:
            raise FileNotFoundError(f"No matching checkpoint files found in {checkpoint_dir}")
        all_ckpts.sort(key=lambda f: int(ckpt_pattern.match(f).group(1)))  # ascending
        earliest_ckpt = os.path.join(checkpoint_dir, all_ckpts[0])

        # Load model
        model = TAYLR.load_from_checkpoint(checkpoint_path=earliest_ckpt)
        models.append(model)

    return events_test, scaler_x, models, scalers_y
