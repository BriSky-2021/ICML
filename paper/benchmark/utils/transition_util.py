import os
import numpy as np

def build_transition_matrices():
    """
    Load precomputed health state transition matrices.

    Returns:
        Dict mapping action (0-3) to corresponding state transition matrix.
    """
    transition_metrics_dir = 'paper/dataset/data/transition_metrics'
    combined_file_path = os.path.join(transition_metrics_dir, 'all_transition_matrices.npz')

    # 1. Prefer loading the combined npz
    if os.path.exists(combined_file_path):
        try:
            matrices = np.load(combined_file_path)
            transition_matrices = {
                0: matrices['no_action'],
                1: matrices['minor_repair'],
                2: matrices['major_repair'],
                3: matrices['replacement']
            }
            return transition_matrices
        except Exception as e:
            print(f"Failed to load npz transition matrices: {e}")
            print("Will try loading from individual npy files...")

    # 2. Load npy files one by one
    try:
        action_names = ['no_action', 'minor_repair', 'major_repair', 'replacement']
        transition_matrices = {}
        for a, action_name in enumerate(action_names):
            file_path = os.path.join(transition_metrics_dir, f'transition_matrix_{action_name}.npy')
            if os.path.exists(file_path):
                transition_matrices[a] = np.load(file_path)
        if len(transition_matrices) == 4:
            return transition_matrices
    except Exception as e:
        print(f"Failed to load transition matrices from npy files: {e}")

    # 3. Fallback on failure
    print("Failed to load precomputed health state transition matrices.")
    exit(0)
    