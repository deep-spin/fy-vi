import os
import scipy.io
import numpy as np
from itertools import permutations
import random
random.seed(42)

def load_hopkins155(dataset_path, N=200, train_split=0.8, dev_split=0.1, noise_value=0.025, noise_type='default', data_names=None):
    """
    Load Hopkins 155 dataset and split into train and test sets.

    Args:
        dataset_path (str): Path to the Hopkins155 dataset directory.
        train_split (float): Fraction of data to use for training.

    Returns:
        train_data_list (list of np.ndarray): Training data arrays, each of shape (2 * F, p).
        train_assign_list (list of np.ndarray): Training cluster assignments.
        test_data_list (list of np.ndarray): Testing data arrays, each of shape (2 * F, p).
        test_assign_list (list of np.ndarray): Testing cluster assignments.
        metadata_list (list of dict): Metadata for each dataset entry, including classes and permutations.
    """
    train_data_list = []
    train_assign_list = []
    test_data_list = []
    test_assign_list = []
    metadata_list = []
    dev_data_list = []
    dev_assign_list = []

    # List all subdirectories in dataset_path, excluding 'README.txt'
    data_dirs = [
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d)) and d != "README.txt"
    ]

    random.shuffle(data_dirs)

    for data_dir in data_dirs[:N]:
        if data_names and data_dir not in data_names:
            continue
        mat_file = f"{data_dir}_truth.mat"
        mat_path = os.path.join(dataset_path, data_dir, mat_file)

        if not os.path.isfile(mat_path):
            print(f"Warning: {mat_file} not found in {data_dir}. Skipping this directory.")
            continue

        try:
            mat = scipy.io.loadmat(mat_path)
        except Exception as e:
            print(f"Error loading {mat_file}: {e}")
            continue

        # Extract cluster assignments
        if 's' not in mat:
            print(f"Warning: 's' key not found in {mat_file}. Skipping.")
            continue
        assgn = mat['s'].flatten()

        # Extract coordinates
        if 'y' not in mat:
            print(f"Warning: 'y' key not found in {mat_file}. Skipping.")
            continue
        y = mat['y'][0:2, :, :]  # Extract first two rows (x and y coordinates)

        # Extract number of points and frames
        if 'points' not in mat or 'frames' not in mat:
            print(f"Warning: 'points' or 'frames' key not found in {mat_file}. Skipping.")
            continue
        p = int(mat['points'].squeeze())
        F = int(mat['frames'].squeeze())
        n = p  # Assuming n is the number of points/trajectories

        # Verify the shape of y
        if y.shape != (2, p, F):
            print(f"Warning: Unexpected shape of 'y' in {mat_file}. Expected (2, {p}, {F}), got {y.shape}. Skipping.")
            continue

        # Permute dimensions to match Julia's permutedims([1,3,2])
        # Julia's [1,3,2] corresponds to Python's axes (0, 2, 1)
        y_perm = np.transpose(y, (0, 2, 1))  # Shape: (2, F, p)

        # Reshape to (2 * F, p) where each column represents a trajectory
        Y = y_perm.reshape(2 * F, p)  # Shape: (2 * F, p)

        # adds noise
        if noise_type=="constant":
            Y += noise_value * np.random.randn(*Y.shape) 
        elif noise_type =="sparse":
            outlier_indices = np.random.choice(Y.shape[1], size=int(noise_value * Y.shape[1]), replace=False)  # 5% outliers
            Y[:, outlier_indices] += 10 * np.random.randn(Y.shape[0], len(outlier_indices))  
        elif noise_type =="low-rank":
            U, S, Vt = np.linalg.svd(np.random.randn(Y.shape[0], Y.shape[1]), full_matrices=False)
            rank_reduction_factor = 0.8
            k = int(rank_reduction_factor * min(Y.shape))
            S_truncated = np.diag(S[:k])
            low_rank_noise = U[:, :k] @ S_truncated @ Vt[:k, :]
            Y += low_rank_noise
        elif noise_type == "non-gaussian":
            Y += np.random.laplace(0, noise_value, Y.shape)
        elif noise_type == "default":
            Y_enrgy = np.max(np.linalg.norm(Y, axis=0))  # Max energy of all trajectories
            Y += np.sqrt(noise_value * Y_enrgy)*np.random.normal(0.0, 1.0, (Y.shape))

        # Verify the reshape
        if Y.size != y_perm.size:
            print(f"Error: Reshape size mismatch for {mat_file}.")
            continue

        # Check for NaNs or Infs
        if np.isnan(Y).any() or np.isinf(Y).any():
            print(f"Warning: Data contains NaNs or Infs in {mat_file}. Skipping this dataset.")
            continue

        # Optional: Normalize data
        Y_mean = np.mean(Y, axis=1, keepdims=True)
        Y_std = np.std(Y, axis=1, keepdims=True) + 1e-8  # Add small epsilon to avoid division by zero
        Y_normalized = (Y - Y_mean) / Y_std

        # Replace Y with normalized data
        Y = Y_normalized

        # Determine train and test split indices
        n_train = int(round(n * train_split))
        # n_test = n - n_train
        n_dev = int(round(n * train_split * dev_split))

        # print(n_train, n_dev, n)

        # Split the data
        Y_train = Y[:, :-n_dev]
        assgn_train = assgn[:-n_dev]

        Y_dev= Y[:, n_dev:n_train]
        assgn_dev= assgn[n_dev:n_train]

        Y_test = Y[:, n_train:]
        assgn_test = assgn[n_train:]

        # Append to respective lists
        train_data_list.append(Y_train)
        train_assign_list.append(assgn_train)
        dev_data_list.append(Y_dev)
        dev_assign_list.append(assgn_dev)
        test_data_list.append(Y_test)
        test_assign_list.append(assgn_test)

        # Process metadata: classes, J, k, perms
        classes = np.unique(assgn)
        J = len(classes)  # Number of subspaces/motions

        # Generate all unique permutations of classes
        perms = list(permutations(classes))

        # Store metadata
        metadata = {
            'name':f"{data_dir}", 
            'classes': classes,
            'J': J,
            'perms': perms
        }
        metadata_list.append(metadata)

    return train_data_list, train_assign_list, dev_assign_list, dev_data_list, test_data_list, test_assign_list, metadata_list