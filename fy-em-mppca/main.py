from __future__ import print_function
import numpy as np
from scipy.linalg import inv, pinv
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
import itertools  # for permutations
from kss import mppca_init_KSS
from mppca import homppca_tipping
from data import load_hopkins155
from smppca import smppca_tipping_final
import sys
import optuna
from contextlib import contextmanager
import sys
from pathlib import Path
from tqdm import tqdm
np.random.seed(42)

functions = {
             "mppca": homppca_tipping,  
             "smppca": smppca_tipping_final}
func_name = sys.argv[1]
# noise = float(sys.argv[2])
# alpha = float(sys.argv[3])
type_noise = "default"
N = 156
data_names=None

@contextmanager
def log_stdout(filepath, mute_stdout=False):
    '''Context manager to write both to stdout and to a file'''

    class MultipleStreamsWriter:
        def __init__(self, streams):
            self.streams = streams

        def write(self, message):
            for stream in self.streams:
                stream.write(message)

        def flush(self):
            for stream in self.streams:
                stream.flush()

    save_stdout = sys.stdout
    log_file = open(filepath, 'w')
    if mute_stdout:
        sys.stdout = MultipleStreamsWriter([log_file])  # Write to file only
    else:
        sys.stdout = MultipleStreamsWriter([save_stdout, log_file])  # Write to both stdout and file
    try:
        yield
    finally:
        sys.stdout = save_stdout
        log_file.close()


def test_assignments(F_hat, mu_hat, Y_test, prop_hat, var_hat, epsilon=1e-6):
    """
    Compute the predicted assignments for test data using the trained model.
    """
    d, k = F_hat[0].shape
    n = Y_test.shape[1]
    J = len(F_hat)

    M_inv = []
    C_inv = []

    for j in range(J):
        M_j = var_hat[j] * np.eye(k) + F_hat[j].T @ F_hat[j]
        if not np.all(np.isfinite(M_j)):
            print(f"M_j contains NaNs or Infs at component {j+1}, adding regularization.")
            M_j += epsilon * np.eye(k)
        try:
            M_inv_j = inv(M_j)
        except np.linalg.LinAlgError:
            M_inv_j = pinv(M_j)
        M_inv.append(M_inv_j)

        C_inv_j = (1.0 / var_hat[j]) * (np.eye(d) - F_hat[j] @ M_inv_j @ F_hat[j].T)
        if not np.all(np.isfinite(C_inv_j)):
            print(f"C_inv_j contains NaNs or Infs at component {j+1}, adding regularization.")
            C_inv_j += epsilon * np.eye(d)
        C_inv.append(C_inv_j)

    log_likelihood = []
    for j in range(J):
        # Avoid zero or negative determinants
        sign, logdet = np.linalg.slogdet(C_inv[j])
        if sign <= 0 or not np.isfinite(logdet):
            print(f"log determinant not positive definite at component {j+1}, using epsilon.")
            logdet = np.log(epsilon)
        tmp = -0.5 * np.sum((Y_test - mu_hat[:, [j]]) * (C_inv[j] @ (Y_test - mu_hat[:, [j]])), axis=0)
        prop_j = max(prop_hat[j], epsilon)
        log_l = np.log(prop_j) + (-d / 2) * np.log(2 * np.pi) + 0.5 * logdet + tmp
        log_likelihood.append(log_l)

    log_likelihood = np.vstack(log_likelihood)
    assgn_hat_indices = np.argmax(log_likelihood, axis=0)
    # Map indices back to labels starting from 1
    assgn_hat = np.array([j + 1 for j in assgn_hat_indices])
    return assgn_hat

def get_test_error(Y_test, assgn_test, F0, mu0, prop0, var0, classes):

    #  Create mappings between labels and indices
    label_to_index = {label: idx for idx, label in enumerate(classes)}
    assgn_test_indices = np.array([label_to_index[label] for label in assgn_test])

    # Evaluate the KSS initialization on the test data
    min_error_init = float('inf')
    perms = list(itertools.permutations(range(len(classes))))  # Permutations over component indices
    for perm in perms:
        # Convert perm to a list or array for indexing
        perm_indices = np.array(perm)

        # Permute the initial parameters
        F_perm = [F0[i] for i in perm_indices]
        mu_perm = mu0[:, perm_indices]
        var_perm = var0[perm_indices]
        prop_perm = prop0[perm_indices]

        # Compute predicted assignments using the test data
        assgn_pred = test_assignments(F_perm, mu_perm, Y_test, prop_perm, var_perm, epsilon=1e-6)

        # Map predicted assignments to indices
        assgn_pred_indices = np.array([label_to_index[label] for label in assgn_pred])

        # Compute confusion matrix
        cm = confusion_matrix(assgn_test_indices, assgn_pred_indices, labels=range(len(classes)))
        row_ind, col_ind = linear_sum_assignment(-cm)
        mapping = {col_ind[i]: row_ind[i] for i in range(len(row_ind))}
        assgn_pred_mapped = np.array([mapping[idx] for idx in assgn_pred_indices])

        # Compute classification error
        error = np.mean(assgn_pred_mapped != assgn_test_indices)
        if error < min_error_init:
            min_error_init = error
    return min_error_init

def train(params, N=156, init="KSS", max_iter=500, data_names=None):
    dataset_path = 'Hopkins'  # Update this path accordingly
    train_split = 0.8  # Fraction of data to use for training

    # Load the dataset
    train_data_list, train_assign_list, dev_assign_list, dev_data_list, test_data_list, test_assign_list, metadata_list = load_hopkins155(dataset_path, 
                                                                                                                                          train_split=train_split, 
                                                                                                                                          N=N, 
                                                                                                                                          noise_value=params['noise'], 
                                                                                                                                          noise_type=params['type_noise'], 
                                                                                                                                          data_names=data_names)

    # Initialize accumulators for aggregation
    total_min_error_init = 0.0  # Accumulator for KSS Initialization errors
    total_min_error = 0.0       # Accumulator for PCCA errors
    num_datasets = 0            # Counter for the number of datasets processed

    # Iterate over each dataset in the Hopkins155 collection
    for idx, (Y_train, assgn_train, Y_dev, assign_dev, _, _, metadata) in tqdm(enumerate(zip(
        train_data_list, train_assign_list, dev_data_list, dev_assign_list, test_data_list, test_assign_list, metadata_list))):

        # Normalize data
        Y_mean = np.mean(Y_train, axis=1, keepdims=True)
        Y_std = np.std(Y_train, axis=1, keepdims=True) + 1e-8
        Y_train = (Y_train - Y_mean) / Y_std
        Y_dev = (Y_dev - Y_mean) / Y_std

        classes = np.unique(assgn_train)  # Labels starting from 1
        J = len(classes)
        k = 4  # metadata['k']  # Assuming k is set to 4; adjust as needed

        # Initialize model
        if init == "KSS":
            F0, _, mu0, prop0, var0 = mppca_init_KSS(Y_train, J, k, max_iter=max_iter)
        else:
            F0 = [np.random.randn(Y_train.shape[0], k) for _ in range(J)]
            mu0 = Y_train[:, np.random.choice(Y_train.shape[1], J, replace=False)]  # Random data points as means
            prop0 = np.random.dirichlet(np.ones(J), size=1).flatten()
            var0 = np.random.uniform(0.1, 1, size=J)

        # get error
        min_error_init = get_test_error(Y_dev, assign_dev, F0, mu0, prop0, var0, classes)

        # Train the model
        F_hat, mu_hat, var_hat, prop_hat = functions[params["func"]](Y_train, F0, mu0, prop0, var0, niter=max_iter, epsilon=1e-6,
                                                                T=params['temperature'], alpha=params['alpha_ent'], anneal=False)

        min_error = get_test_error(Y_dev, assign_dev, F_hat, mu_hat, prop_hat, var_hat, classes)

        # Accumulate the errors
        total_min_error_init += min_error_init
        total_min_error += min_error
        num_datasets += 1  # Increment the dataset counter

        # Print per-dataset results
        print(f"{metadata['name']} - {init} Init Classification Error: {min_error_init * 100:.2f}%")
        print(f"{metadata['name']} - PCCA Classification Error: {min_error * 100:.2f}%")

    # After processing all datasets, compute and print average errors
    if num_datasets > 0:
        average_min_error_init = total_min_error_init / num_datasets
        average_min_error = total_min_error / num_datasets

        print("=== Average Classification Errors Over All Datasets ===")
        print(f"{init} Initialization Average Error: {average_min_error_init * 100:.2f}%")
        print(f"Average Error: {average_min_error * 100:.2f}%")
    else:
        print("No datasets were processed.")
    
    return average_min_error 

def objective(trial: optuna.trial.Trial):
    
    params = {
        # 'noise': trial.suggest_loguniform('noise', 0.025, 0.25),
        # 'alpha_ent': trial.suggest_loguniform('alpha_ent', 1, 4),
        # 'noise': trial.suggest_categorical('noise', [0.001, 0.005, 0.025, 0.125]),
        # 'alpha_ent': trial.suggest_categorical('alpha_ent', [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]),
        'alpha_ent': trial.suggest_float('alpha_ent', 0.1, 1.0, step=0.1),
    }
    # params = {}
    params['func'] = func_name
    params["type_noise"] = type_noise
    params['anneal'] = False
    params['temperature'] = 1.0
    params['noise'] = 0.001
    # params['alpha_ent'] = 1.0
    best_val_loss = train(params)
    return best_val_loss

def main():
    # Path to the Hopkins155 dataset directory    
    exp_dir = Path("experiments/")
    tuning_log_dir = exp_dir / 'tuning_logs'
    tuning_log_dir.mkdir(parents=True, exist_ok=True)
    i = 1
    tuning_logs = tuning_log_dir / f'logs_{i}.txt'
    while tuning_logs.exists():
        i += 1
        tuning_logs = tuning_log_dir / f'logs_{i}.txt'

    with log_stdout(tuning_logs):
        study = optuna.create_study(study_name=func_name, direction="minimize",
                                    storage=f'sqlite:///{tuning_log_dir}/{func_name}_alpha_logp_0.1_1_0.1.db', load_if_exists=True)
        study.optimize(objective, n_trials=10, n_jobs=2, show_progress_bar=True)

        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
    


if __name__ == '__main__':
    # params = {}
    # params['func'] = func_name
    # params['noise'] = noise
    # params['temperature'] = 1
    # params['alpha_ent'] = alpha
    # params['type_noise'] = type_noise

    # with log_stdout(f"logs/{func_name}_t{params['temperature']}_n{params['noise']}_a{params['alpha_ent']}_{params['type_noise']}_{N}"):
    #     train(params, N=N, data_names=data_names)
    main()
