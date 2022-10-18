# %%
import numpy as np
from pathlib import Path
from joblib import Memory

import mne
from alphacsc import GreedyCDL, BatchCDL

from moabb.datasets import BNCI2014001

from utils_csc import get_data
from utils_plot import plot_atoms

# Define the cache directory for joblib.Memory
CACHEDIR = Path('./__cache__')
if not CACHEDIR.exists():
    CACHEDIR.mkdir(parents=True)

memory = Memory(CACHEDIR, verbose=0)

# download full dataset
dataset = BNCI2014001()
# dataset.download()

SFREQ = 250.0
cdl_params = {
    # Shape of the dictionary
    'n_atoms': 40,
    'n_times_atom': int(round(SFREQ * 1.0)),
    # Request a rank1 dictionary with unit norm temporal and spatial maps
    'rank1': True,
    'uv_constraint': 'separate',
    # apply a temporal window reparametrization
    'window': True,
    # at the end, refit the activations with fixed support
    # and no reg to unbias
    'unbiased_z_hat': True,
    # Initialize the dictionary with random chunk from the data
    'D_init': 'chunk',
    # rescale the regularization parameter to be a percentage of lambda_max
    'lmbd_max': "scaled",  # original value: "scaled"
    'reg': 0.1,
    # Number of iteration for the alternate minimization and cvg threshold
    'n_iter': 100,  # original value: 100
    'eps': 1e-4,  # original value: 1e-4
    # solver for the z-step
    'solver_z': "lgcd",
    'solver_z_kwargs': {'tol': 1e-3,  # stopping criteria
                        'max_iter': 10_000},
    # solver for the d-step
    'solver_d': 'alternate_adaptive',
    'solver_d_kwargs': {'max_iter': 300},  # original value: 300
    # sort atoms by explained variances
    'sort_atoms': True,
    # Technical parameters
    'verbose': 1,
    'random_state': 0,
    'n_jobs': 5
}


@memory.cache()
def run_cdl(subjects, cdl_params, use_greedy=True):
    """

    """
    X = get_data(dataset=dataset, subjects=subjects, fmin=2, fmax=None)

    # apply CDL
    if use_greedy:
        cdl = GreedyCDL(**cdl_params)
    else:
        cdl = BatchCDL(**cdl_params)
    cdl.fit(X)

    z_hat = cdl.transform(X)
    u_hat_, v_hat_ = cdl.u_hat_, cdl.v_hat_

    # get one mne.Info instance
    if subjects is None:
        subjects = dataset.subject_list
    raw = dataset.get_date(subjects[0])[subjects[0]]['session_T']['run_0']

    plot_atoms(
        cdl, info=raw.info, plotted_atoms='all', sfreq=SFREQ,
        fig_name=f"atoms_{subjects}_{'greedy' if use_greedy else 'batch'}")

    return u_hat_, v_hat_, z_hat


u_hat_, v_hat_, z_hat = run_cdl(
    subjects=[1], cdl_params=cdl_params, use_greedy=True)

# %%
