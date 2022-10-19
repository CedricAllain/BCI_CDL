# %%
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm

from sklearn.metrics import accuracy_score

from moabb.datasets import BNCI2014001

dataset = BNCI2014001()


def classify(z_train, z_test, labels_train, normalize=True):
    """

    """

    n_trials, n_atoms, n_times = z_train.shape

    if normalize:
        norm_train = np.linalg.norm(z_train, axis=2, ord=0).reshape(
            n_trials, n_atoms, 1)
        z_train /= norm_train

        norm_test = np.linalg.norm(z_test, axis=2, ord=0).reshape(
            n_trials, n_atoms, 1)
        z_test /= norm_test

    z_classes = dict()
    for label in np.unique(labels_train):
        idx_label = np.where(labels_train == label)[0]
        z_classes[label] = z_train[idx_label]

    labels_test = []
    for this_z in tqdm(z_test):
        corr_z = 0
        lbl = None
        for label, z_class in z_classes.items():
            corr_lbl = np.max([np.nansum(np.dot(this_z, z.T))
                              for z in z_class])
            if corr_lbl > corr_z:
                lbl = label
                corr_z = corr_lbl

        labels_test.append(lbl)

    return labels_test


def evaluate(dataset, true_label_test, labels_test):

    true_label_test = [dataset.event_id[lbl] for lbl in true_label_test]
    labels_test = [dataset.event_id[lbl] for lbl in labels_test]

    res = accuracy_score(true_label_test, labels_test)

    return res


subject = 1
subject_dir = Path(f'./subject_{subject}')
z_train = np.load(subject_dir / "z_hat_train.npy")
z_test = np.load(subject_dir / "z_hat_test.npy")
labels_train = np.load(subject_dir / "labels_train.npy")
true_label_test = np.load(subject_dir / "labels_test.npy")

labels_test = classify(z_train, z_test, labels_train, normalize=True)

print(evaluate(dataset, true_label_test, labels_test))

# %%
