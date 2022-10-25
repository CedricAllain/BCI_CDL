# %%
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, cohen_kappa_score

from moabb.datasets import BNCI2014001

from utils_plot import plot_atoms

dataset = BNCI2014001()


def corr_classify(z_train, z_test, labels_train, normalize=True, tmin=0):
    """

    tmin : int | float
        seconde à partir de laquelle prendre en compte le vecteur z

    """

    tmin_idx = np.rint(tmin * 250.).astype(int)
    z_train = z_train[:, :, tmin_idx:]
    z_test = z_test[:, :, tmin_idx:]

    n_trials, n_atoms, _ = z_train.shape

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


def tfidf_classifier(z_train, z_test, labels_train, tmin=0, norm_ord=0,
                     threshold=0, top_n=5, sfreq=250., plot=False):
    """

    threshold : 
        Percentiles must be in the range [0, 100]

    """

    tmin_idx = np.rint(tmin * sfreq).astype(int)
    z_train = z_train[:, :, tmin_idx:]
    z_test = z_test[:, :, tmin_idx:]

    # apply threshold
    def apply_threshold(z, threshold):
        z_nan = z_train.copy()
        z_nan[z_nan == 0] = np.nan
        threshold = np.nanpercentile(z_nan, threshold, axis=2, keepdims=True)
        mask = z < threshold
        z[mask] = 0
        return z

    z_train = apply_threshold(z_train, threshold)
    z_test = apply_threshold(z_test, threshold)

    if norm_ord == 0:
        norm_func = np.count_nonzero
    else:
        def norm_func(x, axis=None):
            return np.linalg.norm(x, ord=norm_ord, axis=axis)

    tf = np.array([norm_func(z, axis=1) / norm_func(z) for z in z_train])

    tf_class = dict()
    data_heatmap = []
    for label in np.unique(labels_train):
        idx_label = np.where(labels_train == label)[0]
        tf_class[label] = tf[idx_label].mean(axis=0)
        data_heatmap.append(tf_class[label])
    data_heatmap = np.array(data_heatmap)

    fig, ax = plt.subplots(figsize=(4, 3))
    fig = sns.heatmap(data_heatmap, yticklabels=tf_class.keys(), ax=ax)
    # plt.yticks(rotation=90)
    plt.show()

    if plot:
        df_tf = pd.DataFrame(tf_class)
        df_tf.plot()
        plt.xlim(0, 39)
        plt.show()

    top_atoms = {label: np.argpartition(this_tf, -top_n)[-top_n:]
                 for label, this_tf in tf_class.items()}
    # top_atoms = {
    #     'feet': [12, 11, 25],
    #     'left_hand': [35, 39, 36],
    #     'right_hand': [20, 35, 39],
    #     'tongue': [3, 4, 25]
    # }

    labels_test = []
    n_random = 0
    for this_z in tqdm(z_test):
        # for each new trial, compute relative atom frequency
        new_tf = np.array(norm_func(
            this_z, axis=1) / norm_func(this_z))
        idx_tf = top_atoms[label]
        norm = np.inf
        lbl = None
        for label, this_tf in tf_class.items():
            norm_lbl = np.linalg.norm(new_tf[idx_tf] - tf_class[label][idx_tf])
            if norm_lbl < norm:
                lbl = label
                norm = norm_lbl

        if lbl is None:
            lbl = np.random.choice(np.unique(labels_train), size=1)[0]
            n_random += 1
        labels_test.append(lbl)

    if n_random:
        print(f"Number of random assignments: {n_random}")

    return labels_test, tf_class, top_atoms


def dummy_classifier(z_test, labels_train):
    """

    """
    labels_set = np.unique(labels_train)
    labels_test = np.random.choice(
        labels_set, size=z_test.shape[0], replace=True)
    return labels_test


def evaluate(dataset, true_label, predict_labels, separate_hands=True):

    event_id = dataset.event_id
    if not separate_hands:
        event_id['right_hand'] = event_id['left_hand']

    true_label = [event_id[lbl] for lbl in true_label]
    predict_labels = [event_id[lbl] for lbl in predict_labels]

    acc = accuracy_score(true_label, predict_labels)
    kappa = cohen_kappa_score(true_label, predict_labels)

    return acc, kappa


def cohen_kappa_coeff(dataset, true_label_test, labels_test, only_acc=False,
                      separate_hands=True):
    """
    Source: Schlögl et al., 2007
    https://pub.ist.ac.at/~schloegl/publications/Schloegl2007_EvaluationCriteria.pdf
    """

    coeff = None

    event_id = dataset.event_id
    if not separate_hands:
        event_id['right_hand'] = event_id['left_hand']

    true_label_test = [event_id[lbl] for lbl in true_label_test]
    labels_test = [event_id[lbl] for lbl in labels_test]

    n_classes = len(np.unique(true_label_test))

    p0 = accuracy_score(true_label_test, labels_test)
    if only_acc:
        return p0

    conf_matrix = confusion_matrix(true_label_test, labels_test)
    pe = np.sum([conf_matrix[:, i].sum() * conf_matrix[i, :].sum()
                 for i in range(n_classes)])
    pe /= n_classes**2

    coeff = (p0 - pe) / (1 - pe)

    return coeff


if __name__ == '__main__':

    subject = 9
    subject_dir = Path(f'./subject_{subject}')

    z_train = np.load(subject_dir / "z_hat_train.npy")
    z_test = np.load(subject_dir / "z_hat_test.npy")
    labels_train = np.load(subject_dir / "labels_train.npy")
    true_label_test = np.load(subject_dir / "labels_test.npy")

    tmin = 2.5

    labels_test = corr_classify(
        z_train, z_test, labels_train, normalize=True, tmin=tmin)
    corr_accuracy, corr_kappa = evaluate(dataset, true_label_test, labels_test)
    print(f"Correlation classifier accurracy: {corr_accuracy}")

    labels_test, tf_class, top_atoms = tfidf_classifier(
        z_train, z_test, labels_train, tmin=tmin, norm_ord=0, top_n=5)
    tf_accuracy, tf_kappa = evaluate(dataset, true_label_test, labels_test)
    print(f"TF classifier accurracy: {tf_accuracy}")
# %%
