# %%
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns

from moabb.datasets import BNCI2014001

from utils_evaluate import tfidf_classifier, evaluate


def get_tfidf_acc(subject_id, tmin=2.5, norm_ord=0,
                  threshold=0, n_atoms_max=20, separate_hands=True):
    subject_dir = Path(f'./subject_{subject_id}')

    z_train = np.load(subject_dir / "z_hat_train.npy")
    z_test = np.load(subject_dir / "z_hat_test.npy")
    labels_train = np.load(subject_dir / "labels_train.npy")
    true_label_test = np.load(subject_dir / "labels_test.npy")

    n_trial, n_atoms, n_times_trial = z_train.shape
    n_atoms_max = np.min([n_atoms, n_atoms_max])

    xx = np.array(range(n_atoms_max)) + 1
    yy_acc_E = []
    yy_kappa_E = []
    yy_acc_T = []
    yy_kappa_T = []
    for this_n in xx:
        # evaluate on session E (test set)
        labels_test, _, _ = tfidf_classifier(
            z_train, z_test, labels_train, tmin=tmin, norm_ord=norm_ord,
            threshold=threshold, top_n=this_n)
        tf_accuracy, tf_kappa = evaluate(
            dataset, true_label_test, labels_test, separate_hands)
        yy_acc_E.append(tf_accuracy)
        yy_kappa_E.append(tf_kappa)
        # evaluate on session T (train set)
        labels_test, _, _ = tfidf_classifier(
            z_train, z_train, labels_train, tmin=tmin, norm_ord=norm_ord,
            threshold=threshold, top_n=this_n)
        tf_accuracy, tf_kappa = evaluate(
            dataset, true_label_test, labels_test, separate_hands)
        yy_acc_T.append(tf_accuracy)
        yy_kappa_T.append(tf_kappa)

    df_acc_E = pd.DataFrame(
        data=dict(top_n=xx, value=yy_acc_E, metric='accuracy', session='E'))
    df_kappa_E = pd.DataFrame(
        data=dict(top_n=xx, value=yy_kappa_E, metric="Cohen's kappa", session='E'))
    df_acc_T = pd.DataFrame(
        data=dict(top_n=xx, value=yy_acc_T, metric='accuracy', session='T'))
    df_kappa_T = pd.DataFrame(
        data=dict(top_n=xx, value=yy_kappa_T, metric="Cohen's kappa", session='T'))
    df = pd.concat([df_acc_E, df_kappa_E, df_acc_T, df_kappa_T])
    df['subject_id'] = subject_id

    return df


def plot_evaluate_tfdif(dataset=BNCI2014001()):
    """
    """
    classifier_kwargs = dict(
        tmin=2.5, norm_ord=0, threshold=0, n_atoms_max=20, separate_hands=False)
    subjects = dataset.subject_list
    new_dfs = Parallel(n_jobs=len(subjects), verbose=1)(
        delayed(get_tfidf_acc)(this_subject, **classifier_kwargs)
        for this_subject in subjects)

    df = pd.concat(new_dfs)

    print(classifier_kwargs)
    fig = sns.lineplot(x="top_n", y="value",
                       hue="session", style="metric",
                       data=df)
    plt.xlabel("Number of top atoms")
    plt.title("TF classifier")
    plt.xlim(1, classifier_kwargs['n_atoms_max'])
    plt.show()
# %%
