# %%
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture
from scipy import odr
import mne
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import ot


def plot_atoms(cdl, info, plotted_atoms='all', sfreq=150., fig_name='all_atoms'):
    """Plot spatial and temporal representations of learned atoms.
    Parameters
    ----------
    u, v : array-like XXX change for cdl object (instance or dict)
    plotted_atoms : int, list, 'all'
        if int, the number of atoms to plots
        if list, the list of atom indexes to plot
        if 'all', plot all the learned atom
        defaults is 'all'
    sfreq : float
        sampling frequency, the signal will be resampled to match this.
    Returns
    -------
    None
    """

    if isinstance(cdl, dict):
        u, v = cdl["u_hat_"], cdl["v_hat_"]
        n_atoms = u.shape[0]
        n_times_atom = int(v.shape[1])  # 1000. ms
    else:
        u, v = cdl.u_hat_, cdl.v_hat_
        n_atoms = cdl.n_atoms
        n_times_atom = cdl.n_times_atom

    if plotted_atoms == 'all':
        plotted_atoms = range(n_atoms)

    t = np.arange(n_times_atom) / sfreq

    # number of plots by atom
    n_plots = 2
    n_columns = min(6, len(plotted_atoms))
    split = int(np.ceil(len(plotted_atoms) / n_columns))
    figsize = (4 * n_columns, 3 * n_plots * split)
    fig, axes = plt.subplots(n_plots * split, n_columns, figsize=figsize)

    for ii, kk in enumerate(plotted_atoms):

        # Select the axes to display the current atom
        i_row, i_col = ii // n_columns, ii % n_columns
        it_axes = iter(axes[i_row * n_plots:(i_row + 1) * n_plots, i_col])

        # Select the current atom
        v_k = v[kk, :]
        u_k = u[kk, :]

        # Plot the spatial map of the atom using mne topomap
        ax = next(it_axes)
        ax.set_title('Atom % d' % kk, pad=0)

        mne.viz.plot_topomap(data=u_k, pos=info, axes=ax, show=False)
        if i_col == 0:
            ax.set_ylabel('Spatial', labelpad=28)

        # Plot the temporal pattern of the atom
        ax = next(it_axes)
        ax.plot(t, v_k)
        ax.set_xlim(min(t), max(t))
        if i_col == 0:
            ax.set_ylabel('Temporal')

    # save figure
    fig.tight_layout()
    path_fig = fig_name
    plt.savefig(path_fig + ".pdf", dpi=300, bbox_inches='tight')
    plt.savefig(path_fig + ".png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_activation_kde_per_class(z, labels, plotted_atoms='all', tmin=0,
                                  threshold=0, min_acti=10, bw_adjust=0.5, sfreq=250.):
    """

    z : 3d-array, shape (n_trial, n_atoms, n_times_trial)

    labels : 1d-array, shape (n_trials,)

    min_acti : int
        minimum number of remaining activations (after applying tmin and
        threshold) to fit a kde

    """

    # only consider timestamps after tmin
    tmin_idx = np.rint(tmin * sfreq).astype(int)
    z = z[:, :, tmin_idx:]

    n_trial, n_atoms, n_times_trial = z.shape
    n_classes = np.unique(labels).size

    if plotted_atoms == 'all':
        plotted_atoms = range(n_atoms)
    else:
        n_atoms = len(plotted_atoms)

    # apply threshold
    z[z == 0] = np.nan
    threshold = np.nanpercentile(z, threshold, axis=2, keepdims=True)
    # shape (n_trial, n_atoms, 1)

    figsize = (4 * n_classes, 3 * n_atoms)
    fig, axes = plt.subplots(n_atoms, n_classes, figsize=figsize, sharex=True)
    axes = np.atleast_2d(axes)

    xx = tmin + (np.array(range(n_times_trial)) / sfreq)

    for i_col, this_label in enumerate(np.unique(labels)):
        this_label_idx = np.where(labels == this_label)[0]
        z_lbl = z[this_label_idx, :, :]
        thresh_lbl = threshold[this_label_idx, :, :]
        for i_row, kk in enumerate(plotted_atoms):
            ax = axes[i_row, i_col]
            med = []
            for i_trial, trial_lbl in enumerate(z_lbl):
                tt = np.where(trial_lbl[kk, :] >=
                              thresh_lbl[i_trial, kk, 0])[0]
                if len(tt) < min_acti:
                    continue
                kde = gaussian_kde(tmin + tt / sfreq)
                kde.set_bandwidth(kde.factor * bw_adjust)
                yy = kde.evaluate(xx)
                med.append(yy)
                ax.plot(xx, yy, alpha=0.2, color='C0')

            if len(med) == 0:
                continue
            med = np.array(med)
            n = med.shape[1]
            M = ot.utils.dist0(n)
            M /= M.max()
            reg = 1e-3
            bary_wass = ot.bregman.barycenter(med.T, M, reg)
            print(bary_wass)
            ax.plot(np.arange(n, dtype=np.float64),
                    bary_wass, color='k', linestyle='--')
            # med = np.median(med, axis=0)
            # ax.plot(xx, med, color='k', linestyle='--')

            ax.set_xlim(xx.min(), xx.max())
            if i_row == 0:
                ax.set_title(this_label)
            elif i_row == (n_atoms - 1):
                ax.set_xlabel("Time (s)")
            if i_col == 0:
                ax.set_ylabel(f"atom {kk}")

    plt.show()


if __name__ == '__main__':
    subject = 9
    subject_dir = Path(f'./subject_{subject}')

    u_hat_ = np.load(subject_dir / "u_hat_.npy")
    v_hat_ = np.load(subject_dir / "v_hat_.npy")

    z_train = np.load(subject_dir / "z_hat_train.npy")
    z_test = np.load(subject_dir / "z_hat_test.npy")
    labels_train = np.load(subject_dir / "labels_train.npy")
    true_label_test = np.load(subject_dir / "labels_test.npy")

    plot_activation_kde_per_class(
        z_train, labels_train, plotted_atoms=[1], tmin=2.5, threshold=70,
        min_acti=10, bw_adjust=1, sfreq=250.)

# %%
# SFREQ = 250.
# trial_id = 0
# n_times_trial = z_train.shape[2]
# columns_name = [i for i in range(n_times_trial)]
# df_z = pd.DataFrame(data=z_train[trial_id], columns=columns_name)
# df_z['atom'] = df_z.index
# df = pd.melt(df_z, id_vars=['atom'], value_vars=columns_name)
# df['variable'] = df['variable'].astype(np.float64)
# df_tt = df[(df['value'] > 0)]
# sns.kdeplot(data=df_tt, x="variable", hue='atom', cut=0, legend=False)
# plt.show()

# tmin = 3
# tmin_idx = np.rint(tmin * SFREQ).astype(int)
# sns.kdeplot(data=df_tt[df_tt['variable'] >= tmin_idx], x="variable",
#             hue='atom', cut=0, legend=False, bw_method='silverman')
# plt.show()

# # %%

# atom_id = 0

# tmin = 2.5
# tmin_idx = np.rint(tmin * SFREQ).astype(int)
# only_pos_tt = True

# threshold = 50  # remove botom 5% of activations

# if only_pos_tt:
#     this_df = df_tt[df_tt['atom'] == atom_id]
#     this_df = this_df[this_df['variable'] >= tmin_idx]
#     if threshold > 0:
#         threshold = np.percentile(this_df['value'], threshold)
#         this_df = this_df[this_df['value'] >= threshold]
#     print(f"number of points: {len(this_df)}")
# else:
#     this_df = df[df['atom'] == atom_id]
#     this_df = this_df[this_df['variable'] >= tmin_idx]

# df_atom = df[(df['atom'] == atom_id) & (df['variable'] >= tmin_idx)]
# x, y = df_atom['variable'], df_atom['value']
# # y /= y.sum()
# fig, ax = plt.subplots()
# sns.kdeplot(data=this_df, x="variable",  # cut=0,
#             legend=True, bw_adjust=.1, ax=ax)
# ax.set_ylim(0, None)
# ax2 = ax.twinx()
# # y[y > 0] = 1
# ax2.plot(x, y, label="input data", color='C1')
# ax2.set_ylim(0, None)
# plt.xlim(x.min(), x.max())
# plt.legend()
# plt.show()

# # %%
# fit_data = this_df['variable']
# bw_adjust = .1
# kde = gaussian_kde(fit_data)
# kde.set_bandwidth(kde.factor * bw_adjust)

# # %%
# for ord in range(2, 11):
#     plt.plot(x, y, label="input data")
#     poly_model = odr.polynomial(ord)
#     data = odr.Data(x, y)
#     odr_obj = odr.ODR(data, poly_model)
#     output = odr_obj.run()  # running ODR fitting
#     poly = np.poly1d(output.beta[::-1])
#     poly_y = poly(x)
#     print(f"RMSE: {mean_squared_error(y, poly_y)}")
#     plt.plot(x, poly_y, label=f"polynomial ODR {ord}")
#     plt.legend()
#     plt.xlim(x.min(), x.max())
#     plt.show()
# # %%
