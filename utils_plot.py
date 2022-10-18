import os
import mne
import numpy as np
import matplotlib.pyplot as plt


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
        n_times_atom = int(round(sfreq * 1.0))  # 1000. ms
    else:
        u, v = cdl.u_hat_, cdl.v_hat_
        n_atoms = cdl.n_components
        n_times_atom = cdl.kernel_size

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
