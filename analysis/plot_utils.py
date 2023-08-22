import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list
def bicolor_cmap(bounds=[0,0.5,1], colors=['white', 'k']):
    # Define the colormap
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap



def plot_coupling_mask(*is_coupled_all, colors=['white', 'k'],
                       cmap=None, title=["Coupling Map"], set_labels=False,
                       plot_grid=True, plot_ticks_every=1, lw=2,
                       sort=False, high_percentile=98,low_percentile=2, vmax=None, vmin=None):


    num_comp = len(is_coupled_all)
    fig, axs = plt.subplots(1, num_comp)
    if num_comp == 1:
        axs = [axs]
    for k in range(num_comp):

        is_coupled = is_coupled_all[k]
        if (k == 0) and (vmax is None):
            vmax = np.nanpercentile((is_coupled - 10**3 * np.eye(is_coupled.shape[0])).max(axis=0), high_percentile)
        if (k == 0) and (vmin is None):
            vmax = np.nanpercentile((is_coupled - 10**3 * np.eye(is_coupled.shape[0])).max(axis=0), low_percentile)

        print("vmax:", vmax)
        if sort and k == 0:
            # Hierarchical clustering on rows
            link = linkage(is_coupled, method='average')
            order = leaves_list(link)
            is_coupled = is_coupled[order]
            is_coupled = is_coupled[:, order]

        ax = axs[k]
        ax.set_title(title[k])
        if cmap is None:
            cmap = bicolor_cmap(colors=colors)
        else:
            cmap = plt.get_cmap(cmap)

        cax = ax.imshow(is_coupled, cmap=cmap, vmin=vmin, vmax=vmax)

        # Set the ticks themselves
        ax.set_xticks(np.arange(is_coupled.shape[1])[::plot_ticks_every])
        ax.set_yticks(np.arange(is_coupled.shape[0])[::plot_ticks_every])

        # Minor ticks
        ax.set_xticks(np.arange(-.5, is_coupled.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-.5, is_coupled.shape[0], 1), minor=True)

        ax.set_xlabel("receiver neuron")
        ax.set_ylabel("sender neuron")

        if plot_grid:
            ax.grid(which='minor', color='r', linestyle='-', linewidth=lw)

        # Adjusting the colorbar height to match the main plot
        divider = make_axes_locatable(ax)
        cbar_ax = divider.append_axes("right", size="5%", pad=0.05)
        
        if set_labels:
            cbar = plt.colorbar(cax, cax=cbar_ax, ticks=[0, 1])
            cbar.set_ticks([0.25, 0.75])
            cbar.set_ticklabels(['uncoupled', 'coupled'])
        else:
            cbar = plt.colorbar(cax, cax=cbar_ax)
            cbar.ax.set_ylabel("coupling strength")
        
    plt.tight_layout()
    plt.show()
    return fig, axs

def plot_filters(coupling_filter_bank, dt, fig_and_axs=None, **plot_kwargs):
    rows, cols, ws = coupling_filter_bank.shape
    if fig_and_axs is None:
        fig, axs = plt.subplots(rows, cols, figsize=(10, 8))
        if (rows == 1) and (cols == 1):
            axs = np.array([[axs]])
        elif rows == 1 or cols == 1:
            axs = np.array([axs])

    else:
        fig, axs = fig_and_axs

    ylim = [np.inf, -np.inf]
    time = np.arange(0, ws) * dt
    for neu_i in range(rows):
        for neu_j in range(cols):
            ax = axs[neu_i, neu_j]
            ax.set_title(f"neu {neu_j} -> neu {neu_i}")
            ax.plot(time, coupling_filter_bank[neu_i, neu_j], **plot_kwargs)
            y0, y1 = ax.get_ylim()
            ylim[0] = min(y0, ylim[0])
            ylim[1] = max(y1, ylim[1])
    for neu_i in range(rows):
        for neu_j in range(cols):
            axs[neu_i, neu_j].set_ylim(ylim)

    if fig_and_axs is None:
        fig.tight_layout()
    return fig, axs



def plot_psth_by_category(time, predicted_rate, category_vector, rows=7, cols=10, plot_every=6, is_angle=True):
    unq_ori = np.unique(category_vector)
    # Plot psths
    plot_every = 6
    fig = plt.figure(figsize=(12, 6))
    color_ori = {}
    axs = []
    for neu in range(69):
        ax = plt.subplot(rows, cols, neu+1)
        mn, mx = np.inf, -np.inf
        for ori in unq_ori[1::plot_every]:
            psth = predicted_rate[category_vector == ori, :, neu].mean(axis=0)
            mn = min(mn, psth.min())
            mx = max(mx, psth.max())
            p, = ax.plot(time, psth)
            color_ori[ori] = p.get_color()
        if neu // cols == (rows - 1):
            ax.set_xticks([0.25, 0.5])
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
        else:
            ax.set_xticks([])
        if neu%cols == 0:
            plt.ylabel("rate[Hz]")
        ax.set_yticks([mn, mx])
        ax.set_yticklabels([int(mn), int(mx)], fontsize=8)
        axs.append(ax)

    if is_angle:
        ax = plt.subplot(rows, cols, neu + 2)
        ax.set_aspect('equal', 'box')
        for ori in unq_ori[1::plot_every]:
            rad = np.deg2rad(ori)
            x = np.cos(rad)
            y = np.sin(rad)
            # Plot the arrow
            ax.arrow(0, 0, x * 0.9, y * 0.9, head_width=0.05, head_length=0.1, fc=color_ori[ori], ec=color_ori[ori])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        axs.append(ax)
        fig.tight_layout()
    return fig, axs