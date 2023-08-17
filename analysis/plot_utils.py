import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

def bicolor_cmap(bounds=[0,0.5,1], colors=['white', 'k']):
    # Define the colormap
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap



def plot_coupling_mask(*is_coupled_all, colors=['white', 'k'],
                       cmap=None, title=["Coupling Map"], set_labels=False,
                       plot_grid=True):
    num_comp = len(is_coupled_all)
    fig, axs = plt.subplots(1, num_comp)
    if num_comp == 1:
        axs = [axs]
    for k in range(num_comp):
        is_coupled = is_coupled_all[k]
        ax = axs[k]
        ax.set_title(title[k])
        if cmap is None:
            cmap = bicolor_cmap(colors=colors)
        else:
            cmap = plt.get_cmap(cmap)

        cax = ax.imshow(is_coupled, cmap=cmap, vmin=min([cc.min() for cc in is_coupled_all]), vmax=max([cc.max() for cc in is_coupled_all]))

        # Set the ticks themselves
        ax.set_xticks(np.arange(is_coupled.shape[1]))
        ax.set_yticks(np.arange(is_coupled.shape[0]))

        # Minor ticks
        ax.set_xticks(np.arange(-.5, is_coupled.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-.5, is_coupled.shape[0], 1), minor=True)

        ax.set_xlabel("receiver neuron")
        ax.set_ylabel("sender neuron")

        if plot_grid:
            ax.grid(which='minor', color='r', linestyle='-', linewidth=2)

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


