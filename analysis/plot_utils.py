import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def bicolor_cmap(bounds=[0,0.5,1], colors=['white', 'k']):
    # Define the colormap
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap


def plot_coupling_mask(is_coupled, colors=['white', 'k'],
                       cmap=None, title="Coupling Map", set_labels=False):
    fig, ax = plt.subplots()
    ax.set_title(title)
    if cmap is None:
        cmap = bicolor_cmap(colors=colors)
    else:
        cmap = plt.get_cmap(cmap)

    cax = ax.imshow(is_coupled, cmap=cmap, vmin=0, vmax=1)

    # Set the ticks themselves
    ax.set_xticks(np.arange(is_coupled.shape[1]))
    ax.set_yticks(np.arange(is_coupled.shape[0]))

    # Minor ticks
    ax.set_xticks(np.arange(-.5, is_coupled.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, is_coupled.shape[0], 1), minor=True)

    ax.set_xlabel("sender neuron")
    ax.set_ylabel("receiver neuron")

    # Here's the key line:
    ax.grid(which='minor', color='r', linestyle='-', linewidth=2)
    if set_labels:
        cbar = plt.colorbar(cax, ticks=[0, 1])
        cbar.set_ticks([0.25, 0.75])
        cbar.set_ticklabels(['uncoupled', 'coupled'])
    else:
        cbar = plt.colorbar(cax)
        cbar.ax.set_ylabel("coupling strength")
    plt.show()
    return fig

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
            ax.set_title(f"neu {neu_i} -> neu {neu_j}")
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


