import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def bicolor_cmap(bounds=[0,0.5,1], colors=['white', 'k']):
    # Define the colormap
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap

def plot_coupling_mask(is_coupled, colors=['white', 'k']):
    # plot coupling matrix
    fig = plt.figure()
    plt.title('Coupling Map')
    cax = plt.imshow(is_coupled,cmap=bicolor_cmap(colors=colors))

    # Minor ticks
    cax.axes.set_xticks(np.arange(-.5, is_coupled.shape[0], 1), minor=True)
    cax.axes.set_yticks(np.arange(-.5, is_coupled.shape[1], 1), minor=True)
    cax.axes.grid(which='minor', color='r', linestyle='-', linewidth=2)

    cax.axes.set_xlabel("sender neuron")
    cax.axes.set_ylabel("receiver neuron")

    cbar = plt.colorbar()
    cbar.set_ticks([0.25, 0.75])
    cbar.set_ticklabels(['uncoupled', 'coupled'])

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
    time = np.arange(0, ws) * dt
    for neu_i in range(rows):
        for neu_j in range(cols):
            ax = axs[neu_i, neu_j]
            ax.set_title(f"neu {neu_i} -> neu {neu_j}")
            ax.plot(time, coupling_filter_bank[neu_i, neu_j], **plot_kwargs)
    if fig_and_axs is None:
        fig.tight_layout()
    return fig, axs


