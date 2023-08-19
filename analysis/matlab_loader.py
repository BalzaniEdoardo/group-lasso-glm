import numpy as np
import scipy.io as sio
from numpy.typing import NDArray
from typing import Tuple
#import pynapple as nap

data_path = "../data/m691l1#4_second_64_workspace.mat"


def get_rate(spike_times, mn, mx):
    rates = np.zeros(spike_times.shape[0])
    for neu in range(rates.shape[0]):
        for tr in range(spike_times.shape[1]):
            rates[neu] = rates[neu] + len(spike_times[neu,tr].flatten())
    return rates/(spike_times.shape[1] * (mx-mn))


def load_data(path:str, min_fr_th_hz=1.) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """Load experiment with gratings stimuli.

    Parameters
    ----------
    path:
        The path to the .mat file

    Returns
    -------
        curated_units:
            The ID of the unitsShape (n_units, ).
        spike_times:
            The spike times (starting from t0 at stimulus presentation). (n_units, n_trials)
        spatial_frequencies:
            The frequency for each stimulus, values: -1,1,2,3. Shape (n_units, ).
        orientations:
            The orientation for each stimulus in deg, values: -1, 10, 20, 30,..., 360. Shape (n_units, ).
    """
    dat = sio.loadmat(path)
    # extract curated units and set to python indices
    curated_units = dat["clus"].flatten() - 1

    # (n_units, n_trials) sec
    spike_times = dat["ori_sf_struct"][0,0]['passSTs'][curated_units]

    # (n_trials, )
    spatial_frequencies = dat["ori_sf_struct"][0,0]["events_sf"].flatten()

    # (n_trials, ) 0 = orizontal, 90 vertical...
    orientations = dat["ori_sf_struct"][0,0]["events_ori"].flatten()

    mx = max([max(spike_times[i, j]) for i in range(spike_times.shape[0])
               for j in range(spike_times.shape[1])
               if len(spike_times[i, j]) != 0])
    mn = min([min(spike_times[i, j]) for i in range(spike_times.shape[0])
               for j in range(spike_times.shape[1])
               if len(spike_times[i, j]) != 0])
    select = get_rate(spike_times, mn, mx) > min_fr_th_hz
    curated_units = curated_units[select]
    spike_times = spike_times[select]

    print(f"Range of spike times: [{mn[0]}s, {mx[0]}s]")
    return curated_units, spike_times, spatial_frequencies, orientations

# def load_data_pynapple(path:str, min_fr_th_hz=1.) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
#     """Load experiment with gratings stimuli.

#     Parameters
#     ----------
#     path:
#         The path to the .mat file

#     Returns
#     -------
#         curated_units:
#             The ID of the unitsShape (n_units, ).
#         spike_times:
#             The spike times (starting from t0 at stimulus presentation). (n_units, n_trials)
#         spatial_frequencies:
#             The frequency for each stimulus, values: -1,1,2,3. Shape (n_units, ).
#         orientations:
#             The orientation for each stimulus in deg, values: -1, 10, 20, 30,..., 360. Shape (n_units, ).
#     """
#     dat = sio.loadmat(path)
#     # extract curated units and set to python indices
#     curated_units = dat["clus"].flatten() - 1

#     # (n_units, n_trials) sec
#     spike_times = dat["ori_sf_struct"][0,0]['passSTs'][curated_units]

#     # dic ts
#     time_init = dat["ori_sf_struct"][0, 0]["times"][0, 0].flatten()
#     ts_group = {unt: nap.Ts(np.hstack([spike_times[i, tr].flatten() + time_init[tr] for tr in range(spike_times.shape[1])])) for i,unt in enumerate(curated_units)}
#     spike_time_tsg = nap.TsGroup(ts_group)

#     start_time = nap.Ts(time_init)
#     epochs = nap.IntervalSet(start=time_init, end=time_init+0.5)

#     # (n_trials, )
#     spatial_frequencies = dat["ori_sf_struct"][0,0]["events_sf"].flatten()

#     # (n_trials, ) 0 = orizontal, 90 vertical...
#     orientations = dat["ori_sf_struct"][0,0]["events_ori"].flatten()

#     mx = max([max(spike_times[i, j]) for i in range(spike_times.shape[0])
#                for j in range(spike_times.shape[1])
#                if len(spike_times[i, j]) != 0])
#     mn = min([min(spike_times[i, j]) for i in range(spike_times.shape[0])
#                for j in range(spike_times.shape[1])
#                if len(spike_times[i, j]) != 0])
#     select = get_rate(spike_times, mn, mx) > min_fr_th_hz
#     curated_units = curated_units[select]
#     spike_times = spike_times[select]

#     print(f"Range of spike times: [{mn[0]}s, {mx[0]}s]")
#     return curated_units, spike_times, spatial_frequencies, orientations

def bin_spikes(spike_times: NDArray,
               dt_sec: float,
               tp_min: float = 0,
               tp_max: float = 0.5) -> Tuple[NDArray, NDArray]:
    """Bin the spike times.

    Parameters
    ----------
    spike_times:
        The spike times (starting from t0 at stimulus presentation). (n_units, n_trials)
    dt_sec:
        Sampling period in sec.
    tp_min:
        Min time point stamps in sec.
    tp_max:
        Max time point stamps in sec.

    Returns
    -------
        time:
            Shape (n_time_points, ).
        spk_binned:
            Shape (n_trials, n_time_points, n_neurons).
    """
    n_bins = int(np.floor((tp_max - tp_min) / dt_sec))
    edges = np.arange(0, n_bins + 1) * dt_sec
    spk_binned = np.zeros((spike_times.shape[1], n_bins, spike_times.shape[0]))
    for neu in range(spike_times.shape[0]):
        for tr in range(spike_times.shape[1]):
            spk_binned[tr, :, neu] = np.histogram(spike_times[neu, tr].flatten(), bins=edges)[0]
    time = edges[:-1]
    return time, spk_binned


def compute_psth(spk_binned, orientations, spatial_frequencies, frequency=1):
    unq_orientations = np.unique(orientations)
    unq_orientations = unq_orientations[unq_orientations != -1]
    psth = np.zeros((spk_binned.shape[1], unq_orientations.shape[0], spk_binned.shape[2]))

    cnt = 0
    for ori in unq_orientations:
        psth[:, cnt] = spk_binned[(orientations == ori) & (spatial_frequencies == frequency), :, :].mean(axis=0)
        cnt += 1

    return psth
