import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from matlab_loader import load_data
import matplotlib.pylab as plt
from mayavi import mlab

min_rate_hz = 1.
data_path = 'data/m691l1#4_second_64_workspace.mat'

dat_ridge = np.load('results/ridge_rates_DT_1ms_NBasis_5_WindowSize_250.npz')
dat_group_lasso = np.load('results/group-lasso_rates_DT_1ms_NBasis_5_WindowSize_250.npz')

rates_ridge = dat_ridge['predicted_rate']
rates_group_lasso = dat_group_lasso['predicted_rate']


curated_units, spike_times, spatial_frequencies, orientations = load_data(data_path, min_rate_hz)

n_trial, n_time_points, n_neurons = rates_group_lasso.shape
unq_ori = np.unique(orientations)
n_ori = unq_ori.shape[0]
modelX_group_lasso = np.zeros(((n_ori-1)*n_time_points, n_neurons))
modelX_ridge = np.zeros(((n_ori-1)*n_time_points, n_neurons))


cnt_ori = 0
for ori in unq_ori[1:]:
    modelX_group_lasso[cnt_ori*n_time_points:(cnt_ori+1)*n_time_points, :] = rates_group_lasso[orientations==ori].mean(axis=0)
    modelX_ridge[cnt_ori*n_time_points:(cnt_ori+1)*n_time_points, :] = rates_ridge[orientations==ori].mean(axis=0)

    cnt_ori += 1



pipeline = Pipeline([
    ('pca', PCA(n_components=3))  # Set the number of components to retain
])


pcs_group_lasso = pipeline.fit_transform(modelX_group_lasso)
pcs_group_lasso = pcs_group_lasso.reshape(n_ori-1, -1, 3)

fig = plt.figure()
for k in range(1, n_ori, 6):
    plt.plot(pcs_group_lasso[k,:,0],  pcs_group_lasso[k,:,1])

plt.title('PCA visual responses')
plt.xlabel("pc 1")
plt.ylabel("pc 2")
plt.tight_layout()
plt.savefig("results/unscaled_pca_group_lasso_by_ori.png")


pcs_ridge = pipeline.fit_transform(modelX_ridge)
pcs_ridge = pcs_ridge.reshape(n_ori-1, -1, 3)

fig = plt.figure()
for k in range(1, n_ori, 4):
    plt.plot(pcs_ridge[k,:,0], pcs_ridge[k,:,1])

plt.title('PCA visual responses')
plt.xlabel("pc 1")
plt.ylabel("pc 2")
plt.tight_layout()
plt.savefig("results/unscaled_pca_ridge_by_ori.png")



#%%
# Z-scored pca

cnt_ori = 0
for ori in unq_ori[1:]:
    modelX_group_lasso[cnt_ori*n_time_points:(cnt_ori+1)*n_time_points, :] = rates_group_lasso[orientations==ori].mean(axis=0)
    modelX_ridge[cnt_ori*n_time_points:(cnt_ori+1)*n_time_points, :] = rates_ridge[orientations==ori].mean(axis=0)

    cnt_ori += 1
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=3))  # Set the number of components to retain
])

pcs_group_lasso = pipeline.fit_transform(modelX_group_lasso)
pcs_group_lasso = pcs_group_lasso.reshape(n_ori-1, -1, 3)

fig = plt.figure()
for k in range(1, n_ori, 4):
    plt.plot(pcs_group_lasso[k,:,0],  pcs_group_lasso[k,:,1])

plt.title('PCA visual responses')
plt.xlabel("pc 1")
plt.ylabel("pc 2")
plt.tight_layout()
plt.savefig("results/unscaled_pca_group_lasso_by_ori.png")


pcs_ridge = pipeline.fit_transform(modelX_ridge)
pcs_ridge = pcs_ridge.reshape(n_ori-1, -1, 2)

fig = plt.figure()
for k in range(1, n_ori, 6):
    plt.plot(pcs_ridge[k,:,0], pcs_ridge[k,:,1])

plt.title('PCA visual responses')
plt.xlabel("pc 1")
plt.ylabel("pc 2")
plt.tight_layout()
plt.savefig("results/pca_ridge_by_ori.png")

#%%
# 1 orientation

for ori in unq_ori:
    if ori != 60:
        continue

    modelX_group_lasso = rates_group_lasso[orientations==ori].mean(axis=0)
    modelX_ridge = rates_ridge[orientations==ori].mean(axis=0)


    pipeline = Pipeline([
        ('pca', PCA(n_components=3))  # Set the number of components to retain
    ])

    pcs_group_lasso = pipeline.fit_transform(modelX_group_lasso)
    pcs_group_lasso = pcs_group_lasso.reshape(1, -1, 3)

    fig = plt.figure()
    for k in [0]:
        mlab.plot3D(pcs_group_lasso[k,:,0],  pcs_group_lasso[k,:,1], pcs_group_lasso[k,:,2], np.linspace(0,1,n_time_points), 
                    tube_radius=0.025, colormap='Spectral', fig=fig)

    plt.title('PCA visual responses')
    ax.set_xlabel("pc 1")
    ax.set_ylabel("pc 2")
    ax.set_zlabel("pc 3")
    plt.tight_layout()
    plt.savefig(f"results/single_orientation_{ori}_lasso_by_ori.png")


    pcs_ridge = pipeline.fit_transform(modelX_ridge)
    pcs_ridge = pcs_ridge.reshape(1, -1, 3)

    ax = plt.figure().add_subplot(projection='3d')
    for k in [0]:
        ax.plot3D(pcs_ridge[k,:,0], pcs_ridge[k,:,1], pcs_ridge[k,:,2], lw=2, color="grey")

    plt.title('PCA visual responses')
    ax.set_xlabel("pc 1")
    ax.set_ylabel("pc 2")
    ax.set_zlabel("pc 3")
    plt.tight_layout()
    plt.savefig(f"results/single_orientation_{ori}_pca_ridge_by_ori.png")
    plt.close("all")