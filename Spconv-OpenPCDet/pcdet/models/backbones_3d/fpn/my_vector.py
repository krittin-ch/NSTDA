import numpy as np

def fast_normal_vector_gen(voxel):
    voxel = voxel[~np.all(voxel == 0, axis=1)]
    voxel = voxel[:, 0:3]
    voxel_mean = np.mean(voxel, axis=0)
    voxel_demean = voxel - voxel_mean
    _, _, Vt = np.linalg.svd(voxel_demean, full_matrices=False)
    normal_vector = Vt[-1, :]

    return normal_vector