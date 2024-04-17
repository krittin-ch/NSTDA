import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

np.random.seed(1)

demo = np.arange(0, 4, 1)

x, y = np.meshgrid(demo, demo)

z = 2*x + 3*y + 4

noise = np.random.normal(0, 1, z.shape)


z_n = z + noise

zeros = np.zeros((16, 4))

a = np.column_stack((x.flatten(), y.flatten(), z_n.flatten(), np.abs(noise.flatten()/2.2)))

voxel = np.concatenate((zeros, a))
print(voxel)

voxel = voxel[~np.all(voxel == 0, axis=1)]

reg = LinearRegression().fit(voxel[:, :2], voxel[:, 2])

z_reg = reg.coef_[0]*x + reg.coef_[1]*y + reg.intercept_

print(z_reg.shape)

print(reg.coef_)
print(reg.intercept_)

# scatter = go.Scatter3d(
#     x=x.flatten(), 
#     y=y.flatten(), 
#     z=z_n.flatten(), 
#     mode='markers', 
#     marker=dict(size=2)
# )

# plane = go.Surface(
#     x=x, 
#     y=y, 
#     z=z, 
#     colorscale='Viridis',
#     opacity=1
# )

# reg_plane = go.Surface(
#     x=x, 
#     y=y, 
#     z=z_reg, 
#     colorscale='Earth',
#     opacity=0.6
# )

# fig = go.Figure(data=[scatter, plane, reg_plane])



# fig.update_layout(title='3D Plot',
#                   autosize=False,
#                   width=500,
#                   height=500,
#                   margin=dict(l=65, r=50, b=65, t=90))

# fig.show()

def normal_vector_gen(voxel):
    voxel = voxel[~np.all(voxel == 0, axis=1)]
    voxel = voxel[:, 0:3]
    reg = LinearRegression().fit(voxel[:, :2], voxel[:, 2])
    z_reg = reg.coef_[0]*x + reg.coef_[1]*y + reg.intercept_
    normal_vector = np.array([reg.coef_[0], reg.coef_[1], -1])
    normal_vector = normal_vector/np.linalg.norm(normal_vector)
 
    return normal_vector


def fast_normal_vector_gen(voxel):
    voxel = voxel[~np.all(voxel == 0, axis=1)]
    voxel = voxel[:, 0:3]
    voxel_mean = np.mean(voxel, axis=0)
    voxel_demean = voxel - voxel_mean
    _, _, Vt = np.linalg.svd(voxel_demean, full_matrices=False)
    normal_vector = Vt[-1, :]

    return normal_vector

# def fast_normal_vector_gen(voxels):
#     normal_vectors = []
#     for voxel in voxels:
#         voxel = voxel[~np.all(voxel == 0, axis=1)]
#         voxel = voxel[:, 0:3]
#         voxel_mean = np.mean(voxel, axis=0)
#         voxel_demean = voxel - voxel_mean
#         _, _, Vt = np.linalg.svd(voxel_demean, full_matrices=False)
#         normal_vector = Vt[-1, :]
#         normal_vectors.append(normal_vector)

#     return np.array(normal_vectors)


import time
start_time = time.time()

# for i in range(15000):
#     normal_vector = normal_vector_gen(voxel=voxel)

print("--- %s seconds ---" % ((time.time() - start_time)))
print(normal_vector)

start_time = time.time()

for i in range(15000):
    normal_vector = fast_normal_vector_gen(voxel=voxel)

print("--- %s seconds ---" % ((time.time() - start_time)))
print(normal_vector)
