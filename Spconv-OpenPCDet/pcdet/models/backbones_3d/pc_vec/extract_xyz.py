import numpy as np

def get_xyz(norm_vec):
    x = norm_vec[:, :, :1]
    y = norm_vec[:, :, 1:2]
    
    xy = np.power((x**2 + y**2), 0.5)

    z = norm_vec[:, :, -1:]
    print(z)

    return xy, z

