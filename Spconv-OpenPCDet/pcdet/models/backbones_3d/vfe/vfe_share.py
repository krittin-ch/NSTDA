import torch
import torch.nn as nn
import torch.nn.functional as F

class VFE(nn.Module):
    def __init__(self, cin, cout):
        super(VFE, self).__init__()
        self.linear = nn.Linear(cin, cout)
        self.batch_norm = nn.BatchNorm1d(cout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x

class VoxelFeatureEncodingWeightShare(nn.Module):
    def __init__(self, cin, cout):
        super(VoxelFeatureEncodingWeightShare, self).__init__()
        self.vfe_x = VFE(cin, cout)
        self.vfe_y = VFE(cin, cout)
        self.vfe_z = VFE(cin, cout)

    def forward(self, voxels):
        num_voxels, max_points, c = voxels.shape

        # Compute the mean (centroid) of the points in each voxel
        mask = (voxels.sum(dim=-1, keepdim=True) != 0)
        voxels_mean = voxels.sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True)

        # Compute the relative offsets for each axis separately
        relative_offsets_x = voxels[:, :, 0] - voxels_mean[:, :, 0]
        relative_offsets_y = voxels[:, :, 1] - voxels_mean[:, :, 1]
        relative_offsets_z = voxels[:, :, 2] - voxels_mean[:, :, 2]

        # Transform through VFE networks for each axis
        transformed_features_x = self.vfe_x(relative_offsets_x)
        transformed_features_y = self.vfe_y(relative_offsets_y)
        transformed_features_z = self.vfe_z(relative_offsets_z)

        # Concatenate the output features from each axis
        concatenated_features = torch.stack((transformed_features_x, transformed_features_y, transformed_features_z), dim=2)

        return concatenated_features

# Example usage
voxels = torch.tensor([
    [
        [1.0, 2.0, 3.0, 0.5],
        [1.1, 2.1, 3.1, 0.4],
        [0.9, 1.9, 2.9, 0.6],
        [0.0, 0.0, 0.0, 0.0]
    ],
    [
        [2.0, 3.0, 4.0, 0.7],
        [2.1, 3.1, 4.1, 0.8],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]  
    ],
], dtype=torch.float32)

# Input dimensions: cin = 1 (for each axis separately)
# Output dimensions: cout = 4 (same for each axis)
vfe_net = VoxelFeatureEncoding(cin=1, cout=4)
encoded_voxels = vfe_net(voxels)
print(encoded_voxels)
print(encoded_voxels.shape)
