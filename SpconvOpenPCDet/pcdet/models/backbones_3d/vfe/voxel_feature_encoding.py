import torch
import torch.nn as nn
import torch.nn.functional as F

class VFE(nn.Module):
    def __init__(self, cin, cout):
        super(VFE, self).__init__()
        self.linear = nn.Linear(cin, cout // 2)
        self.batch_norm = nn.BatchNorm1d(cout // 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x

class VoxelFeatureEncoding(nn.Module):
    def __init__(self, cin, cout):
        super(VoxelFeatureEncoding, self).__init__()
        self.vfe = VFE(cin, cout)

    def forward(self, voxels):
        num_voxels, max_points, c = voxels.shape

        # Compute the mean (centroid) of the points in each voxel
        mask = (voxels.sum(dim=-1, keepdim=True) != 0)
        voxels_mean = voxels.sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True)

        # Compute the relative offsets
        relative_offsets = voxels[:, :, :3] - voxels_mean[:, :, :3] 

        # Concatenate the original features with the relative offsets
        augmented_voxels = torch.cat((voxels, relative_offsets), dim=-1)

        # Transform through VFE layer
        transformed_features = self.vfe(augmented_voxels.view(-1, augmented_voxels.shape[-1]))
        transformed_features = transformed_features.view(num_voxels, max_points, -1)

        # Element-wise max pooling to get the locally aggregated feature
        aggregated_feature, _ = torch.max(transformed_features, dim=1, keepdim=True)

        # Concatenate point-wise features with aggregated feature
        concatenated_features = torch.cat((transformed_features, aggregated_feature.repeat(1, max_points, 1)), dim=-1)

        return concatenated_features

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

# Input dimensions: cin = 7 (xi, yi, zi, ri, xi - vx, yi - vy, zi - vz)
# Output dimensions: cout = 64
vfe_net = VoxelFeatureEncoding(cin=7, cout=4)
encoded_voxels = vfe_net(voxels)
print(encoded_voxels)
print(voxels.shape)
