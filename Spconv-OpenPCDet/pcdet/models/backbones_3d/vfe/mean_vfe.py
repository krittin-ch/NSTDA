import torch
import torch.nn as nn 
# from .vfe_template import VFETemplate



class VFETemplate(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

    def get_output_feature_dim(self):
        raise NotImplementedError

    def forward(self, **kwargs):
        """
        Args:
            **kwargs:

        Returns:
            batch_dict:
                ...
                vfe_features: (num_voxels, C)
        """
        raise NotImplementedError


class MeanVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features

    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        voxel_features, voxel_num_points = batch_dict['voxels'], batch_dict['voxel_num_points']
        points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
        points_mean = points_mean / normalizer
        batch_dict['voxel_features'] = points_mean.contiguous()

        return batch_dict

batch_dict = {
    'voxels': torch.tensor([
        [
            [1.0, 2.0, 3.0, 0.5],
            [1.5, 2.5, 3.5, 2.0],
            [0.9, 1.9, 2.9, 0.6],
            [0.0, 0.0, 0.0, 0.0]  # Assuming zero padding for unused points
        ],
        [
            [2.0, 3.0, 4.0, 0.7],
            [2.1, 3.1, 4.1, 0.8],
            [2.2, 3.2, 4.2, 0.9],
            [0.0, 0.0, 0.0, 0.0]  # Assuming zero padding for unused points
        ]
    ], dtype=torch.float32),
    'voxel_num_points': torch.tensor([3, 3], dtype=torch.int32)  # Number of points in each voxel
}

model_cfg = {}

# num_point_features = 4. x, y, z, r
mean_vfe = MeanVFE(model_cfg=model_cfg, num_point_features=4)
output_batch_dict = mean_vfe(batch_dict)

print("Voxel Features:")
print(output_batch_dict['voxel_features'])
