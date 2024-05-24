import torch

from .vfe_template import VFETemplate


class TinMeanVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features

    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, voxels, **kwargs):

        """
        Args:
            input_sp_tensor: SparseConvTensor containing:
                features: (N, C)
                indices: (N, 4), [batch_idx, z_idx, y_idx, x_idx]

        Returns:
            output_sp_tensor: SparseConvTensor with mean features (N, C)
        """

        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        voxel_features = voxels

        # Calculate the mean voxel features without normalizer
        points_mean = voxel_features.sum(dim=1, keepdim=False) / voxel_features.size(1)

        return points_mean.contiguous()
