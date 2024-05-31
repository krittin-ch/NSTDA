from functools import partial

# import spconv
from pcdet.ops import spconv as spconv
import torch
import torch.nn as nn
import torch.nn.functional as F
# import spconv.pytorch as spconv

from ...utils import common_utils
from .spconv_backbone import post_act_block

# import my function
from pc_vec.pointclouds_plane import fast_normal_vector_gen
from pc_vec.extract_xyz import get_xyz

# import voxel feature extraction
from vfe import TinMeanVFE

# planes --> road planes use for training only


class Extract_WeightBias(spconv.SparseModule):

    def __init__(self, inplanes, planes, stride=1, downsample=None, indice_key=None, norm_fn=None):
        super(Extract_WeightBias, self).__init__()

        self.conv1 = nn.Conv3d(
            inplanes, planes, kernel_size=(64, 64, 64), stride=stride, padding=(32, 32, 32), bias=False
        )

        self.bn1 = norm_fn(planes)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv3d(
            inplanes, planes, kernel_size=(64, 64, 64), stride=stride, padding=(32, 32, 32), bias=False
        )
        
        self.bn2 = norm_fn(planes)
        self.relu2 = nn.ReLU()

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        return out

class Extract_WeightBiasVoxel(spconv.SparseModule):

    def __init__(self, inplanes, planes, stride=1, downsample=None, indice_key=None, norm_fn=None):
        super(Extract_WeightBias, self).__init__()

        self.conv1 = nn.Conv3d(
            inplanes, planes, kernel_size=(64, 64, 64), stride=stride, padding=(32, 32, 32), bias=False
        )

        self.bn1 = norm_fn(planes)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv3d(
            inplanes, planes, kernel_size=(64, 64, 64), stride=stride, padding=(32, 32, 32), bias=False
        )
        
        self.bn2 = norm_fn(planes)
        self.relu2 = nn.ReLU()

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        return out



class TinBlock(spconv.SparseModule):

    def __init__(self, inplanes, stride=1, downsample=None, indice_key=None, norm_fn=None, layers=5):
        super(TinBlock, self).__init__()

        '''
        Have to write config file
        to obtain the adaptable layers
        '''

        self.flatten = nn.Flatten()
        self.layers = nn.ModuleList()
        self.norm_fn = norm_fn

        # Create the sequence of dense --> ReLU --> batch norm --> max pool layers
        for _ in range(layers):
            dense_layer = nn.Linear(inplanes, inplanes)  # Fully connected layer
            softmax_layer = nn.ReLU(dim=-1)           # ReLU activation
            norm_layer = norm_fn(inplanes) if norm_fn else nn.BatchNorm1d(inplanes)  # Normalization layer
            maxpool_layer = nn.MaxPool1d(kernel_size=2, stride=2)  # Max pooling layer

            self.layers.append(nn.Sequential(dense_layer, softmax_layer, norm_layer, maxpool_layer))

    def forward(self, x):
        # Flatten the input features
        out = self.flatten(x)

        # Initialize a tensor to accumulate the outputs
        accumulated_out = torch.zeros_like(out)

        # Pass through the sequence of layers
        for layer in self.layers:
            out = layer(out)
            accumulated_out += out  # Accumulate the output with skip connection

        return accumulated_out


class TinNet(nn.Module):
    """
    Sparse Convolution based UNet for point-wise feature learning.
    Reference Paper: https://arxiv.org/abs/1907.03670 (Shaoshuai Shi, et. al)
    From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network
    """
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )

        block = post_act_block

        # processor

        # first block
        # obtain the line and surface of the point clouds
        self.pro_conv1 = spconv.SparseSequential(
            block(input_channels, 16, 3, norm_fn=norm_fn, padding=1, conv_type='spconv', indice_key='pro_spconv1'),
            block(16, 16, 3, norm_fn=norm_fn, padding=1, conv_type='subm', indice_key='pro_subm1'),
        )

        self.pro_conv2 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, conv_type='subm', indice_key='pro_subm1'),
        )

        # Normal Vectors
        self.norm_vec1 = fast_normal_vector_gen
        self.extract_wb1 = Extract_WeightBiasVoxel

        self.norm_vec2 = fast_normal_vector_gen

        self.vfe = TinMeanVFE

        self.get_xyz = get_xyz

        self.block = TinBlock

        self.num_point_features = 16

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """


        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        x_pro1 = self.pro_conv1(x)
        x_pro2 = self.pro_conv2(x)

        x_pro = x_pro1 + x_pro2

        # voxel path
        out_voxel = self.vfe(x_pro)

        x_normVec1 = self.norm_vec1(out_voxel)
        x_normVec1 = self.pro_conv1(x_normVec1) # should be some conv with some kernel size (but haven't thought about)


        # normal vector path 1
        x_normVec2 = self.norm_vec2(x_pro)
        x_normVec2 = self.vfe(x_normVec2)

        x_norm = x_normVec1 + x_normVec2


        x_norm = F.normalize(x_norm, p=2, dim=None) # p=2 L2 Normalize
        x_norm = self.pro_conv1(x_norm)

        xy_norm, z_norm = self.get_xyz(x_norm)

        xy_norm = self.block(xy_norm)
        z_norm = self.block(z_norm)



        # batch_dict['norm_vec_feature'] = torch.ravel(x_norm)
        # batch_dict['voxel_features'] = torch.ravel(x_pro)

        # batch_dict['point_features'] = x_up1.features
        # point_coords = common_utils.get_voxel_centers(
        #     x_up1.indices[:, 1:], downsample_times=1, voxel_size=self.voxel_size,
        #     point_cloud_range=self.point_cloud_range
        # )
        # batch_dict['point_coords'] = torch.cat((x_up1.indices[:, 0:1].float(), point_coords), dim=1)
        return batch_dict
