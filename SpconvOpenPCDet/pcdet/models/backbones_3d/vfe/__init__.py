from .mean_vfe import MeanVFE
from .pillar_vfe import PillarVFE
from .image_vfe import ImageVFE
from .vfe_template import VFETemplate
from .tin_mean_vfe import TinMeanVFE
from .voxel_feature_encoding import VoxelFeatureEncoding

__all__ = {
    'VFETemplate': VFETemplate,
    'MeanVFE': MeanVFE,
    'PillarVFE': PillarVFE,
    'ImageVFE': ImageVFE,
    'TinMEANVFE': TinMeanVFE,
    'VoxelFeatureEncoding': VoxelFeatureEncoding
}
