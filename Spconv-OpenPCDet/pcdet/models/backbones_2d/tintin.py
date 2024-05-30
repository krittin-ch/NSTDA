import torch
import torch.nn as nn
import torchvision.models as models

class Tin2D(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        # Initialize ResNet50 model without pre-trained weights
        self.constructor = models.resnet50(pretrained=False)
        # Modify the first convolutional layer to accept the input channels specified
        self.constructor.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Remove the fully connected layer as we only want to use the feature extractor
        self.constructor = nn.Sequential(*list(self.constructor.children())[:-1])
        
    def forward(self, batch_dict):
        image = batch_dict['image']
        batch_dict['image_feature'] = self.constructor(image)
        
        return batch_dict
