import torch.nn as nn
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

class ImageEncode(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, model_name='resnet34', pretrained=True, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        
        self.resnet_model = self.get_resnet_model(model_name, pretrained)


    def get_resnet_model(self, model_name='resnet34', pretrained=True):
        if model_name == 'resnet34':
            model = models.resnet34(pretrained=pretrained)
        elif model_name == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
        elif model_name == 'resnet101':
            model = models.resnet101(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

    def preprocess_image(self, image_path):
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        image = Image.open(image_path)
        image = preprocess(image)
        image = image.unsqueeze(0)  # Add batch dimension
        return image
    
    def extract_features(self, image_tensor):
        with torch.no_grad():
            features = self.resnet_model(image_tensor)
        return features


    def forward(self, batch_dict):
        image = batch_dict['images']

        image_tensor = self.preprocess_image(image)
        image_feature = self.extract_features(image_tensor)

        batch_dict['image_feature'] = image_feature
        return batch_dict