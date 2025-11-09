import torch
import torch.nn as nn
import torchvision


class ResNet101Encoder(nn.Module):
    def __init__(self, fine_tune=False):
        super().__init__()
        resnet_model = torchvision.models.resnet101(weights="IMAGENET1K_V2")

        self.base_model = nn.Sequential(*list(resnet_model.children())[:-2])
        self.transforms = torchvision.models.ResNet101_Weights.IMAGENET1K_V2.transforms()
        self.fine_tune = fine_tune
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.layer_norm = nn.LayerNorm(2048)

        for p in self.base_model.parameters():
            p.requires_grad = False
        
        if self.fine_tune:
            for c in list(self.base_model.children())[-2:]:
                for p in c.parameters():
                    p.requires_grad = True
        
    def forward(self, images):
        batch_size = images.shape[0]
        images = (images - self.mean) / self.std
        features = self.base_model(images)
        features = features.view(batch_size, 2048, -1).permute(0, 2, 1)
        features = self.layer_norm(features)
        return features

class ResNet50Encoder(nn.Module):
    def __init__(self, fine_tune=False):
        super().__init__()
        resnet_model = torchvision.models.resnet50(weights="IMAGENET1K_V2")

        self.base_model = nn.Sequential(*list(resnet_model.children())[:-2])
        self.transforms = torchvision.models.ResNet50_Weights.IMAGENET1K_V2.transforms()
        self.fine_tune = fine_tune
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.layer_norm = nn.LayerNorm(2048)

        for p in self.base_model.parameters():
            p.requires_grad = False
        
        if self.fine_tune:
            for c in list(self.base_model.children())[-2:]:
                for p in c.parameters():
                    p.requires_grad = True
        
    def forward(self, images):
        batch_size = images.shape[0]
        images = (images - self.mean) / self.std
        features = self.base_model(images)
        features = features.view(batch_size, 2048, -1).permute(0, 2, 1)
        features = self.layer_norm(features)
        return features
