import torch
import torch.nn as nn
import torchvision


class ResNet101Encoder(nn.Module):
    def __init__(self, encoded_img_dim = 8,fine_tune=False):
        super().__init__()
        resnet_model = torchvision.models.resnet101(weights="IMAGENET1K_V2")

        self.base_model = nn.Sequential(*list(resnet_model.children())[:-2])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_img_dim, encoded_img_dim))
        self.fine_tune = fine_tune

        for p in self.base_model.parameters():
            p.requires_grad = False
        
        if self.fine_tune:
            for c in list(self.base_model.children())[-2:]:
                for p in c.parameters():
                    p.requires_grad = True
        
    def forward(self, images):
        features = self.base_model(images)
        out = self.adaptive_pool(features)
        return out