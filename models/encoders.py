import torch
import math
import torch.nn as nn
import torchvision
from PIL import ImageOps
from models.attention import MultiHeadAttention, TransformerFeedForward

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

class EncoderTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.self_attn = MultiHeadAttention(num_heads, d_model)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(d_model)

        self.ff = TransformerFeedForward(d_model, ff_dim=4*d_model)
        self.layer_norm_ff = nn.LayerNorm(d_model)
    def forward(self, x):
        out1, __ = self.self_attn(x, x, x)
        out1 = self.layer_norm(self.dropout(out1) + x)

        out2 = self.ff(out1)
        out2 = self.layer_norm_ff(self.dropout(out2) + out1)
        return out2

class EncoderTransformer(nn.Module):
    def __init__(self, patch_size, d_model, num_heads, num_layers):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        self.transforms = torchvision.transforms.Compose([
            ImageOps.exif_transpose,
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.patch_embedding = nn.Linear(3*patch_size*patch_size, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(self.d_model)

        self.layers = nn.ModuleList([
            EncoderTransformerLayer(d_model, num_heads) for _ in range(num_layers)
        ])
    
    def get_positional_encoding_2d(self, ndim, embed_dim, device, include_cls_token=False):
        assert embed_dim % 4 == 0

        half_dim = embed_dim // 2
        quarter_dim = embed_dim // 4

        pe_h = torch.zeros(1, ndim * ndim, half_dim, device=device)
        num_h = torch.arange(ndim, device=device).repeat_interleave(ndim).unsqueeze(-1)
        div_term_h = torch.exp(
            torch.arange(0, quarter_dim, device=device) * -(math.log(10000.0) / quarter_dim)
        )
        pe_h[:, :, 0::2] = torch.sin(num_h / div_term_h)
        pe_h[:, :, 1::2] = torch.cos(num_h / div_term_h)

        pe_w = torch.zeros(1, ndim * ndim, half_dim, device=device)
        num_w = torch.arange(ndim, device=device).repeat(ndim).unsqueeze(-1)
        div_term_w = torch.exp(
            torch.arange(0, quarter_dim, device=device) * -(math.log(10000.0) / quarter_dim)
        )
        pe_w[:, :, 0::2] = torch.sin(num_w / div_term_w)
        pe_w[:, :, 1::2] = torch.cos(num_w / div_term_w)
        
        pe = torch.cat((pe_h, pe_w), dim=-1)

        if include_cls_token:
            cls_pe = torch.zeros(1, 1, embed_dim, device=device)
            pe = torch.cat([cls_pe, pe], dim=1)
        return pe

    
    def forward(self, x):
        assert x.shape[2] % self.patch_size == 0 and x.shape[3] % self.patch_size == 0, "Image dimensions must be divisible by patch size"
        assert x.shape[2] % self.patch_size == x.shape[3] % self.patch_size, "Image height and width must be equal"
        num_patches_dim = x.shape[2] // self.patch_size
        batch_size = x.shape[0]
        device = x.device
        x = (
                x
                .unfold(2, self.patch_size, self.patch_size)
                .unfold(3, self.patch_size, self.patch_size)
                .permute(0, 2, 3, 1, 4, 5)
                .reshape(batch_size, num_patches_dim*num_patches_dim, 3*self.patch_size*self.patch_size)
        )
        x = self.patch_embedding(x)
        x = torch.cat([self.cls_token.repeat(batch_size, 1, 1), x], dim=1)
        pos_enc = self.get_positional_encoding_2d(num_patches_dim, self.d_model, device=device, include_cls_token=True)
        x = self.dropout(x + pos_enc)

        for layer in self.layers:
            x = layer(x)

        return x