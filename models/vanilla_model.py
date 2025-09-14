import torch
import torch.nn as nn


class VanillaCaptioningModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(VanillaCaptioningModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images, captions):
        img_encoding = self.encoder(images)
        outputs = self.decoder(captions, img_encoding)
        return outputs