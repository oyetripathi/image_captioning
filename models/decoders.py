import torch
import torch.nn as nn


class DecoderLSTM(nn.Module):
    def __init__(self, feature_dim, vocab_size, embed_dim, pretrained_embeddings=None):
        super().__init__()
        if pretrained_embeddings:
            self.embed = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        else:
            self.embed = nn.Embedding(vocab_size, embed_dim)
        
        self.init_h = nn.Linear(feature_dim, 512)
        self.lstm = nn.LSTM(embed_dim, 512, num_layers=1, batch_first=True)
        self.fc = nn.Linear(512, vocab_size)
    
    def forward(self, captions, img_encoding):
        embeddings = self.embed(captions)
        img_encoding = img_encoding.mean(dim=[-2, -1]).unsqueeze(0)
        img_encoding = self.init_h(img_encoding)
        lstm_out, _ = self.lstm(embeddings, (img_encoding, img_encoding))
        outputs = self.fc(lstm_out)
        return outputs