import torch
import torch.nn as nn
from models.attention import SoftAttention, MultiHeadAttention


class DecoderLSTM(nn.Module):
    def __init__(self, feature_dim, vocab_size, embed_dim, pretrained_embeddings=None):
        super().__init__()
        if pretrained_embeddings is not None:
            self.embed = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        else:
            self.embed = nn.Embedding(vocab_size, embed_dim)
        
        self.init_h = nn.Linear(feature_dim, 512)
        self.lstm1 = nn.LSTM(embed_dim, 512, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(512, 128, num_layers=2, batch_first=True)
        self.fc = nn.Linear(128, vocab_size)
    
    def forward(self, captions, img_encoding):
        embeddings = self.embed(captions)
        img_encoding = img_encoding.mean(dim=[-2, -1]).unsqueeze(0)
        img_encoding = self.init_h(img_encoding)
        lstm_out, _ = self.lstm1(embeddings, (img_encoding, img_encoding))
        lstm_out, _ = self.lstm2(lstm_out)
        outputs = self.fc(lstm_out)
        return outputs

class DecoderLSTMWithSoftAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, feature_dim, hidden_dim, attention_dim, pretrained_embeddings=None):
        super().__init__()
        self.attention_dim = attention_dim
        self.vocab_size = vocab_size
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        if pretrained_embeddings is not None:
            self.embed = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        else:
            self.embed = nn.Embedding(vocab_size, embed_dim)
        self.attention = SoftAttention(feature_dim, hidden_dim, attention_dim)
        self.lstm = nn.LSTMCell((embed_dim + feature_dim), hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
        self.relu = nn.ReLU()
    
    def decode_step(self, img_encodings, captions_step, h, c):
        embeddings = self.embed(captions_step)
        weighted_img_encoding, attn_wt = self.attention(img_encodings, h)
        h,c = self.lstm(torch.cat([embeddings, weighted_img_encoding], dim=1), (h,c))
        out = self.fc1(h)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        return out, h,c, attn_wt
    
    def forward(self, captions, img_encodings):
        batch_size = captions.shape[0]
        seq_len = captions.shape[1]
        device = captions.device
        img_encodings = img_encodings.view(batch_size, self.feature_dim, -1).permute((0, 2, 1))
        h,c = self.init_hidden_state(batch_size, self.hidden_dim, device)
        outputs = torch.zeros((batch_size, seq_len, self.vocab_size)).to(device)
        attn_wts = torch.zeros((batch_size, seq_len, img_encodings.shape[1])).to(device)
        for t in range(seq_len):
            out, h,c, attn_wt = self.decode_step(img_encodings, captions[:, t], h, c)
            outputs[:, t, :] = out
            attn_wts[:, t, :] = attn_wt
        return outputs, attn_wts
    
    def init_hidden_state(self, batch_size, hidden_dim, device):
        h = torch.zeros((batch_size, hidden_dim)).to(device)
        c = torch.zeros((batch_size, hidden_dim)).to(device)
        return h,c

class TransformerFeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        return out

class DecoderTransformerLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim, feature_dim, num_heads, pretrained_embeddings=None):
        super().__init__()
        self.self_attn = MultiHeadAttention(num_heads, embed_dim)
        self.layer_norm1 = nn.LayerNorm(embed_dim)

        self.feature_dim = feature_dim
        self.encoder_proj = nn.Linear(feature_dim, embed_dim)
        self.cross_attn = MultiHeadAttention(num_heads, embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

        self.ff = TransformerFeedForward(embed_dim, embed_dim)
        self.layer_norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, embeddings, img_encodings, mask=None):
        batch_size = embeddings.shape[0]
        seq_len = embeddings.shape[1]

        img_encodings = img_encodings.view(batch_size, self.feature_dim, -1).permute((0, 2, 1))
        img_encodings = self.encoder_proj(img_encodings)

        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=embeddings.device))
        out1, _ = self.self_attn(embeddings, embeddings, embeddings, causal_mask)
        out1 = self.layer_norm1(self.dropout(out1) + embeddings)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
        out2, __ = self.cross_attn(out1, img_encodings, img_encodings, mask)
        out2 = self.layer_norm2(self.dropout(out2) + out1)

        out3 = self.ff(out2)

        out = self.layer_norm3(self.dropout(out3) + out2)
        return out


class DecoderTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, feature_dim, num_heads, num_layers, pretrained_embeddings=None):
        super().__init__()
        self.embed_dim = embed_dim
        if pretrained_embeddings is not None:
            self.embed = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        else:
            self.embed = nn.Embedding(vocab_size, self.embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

        self.layers = nn.ModuleList([
            DecoderTransformerLayer(
                vocab_size,
                embed_dim,
                feature_dim,
                num_heads,
                pretrained_embeddings
            ) for _ in range(num_layers)
        ])
        self.linear = nn.Linear(embed_dim, vocab_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
    
    def get_positional_encoding(self, embeddings, device):
        batch_size = embeddings.shape[0]
        seq_len = embeddings.shape[1]
        pe = torch.zeros((seq_len, self.embed_dim), dtype=torch.float32).to(device)
        phase_num = torch.arange(0, seq_len, dtype=torch.float32)
        phase_denom = (10**4) ** (-1*torch.arange(0, self.embed_dim, 2)/(self.embed_dim))
        phase = phase_num.unsqueeze(-1) * phase_denom.unsqueeze(0)
        pe[:, ::2] = torch.sin(phase)
        pe[:, 1::2] = torch.cos(phase)[:, :self.embed_dim//2]
        pe = pe.unsqueeze(0).repeat(batch_size, 1, 1)
        return pe
        
    def forward(self, captions, img_encodings, mask=None):
        embeddings = self.embed(captions)** (self.embed_dim ** 0.5)
        pos_enc = self.get_positional_encoding(embeddings, embeddings.device)
        
        out = self.dropout(self.layer_norm(embeddings + pos_enc))

        for layer in self.layers:
            out = layer(out, img_encodings, mask)

        out = self.linear(out)
        out = self.relu(out)
        return out