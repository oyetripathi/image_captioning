import math
import torch
import torch.nn as nn


class SoftAttention(nn.Module):
    def __init__(self, feature_dim, hidden_dim, attention_dim):
        super().__init__()
        self.encoder_attn = nn.Linear(feature_dim, attention_dim)
        self.decoder_attn = nn.Linear(hidden_dim, attention_dim)
        self.encoder_wt = nn.Parameter(torch.randn(attention_dim))
        self.decoder_wt = nn.Parameter(torch.randn(attention_dim))
        self.bias = nn.Parameter(torch.randn(attention_dim))
        self.relu = nn.ReLU()
        self.full_attn = nn.Linear(attention_dim, 1)

    def forward(self, img_encoding, decoder_hs):
        att1 = self.encoder_attn(img_encoding)
        att2 = self.decoder_attn(decoder_hs).unsqueeze(1)

        att = self.relu(
            self.encoder_wt * att1 + self.decoder_wt * att2 + self.bias
        )

        e = self.full_attn(att).squeeze(-1)
        attn_wts = torch.softmax(e, dim=1)
        weighted_img_encoding = (img_encoding * attn_wts.unsqueeze(-1)).sum(dim=1).squeeze(1)
        return weighted_img_encoding, attn_wts


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, query, key, value, mask=None):
        d_k = query.shape[-1]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores.masked_fill_(mask == 0, float('-inf'))
        attn_weights = self.dropout(self.softmax(scores))
        output = torch.matmul(attn_weights, value)
        return output, attn_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, model_dim):
        super().__init__()
        assert (model_dim % num_heads == 0)
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.head_dim = model_dim // num_heads
        self.query_linear = nn.Linear(model_dim, model_dim)
        self.key_linear = nn.Linear(model_dim, model_dim)
        self.value_linear = nn.Linear(model_dim, model_dim)
        self.out_linear = nn.Linear(model_dim, model_dim)
        self.scaled_dot_product_attention = ScaledDotProductAttention()
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        def transform(x, linear):
            x = linear(x)
            x = x.view(x.shape[0], x.shape[1], self.num_heads, self.head_dim)
            x = x.transpose(1, 2)
            return x
        
        query = transform(query, self.query_linear)
        key = transform(key, self.key_linear)
        value = transform(value, self.value_linear)

        attn_output, attn_weights = self.scaled_dot_product_attention(query, key, value, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.model_dim)
        return self.out_linear(attn_output), attn_weights

class TransformerFeedForward(nn.Module):
    def __init__(self, d_model, ff_dim):
        super().__init__()
        self.fc1 = nn.Linear(d_model, ff_dim)
        self.fc2 = nn.Linear(ff_dim, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
