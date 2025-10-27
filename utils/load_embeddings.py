import numpy as np
import torch

def load_pretrained_embeddings(embedding_path, vocab, embed_dim=100):
    vocab_size = len(vocab)
    embedding_matrix = np.random.normal(loc = 0, scale = 1, size = (vocab_size, embed_dim))
    pad_token_id = vocab.get("<pad>")

    embedding_matrix[pad_token_id] = np.zeros((embed_dim), dtype=np.float32)

    embedding_idx = {}
    with open(embedding_path, mode="r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype=np.float32)
            embedding_idx[word] = vector
    found = 0
    for word, idx in vocab.items():
        if word in ["<pad", "<unk", "<start>", "<end>"]:
            continue
        if word in embedding_idx:
            embedding_matrix[idx] = embedding_idx[word]
            found += 1
    print(f"Found {found}/{vocab_size} words in the pretrained embeddings.")
    return torch.tensor(embedding_matrix, dtype=torch.float32)