import json
from collections import Counter
import pandas as pd

def build_word_vocab(df, text_column="caption", min_freq=2, max_vocab=None, save_path="word_vocab.json"):
    all_words = []
    for text in df[text_column]:
        words = str(text).lower().strip().split()
        all_words.extend(words)

    word_counts = Counter(all_words)

    words = [w for w, c in word_counts.items() if c >= min_freq]

    if max_vocab:
        words = [w for w, _ in word_counts.most_common(max_vocab)]

    vocab = {"<pad>": 0, "<unk>": 1, "<start>": 2, "<end>": 3}

    for idx, word in enumerate(words, start=4):
        vocab[word] = idx

    with open(save_path, "w") as f:
        json.dump(vocab, f, indent=4)

    print(f"Vocabulary saved to {save_path}. Total size: {len(vocab)}")
    return vocab
