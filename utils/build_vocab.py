import re
import json
import random
from collections import Counter
import pandas as pd

def clean_txt(txt):
    txt = txt.strip().lower()
    txt = re.sub("'", "", txt)
    txt = re.sub("[^a-z0-9.,]+", " ", txt)
    txt = " ".join(txt.split())
    return txt

def build_word_vocab(df_path, text_column="caption", min_freq=2, max_vocab=None):
    df = pd.read_csv(df_path)
    all_words = []
    for text in df[text_column]:
        words = clean_txt(text).split()
        all_words.extend(words)
    random.shuffle(all_words)

    word_counts = Counter(all_words)

    words = [w for w, c in word_counts.items() if c >= min_freq]

    if max_vocab:
        words = [w for w, _ in word_counts.most_common(max_vocab)]

    vocab = {"<pad>": 0, "<unk>": 1, "<start>": 2, "<end>": 3}

    for idx, word in enumerate(words, start=4):
        vocab[word] = idx

    return vocab
