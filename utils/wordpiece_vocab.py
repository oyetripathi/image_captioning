import re
from collections import Counter, defaultdict

def preprocess_sentence(txt):
    txt = txt.strip().lower()
    txt = re.sub(r"[^a-z ]", "", txt)
    txt = re.sub(r"\s+", " ", txt)
    return txt.split()

def get_word_freq(corpus):
    word_freq = Counter()
    for sentence in corpus:
        sentence = preprocess_sentence(sentence)
        for word in sentence:
            word_freq[word] += 1
    return word_freq

def get_splits(word_freq):
    return {
        word: [c if i==0 else f"##{c}" for i,c in enumerate(word)] for word in word_freq
    }

def compute_pair_scores(word_freq, splits, top_k=3):
    letter_freq = Counter()
    pair_freq = Counter()
    for word, freq in word_freq.items():
        split = splits[word]
        if len(split) == 1:
            letter_freq[split[0]] += freq
            continue
        for i in range(len(split)-1):
            pair = (split[i], split[i+1])
            pair_freq[pair] += freq
            letter_freq[split[i]] += freq
        letter_freq[split[-1]] += freq
    scores = {pair: pair_freq[pair] / (letter_freq[pair[0]] * letter_freq[pair[1]]) for pair in pair_freq}
    sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k])
    return sorted_scores

def merge_one_pair(a, b, word_freq, splits):
    for word in word_freq:
        split = splits[word]
        if len(split) == 1:
            continue
        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i+1] == b:
                split = split[:i] + [a + b.replace("##", "")] + split[i+2:]
            else:
                i += 1
        splits[word] = split
    return splits

def merge_pairs(pair_scores, word_freq, splits):
    used_tokens = set()
    actual_merges = []
    for (t1, t2) in pair_scores.keys():
        if (t1 in used_tokens) or (t2 in used_tokens):
            continue
        used_tokens.add(t1)
        used_tokens.add(t2)
        actual_merges.append((t1, t2))
        splits = merge_one_pair(t1, t2, word_freq, splits)
    return splits, actual_merges

def build_word_vocab(corpus, vocab_size, top_k=3):
    vocab = set(['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', ",", '.', "<pad>", "<unk>", "<start>", "<end>"])
    word_freq = get_word_freq(corpus)
    print(f"Initial unique words: {len(word_freq)}")
    
    for word in word_freq:
        if not (word[0] in vocab):
            vocab.add(word[0])
        for letter in word[1:]:
            if not (f"##{letter}" in vocab):
                vocab.add(f"##{letter}")

    splits = get_splits(word_freq)

    while len(vocab) < vocab_size:
        pair_scores = compute_pair_scores(word_freq, splits, top_k)
        splits, merges = merge_pairs(pair_scores, word_freq, splits)
        for (a, b) in merges:
            vocab.add(a + b.replace("##", ""))
        print(f"vocab size: {len(vocab)}, Merged {len(merges)} pairs")
    
    return {word:i for i,word in enumerate(vocab)}