import numpy as np
from collections import Counter, defaultdict

def ngrams(txt, n):
    seq = txt.split()
    return [tuple(seq[i:i+n]) for i in range(0, len(seq)-n+1)]

def compute_idf(all_refs, max_n):
    idf = [{}] * max_n
    N = len(all_refs)

    for n in range(1, max_n+1):
        df = defaultdict(int)
        for ref in all_refs:
            unique_ngrams = set()
            for r in ref:
                unique_ngrams.update(ngrams(r, n))
            for ng in unique_ngrams:
                df[ng] += 1
        idf[n-1] = {ng: np.log((N+1)/(cnt+1)) for ng, cnt in df.items()}
    return idf

def compute_tf(ng_list):
    tf_counts = Counter(ng_list)
    total = sum(tf_counts.values())
    return {ng: cnt/total for ng, cnt in tf_counts.items()}

def compute_ref_vectors(all_refs, idfs, max_n):
    ref_vecs = []
    for ref in all_refs:
        ref_vec = []
        for r in ref:
            ngram_vec = {}
            for n in range(1, max_n+1):
                tf = compute_tf(ngrams(r, n))
                for ng, val in tf.items():
                    ngram_vec[(n, ng)] = val * idfs[n-1].get(ng, 0)
            ref_vec.append(ngram_vec)
        ref_vecs.append(ref_vec)
    return ref_vecs

def cosine_sparse(vec1, vec2):
    if not vec1 or not vec2:
        return 0.0
    common_keys = set(vec1.keys()) & set(vec2.keys())
    num = sum(vec1[k]*vec2[k] for k in common_keys)
    norm1 = np.sqrt(sum(v**2 for v in vec1.values()))
    norm2 = np.sqrt(sum(v**2 for v in vec2.values()))
    return num / (norm1*norm2 + 1e-9)

def cider_score(pred, ref_vecs, idfs, max_n):
    pred_vec = {}
    for n in range(1, max_n+1):
        tf = compute_tf(ngrams(pred, n))
        for ng, val in tf.items():
            pred_vec[(n, ng)] = val * idfs[n-1].get(ng, 0)
    scores = [cosine_sparse(pred_vec, ref_vec) for ref_vec in ref_vecs]
    return np.mean(scores)

def compute_cider(all_preds, all_refs, max_n=5):
    idfs = compute_idf(all_refs, max_n)
    ref_vecs = compute_ref_vectors(all_refs, idfs, max_n)
    scores = [cider_score(pred, r_vec, idfs, max_n) for pred, r_vec in zip(all_preds, ref_vecs)]
    return 10*float(np.mean(scores))