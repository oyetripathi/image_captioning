import math
from collections import Counter

def ngrams(txt, n):
    sequence = txt.split()
    return Counter([tuple(sequence[i:i+n]) for i in range(0, len(sequence)-n+1)])

def bleu_score(refs, pred, n ):
    precisions = []
    for i in range(1, n+1):
        pred_ngrams = ngrams(pred, i)
        max_ref_cnt = Counter()
        for ref in refs:
            ref_ngrams = ngrams(ref, i)
            for gram in ref_ngrams:
                if ref_ngrams[gram] > max_ref_cnt[gram]:
                    max_ref_cnt[gram] = ref_ngrams[gram]
        overlap = {gram: min(count, max_ref_cnt[gram]) for gram, count in pred_ngrams.items() if gram in max_ref_cnt}
        num = sum(overlap.values())
        denom = sum(pred_ngrams.values())
        precisions.append(num / denom if  num>0 else 1e-6)

    ref_lens = [len(r.split()) for r in refs]
    pred_len = len(pred.split())
    ref_len = min(ref_lens, key = lambda x: abs(x - pred_len))

    brevity_penalty = math.exp(1 - min(1, pred_len / ref_len))

    score = brevity_penalty * math.exp(sum([math.log(p) for p in precisions])/n)
    return score

def compute_bleu(all_refs, all_preds, n=4):
    total = 0.0
    for refs, pred in zip(all_refs, all_preds):
        total += bleu_score(refs, pred, n)
    return total / len(all_preds)