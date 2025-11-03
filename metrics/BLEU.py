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
    nums = [[] for _ in range(n)]
    denoms = [[] for _ in range(n)]
    total_ref_len = 0
    total_pred_len = 0
    for refs, pred in zip(all_refs, all_preds):
        for i in range(1, n+1):
            pred_ngrams = ngrams(pred, i)
            max_ref_cnt = Counter()
            for ref in refs:
                ref_ngrams = ngrams(ref, i)
                for gram in ref_ngrams:
                    if ref_ngrams[gram] > max_ref_cnt[gram]:
                        max_ref_cnt[gram] = ref_ngrams[gram]
            overlap = {gram: min(count, max_ref_cnt[gram]) for gram, count in pred_ngrams.items() if gram in max_ref_cnt}
            nums[i-1].append(sum(overlap.values()))
            denoms[i-1].append(sum(pred_ngrams.values()))

        pred_len = len(pred.split())
        ref_lens = [len(r.split()) for r in refs]
        total_ref_len += min(ref_lens, key = lambda x: abs(x - pred_len))
        total_pred_len += pred_len

    precisions = []
    for i in range(n):
        if (sum(denoms[i])) == 0:
            precisions.append(1e-6)
        else:
            p = sum(nums[i]) / sum(denoms[i])
            precisions.append(max(p, 1e-6))

    brevity_penalty = math.exp(1 - total_ref_len / total_pred_len) if total_pred_len < total_ref_len else 1.0

    score = brevity_penalty * math.exp(sum([math.log(p) for p in precisions])/n)
    return score