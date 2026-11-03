"""Retrieval evaluation on the hand-labelled gold set.

Metrics (article level): Hit@1, Hit@3, Hit@5 (a gold article in top-k), MRR@10,
and Precision@5 / Recall@5 as required by the assignment.

    python -m eval.eval_retrieval                 # full system ablation
    EMBED_MODEL=intfloat/multilingual-e5-base python -m eval.eval_retrieval
"""
from __future__ import annotations

import argparse
import json
import time

from rag import config
from rag.retriever import HybridRetriever

from .gold_questions import GOLD


def evaluate(retriever: HybridRetriever, rewrites: dict | None = None, verbose: bool = False, **flags) -> dict:
    hit = {1: 0, 3: 0, 5: 0}
    mrr = prec = rec = 0.0
    misses, by_lang = [], {}
    t0 = time.time()
    for lang, q, gold, _ in GOLD:
        hits = retriever.search(q, k=10, rewritten=(rewrites or {}).get(q), **flags)
        ids = [h.article.article_id for h in hits]
        rank = next((i + 1 for i, a in enumerate(ids) if a in gold), None)
        for k in hit:
            hit[k] += int(rank is not None and rank <= k)
        mrr += 1.0 / rank if rank else 0.0
        top5 = set(ids[:5])
        prec += len(top5 & set(gold)) / 5
        rec += len(top5 & set(gold)) / len(gold)
        b = by_lang.setdefault(lang, [0, 0])
        b[0] += int(rank is not None and rank <= 5)
        b[1] += 1
        if rank is None or rank > 5:
            misses.append((q, gold, ids[:5]))
    n = len(GOLD)
    res = {f"hit@{k}": round(v / n, 3) for k, v in hit.items()}
    res.update({"mrr": round(mrr / n, 3), "p@5": round(prec / n, 3), "r@5": round(rec / n, 3),
                "by_lang": {k: f"{a}/{b}" for k, (a, b) in by_lang.items()},
                "sec_per_q": round((time.time() - t0) / n, 3)})
    if verbose:
        for q, g, got in misses:
            print(f"  MISS {q}  gold={g}  got={got}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rewrites", help="json file {question: rewritten_query}")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--full-only", action="store_true", help="only the complete system (needs --rewrites)")
    args = ap.parse_args()
    rewrites = json.load(open(args.rewrites, encoding="utf-8")) if args.rewrites else None

    r = HybridRetriever()
    print(f"embed={config.EMBED_MODEL} rerank={config.RERANK_MODEL if r.reranker else None} chunks={len(r.chunks)}")
    configs = {
        "bm25": dict(use_dense=False, use_rerank=False),
        "dense": dict(use_bm25=False, use_rerank=False),
        "hybrid(rrf)": dict(use_rerank=False),
    }
    if r.reranker:
        configs["hybrid+rerank"] = {}
    for name, flags in ({} if args.full_only else configs).items():
        print(f"{name:15s}", evaluate(r, None, verbose=args.verbose and name == list(configs)[-1], **flags))
    if rewrites:
        print(f"{'+rewrite':15s}", evaluate(r, rewrites, verbose=args.verbose))


if __name__ == "__main__":
    main()
