"""Score the ORIGINAL devoir-2 retriever (export_final.csv + MiniLM + 300-char chunks) on the gold set.

The old CSV rows are PDF pages, not articles, so a retrieved row counts as a hit when its page
range overlaps the page range of a gold article (lenient for the old system).
"""
import json
import re

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .gold_questions import GOLD


def norm(t):
    t = re.sub(r"[ؗ-ًؚ-ٟ]", "", str(t)); t = re.sub(r"[أإآ]", "ا", t)
    return re.sub(r"\s+", " ", t).strip()


df = pd.read_csv("export_final.csv", encoding="utf-8-sig").fillna("")
recs = []
for _, r in df.iterrows():
    text = norm(f"المادة {r['article_id']} | {r['type_infraction']} | {r['infraction_desc']} | {r['sanction_desc']} | {r['mots_cles']}")
    for s in range(0, len(text), 250):
        recs.append((text[s:s + 300], int(r["source_page"]), int(r["source_page_end"] or r["source_page"])))
m = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
emb = m.encode([c for c, *_ in recs], normalize_embeddings=True, batch_size=64)
idx = faiss.IndexFlatIP(emb.shape[1]); idx.add(np.asarray(emb, dtype="float32"))
pages = {json.loads(l)["article_id"]: (json.loads(l)["page_start"], json.loads(l)["page_end"]) for l in open("data/articles.jsonl")}
hit = 0
for _, q, gold, _ in GOLD:
    _, I = idx.search(np.asarray(m.encode([q], normalize_embeddings=True), dtype="float32"), 5)
    got = [recs[i][1:] for i in I[0]]
    ok = any(ps <= ge and gs <= pe for (ps, pe) in got for g in gold for (gs, ge) in [pages[g]])
    hit += ok
print(f"OLD system page-level Hit@5 = {hit}/{len(GOLD)} = {hit/len(GOLD):.3f}")
