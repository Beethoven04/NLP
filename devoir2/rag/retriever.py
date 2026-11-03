"""Hybrid retriever: BM25 (Arabic-normalized, stemmed) + dense (bge-m3) fused with RRF,
then re-scored by a multilingual cross-encoder and aggregated to article level.

Chunking keeps legal structure: every chunk carries its article number, its
book/part/chapter breadcrumb and, for long enumerations, the lead sentence that
states the penalty (e.g. "يعاقب بغرامة من 700 إلى 1400 درهم ... المخالفات التالية").
Without that lead, the chunk "4 - عدم احترام الضوء الأحمر" would not mention any fine.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

import numpy as np

from . import config
from .text import normalize, tokenize

# ─────────────────────────────────────────────────────────────────────────────
# Query expansion: everyday / Darija / French words -> vocabulary of law 52.05
# ─────────────────────────────────────────────────────────────────────────────
_SYNONYMS = [
    (r"قياد|اسوق|يسوق|تسوق|نسوق|كنسوق|سقت|سايق|conduire|conduite|volant", "سياقه السائق"),
    (r"اشاره حمرا|الاشاره الحمرا|ضو الحمر|الضو الحمر|ضوء احمر|الضوء الاحمر|feu rouge|الفو روج", "الضوء الاحمر التشوير الاحمر الوقوف المفروض"),
    (r"هاتف|تيليفون|تلفون|جوال|بورطابل|portable|t[ée]l[ée]phone", "الهاتف ممسوكا باليد"),
    (r"حزام الامان|سمطه|ceinture", "حزام السلامه"),
    (r"بيرمي|permis|رخصه القياده|رخصه السواقه", "رخصه السياقه"),
    (r"نقاط|نقطه|points?\b", "النقط رصيد خصم"),
    (r"سرعه|فيتاس|vitesse|exc[eè]s", "تجاوز السرعه القصوي المسموح بها"),
    (r"سكران|سكر|خمر|شراب|alcool|ivresse|كحول", "حاله سكر تاثير الكحول"),
    (r"مخدر|حشيش|drogue|stup[ée]fiant", "تاثير المواد المخدره"),
    (r"غرامه|خلاص|ذعيره|amende|مخالفه", "غرامه يعاقب مخالفه"),
    (r"عقوب|sanction|risque|peine|شنو كيوقع", "يعاقب غرامه الحبس"),
    (r"طوموبيل|طونوبيل|voiture|سياره|automobile", "مركبه"),
    (r"رادار|radar", "اجهزه تقنيه تعمل بطريقه اليه المراقبه"),
    (r"خوذه|كاسك|casque", "خوذه معتمده الدراجات الناريه"),
    (r"موطور|موطو|moto|دراجه ناريه", "الدراجات الناريه"),
    (r"حادث|accident|كسيدة", "حادثه سير"),
    (r"فوريان|fourri[eè]re|حجز السياره", "المحجز الايداع"),
    (r"اختبار الكحول|alcootest|النفخ", "رائز النفس الكحول"),
    (r"طعن|contest|r[ée]clamation|نعترض", "المنازعه في المخالفه شكايه"),
    (r"دفع|خلص|payer|paiement|اداء", "اداء الغرامه"),
    (r"شاحن|كاميو|camion|حافل|طوبيس|bus", "نقل البضائع النقل الجماعي للاشخاص"),
    (r"سن|عمر|[âa]ge", "سنه شمسيه كامله"),
    (r"فحص طبي|visite m[ée]dicale|طبيب", "فحص طبي"),
    (r"طاكسي|taxi", "سيارات الاجره"),
    (r"اوطوروت|autoroute|الطريق السيار", "الطريق السيار"),
    (r"راجل|piéton|pi[ée]ton|المشاه", "الراجلين"),
    (r"اولويه|priorit", "حق الاسبقيه"),
    (r"اتجاه ممنوع|sens interdit|ضد الاتجاه", "السير في اتجاه ممنوع"),
    (r"مراقبه تقنيه|فيزيت|visite technique|contr[ôo]le technique", "المراقبه التقنيه"),
    (r"تسجيل|كارط كريز|carte grise|نقل الملكيه|mutation", "شهاده التسجيل نقل الملكيه"),
    (r"مدرسه السياقه|اوطو ايكول|auto[- ]?[ée]cole", "مؤسسه لتعليم السياقه"),
]
_SYN_RE = [(re.compile(p, re.IGNORECASE), exp) for p, exp in _SYNONYMS]


def expand_query(query: str) -> str:
    q = normalize(query)
    extra = [exp for rx, exp in _SYN_RE if rx.search(q)]
    return q + (" " + " ".join(extra) if extra else "")


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Article:
    article_id: str
    text: str
    breadcrumb: str
    page_start: int
    page_end: int

    @property
    def title(self) -> str:
        return f"المادة {self.article_id}"


@dataclass
class Chunk:
    chunk_id: int
    article_id: str
    text: str          # text shown to the LLM / reranker
    index_text: str    # text indexed (adds breadcrumb)


@dataclass
class Hit:
    article: Article
    score: float
    chunks: list[str] = field(default_factory=list)
    signals: dict = field(default_factory=dict)


def load_articles(path=config.ARTICLES_PATH) -> list[Article]:
    arts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if len(r["text"].strip()) < 15:  # abrogated articles ("نسخت")
                continue
            arts.append(Article(r["article_id"], r["text"], r.get("breadcrumb", ""), r["page_start"], r["page_end"]))
    return arts


def build_chunks(articles: list[Article], max_words: int = config.CHUNK_WORDS) -> list[Chunk]:
    chunks: list[Chunk] = []
    by_id = {a.article_id: a for a in articles}
    for a in articles:
        header = f"المادة {a.article_id}"
        crumb = a.breadcrumb.split(" > ")[-1] if a.breadcrumb else ""
        lines = [l for l in a.text.split("\n") if l.strip()]
        lead = lines[0]
        windows: list[list[str]] = []
        cur: list[str] = []
        n = 0
        for ln in lines:
            w = len(ln.split())
            if cur and n + w > max_words:
                windows.append(cur)
                cur, n = cur[-1:], len(cur[-1].split())  # one-line overlap
            cur.append(ln)
            n += w
        if cur:
            windows.append(cur)
        refs = _reference_note(a, by_id) if len(a.text.split()) <= 80 else ""
        for wi, win in enumerate(windows):
            body = "\n".join(win)
            if wi > 0 and win[0] != lead:
                body = lead + "\n…\n" + body
            text = f"{header}\n{body}" + refs
            chunks.append(Chunk(len(chunks), a.article_id, text, f"{header} {crumb}\n{body}" + refs))
    return chunks


_REF = re.compile(r"(?:للمادة|المادة|للمواد|المواد)\s+(\d+)")


def _reference_note(a: Article, by_id: dict) -> str:
    """Short articles often only point elsewhere ("يعاقب ... عن كل مخالفة لقواعد السير المقررة
    تطبيقا للمادة 94"). Append the lead of the referenced article so the chunk says what it is about
    (art. 94 = pedestrians). Shown as an explicit cross-reference, never merged into the law text."""
    notes = []
    for num in dict.fromkeys(_REF.findall(a.text)):
        ref = by_id.get(num)
        if ref is None or num == a.article_id:
            continue
        lead = " ".join(ref.text.split()[:25])
        notes.append(f"[إحالة إلى المادة {num}: {lead}…]")
        if len(notes) == 2:
            break
    return ("\n" + "\n".join(notes)) if notes else ""


# ─────────────────────────────────────────────────────────────────────────────
# Retriever
# ─────────────────────────────────────────────────────────────────────────────
class HybridRetriever:
    """Two-stage retriever.

    Stage 1 (recall): BM25 + dense over *keys* = law chunks + doc2query questions.
      Each ranking is collapsed to article level (best key per article) and fused with RRF.
    Stage 2 (precision): the cross-encoder scores the real law chunks of the top
      RERANK_ARTICLES articles. Final article score = blend of reranker probability and
      normalized RRF, so a strong lexical match is not discarded by the reranker alone.
    """

    def __init__(self, articles: list[Article] | None = None, embed_model: str = config.EMBED_MODEL,
                 rerank_model: str | None = config.RERANK_MODEL if config.USE_RERANKER else None,
                 device: str | None = None, use_questions: bool = config.USE_DOC2QUERY):
        from rank_bm25 import BM25Okapi

        self.articles = articles or load_articles()
        self.by_id = {a.article_id: a for a in self.articles}
        self.chunks = build_chunks(self.articles)
        self.chunks_of: dict[str, list[int]] = {}
        for c in self.chunks:
            self.chunks_of.setdefault(c.article_id, []).append(c.chunk_id)
        # retrieval keys: (article_id, chunk_id or None, text)
        self.keys = [(c.article_id, c.chunk_id, c.index_text) for c in self.chunks]
        if use_questions and config.QUESTIONS_PATH.exists():
            qs = json.loads(config.QUESTIONS_PATH.read_text(encoding="utf-8"))
            for key, questions in qs.items():
                aid = key.split("#")[0]  # "184#3" = questions generated from chunk 3 of art. 184
                if aid in self.by_id:
                    self.keys += [(aid, None, q) for q in questions]
        self.key_article = [k[0] for k in self.keys]
        self.n_law = len(self.chunks)  # keys[:n_law] = law chunks, keys[n_law:] = doc2query questions
        self.bm25 = BM25Okapi([tokenize(k[2]) for k in self.keys[: self.n_law]], k1=1.2, b=0.6)
        self.device = device or _best_device()
        self.embed_model_name = embed_model
        self.embedder = self._load_embedder(embed_model)
        self.emb = self._encode([k[2] for k in self.keys])
        self.reranker = None
        if rerank_model:
            from sentence_transformers import CrossEncoder

            self.reranker = CrossEncoder(rerank_model, device=self.device, max_length=512)

    # ── dense ───────────────────────────────────────────────────────────────
    def _load_embedder(self, name):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(name, device=self.device)

    def _prefix(self, kind: str) -> str:
        return {"query": "query: ", "passage": "passage: "}[kind] if "e5" in self.embed_model_name else ""

    def _encode(self, raw_texts: list[str]) -> np.ndarray:
        texts = [self._prefix("passage") + t for t in raw_texts]
        key = hashlib.md5(("\n".join(texts) + self.embed_model_name).encode()).hexdigest()[:16]
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = config.CACHE_DIR / f"emb_{self.embed_model_name.replace('/', '_')}_{key}.npy"
        if path.exists():
            return np.load(path)
        emb = self.embedder.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=True)
        emb = np.asarray(emb, dtype="float32")
        np.save(path, emb)
        return emb

    def dense_scores(self, query: str) -> np.ndarray:
        q = self.embedder.encode([self._prefix("query") + query], normalize_embeddings=True)
        return (self.emb @ np.asarray(q, dtype="float32")[0]).astype("float32")

    # ── search ──────────────────────────────────────────────────────────────
    def _article_ranking(self, scores: np.ndarray, offset: int = 0) -> list[str]:
        seen, out = set(), []
        for ki in np.argsort(-scores)[: config.CANDIDATES * 6]:
            aid = self.key_article[ki + offset]
            if aid not in seen:
                seen.add(aid)
                out.append(aid)
                if len(out) >= config.CANDIDATES:
                    break
        return out

    def search(self, query: str, k: int = config.TOP_K, rewritten: str | None = None,
               use_bm25: bool = True, use_dense: bool = True, use_rerank: bool = True) -> list[Hit]:
        """Return the top-k articles for a query.

        `rewritten` is an optional LLM reformulation in legal Arabic; it is fused as an
        extra ranking (helps Darija / French questions) while the original is kept.
        """
        queries = [query] + ([rewritten] if rewritten and rewritten.strip() != query.strip() else [])
        rankings: list[list[str]] = []
        dense_max = 0.0
        chunk_signal = np.zeros(len(self.chunks), dtype="float32")  # to pick chunks inside an article
        for q in queries:
            if use_bm25:
                bm = self.bm25.get_scores(tokenize(expand_query(q)))
                rankings.append(self._article_ranking(bm))
                chunk_signal += _minmax(bm)
            if use_dense:
                ds = self.dense_scores(q)
                law, qs = ds[: self.n_law], ds[self.n_law:]
                dense_max = max(dense_max, float(law.max()))
                rankings.append(self._article_ranking(law))
                chunk_signal += _minmax(law)
                if len(qs):  # doc2query channel: separate ranking, so it cannot swamp the law text
                    rankings.append(self._article_ranking(qs, offset=self.n_law))
        rrf: dict[str, float] = {}
        for r in rankings:
            for rank, aid in enumerate(r):
                rrf[aid] = rrf.get(aid, 0.0) + 1.0 / (config.RRF_K + rank + 1)
        cand = sorted(rrf, key=rrf.get, reverse=True)
        best_rrf = rrf[cand[0]] if cand else 1.0

        def top_chunks(aid: str, n: int) -> list[int]:
            ids = self.chunks_of[aid]
            return sorted(ids, key=lambda ci: -chunk_signal[ci])[:n]

        if self.reranker is None or not use_rerank:
            hits = [Hit(self.by_id[a], rrf[a] / best_rrf, [self.chunks[ci].text for ci in top_chunks(a, 2)],
                        {"dense_max": dense_max, "rerank_max": None}) for a in cand[:k]]
            return hits

        # Rerank the RRF head plus each channel's own top hits: an article that only one
        # channel finds (e.g. via a doc2query paraphrase) still reaches the cross-encoder.
        pool = list(dict.fromkeys(cand[: config.RERANK_ARTICLES] +
                                  [a for r in rankings for a in r[: config.PER_CHANNEL_TOP]]))
        cand = pool
        pairs, owners = [], []
        rq = queries[-1]  # legal-Arabic reformulation when available
        for aid in cand:
            for ci in top_chunks(aid, config.CHUNKS_PER_ARTICLE):
                pairs.append((rq, self.chunks[ci].text))
                owners.append((aid, ci))
        probs = self.reranker.predict(pairs, batch_size=32, show_progress_bar=False)
        best: dict[str, list[tuple[float, int]]] = {}
        for (aid, ci), p in zip(owners, probs):
            best.setdefault(aid, []).append((float(p), ci))
        hits = []
        for aid, lst in best.items():
            lst.sort(reverse=True)
            rprob = lst[0][0]
            score = config.RERANK_WEIGHT * rprob + (1 - config.RERANK_WEIGHT) * rrf[aid] / best_rrf
            hits.append(Hit(self.by_id[aid], score, [self.chunks[ci].text for _, ci in lst[:2]],
                            {"dense_max": dense_max, "rerank_max": rprob}))
        hits.sort(key=lambda h: h.score, reverse=True)
        return self._diversify(hits, k, best)

    def _diversify(self, hits: list[Hit], k: int, best: dict) -> list[Hit]:
        """MMR-style selection: near-duplicate articles (e.g. the four accident articles
        166-1/167/169/172 that share the same aggravating-circumstances list) should not
        take all top-k slots. Penalty grows with cosine similarity above MMR_SIM."""
        if len(hits) <= 1 or config.MMR_PENALTY <= 0:
            return hits[:k]
        vec = {h.article.article_id: self.emb[sorted(best[h.article.article_id], reverse=True)[0][1]] for h in hits}
        chosen: list[Hit] = []
        pool = list(hits)
        while pool and len(chosen) < k:
            def adj(h):
                if not chosen:
                    return h.score
                sim = max(float(vec[h.article.article_id] @ vec[c.article.article_id]) for c in chosen)
                return h.score - config.MMR_PENALTY * max(0.0, sim - config.MMR_SIM) / (1 - config.MMR_SIM)
            nxt = max(pool, key=adj)
            chosen.append(nxt)
            pool.remove(nxt)
        return chosen


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def _best_device() -> str:
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"
