"""Doc2query enrichment: generate the questions a citizen would ask for each article.

The questions are indexed (BM25 + dense) as extra retrieval keys pointing to their
article. This bridges the vocabulary gap between everyday wording ("ما عقوبة عدم دفع
الغرامة؟") and legal wording ("تصدر الإدارة قرارا بسحب رخصة السياقة إذا لم يدفع...").
They are never shown to the LLM as law text.

    python -m rag.enrich   # raw -> data/article_questions_raw.json, filtered -> data/article_questions.json
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import config
from .llm import chat
from .retriever import load_articles

OUT = config.DATA_DIR / "article_questions_raw.json"   # filtered copy -> config.QUESTIONS_PATH

PROMPT = """Voici un article du Code de la route marocain (loi 52.05).

[المادة {aid}] ({crumb})
{text}

Écris jusqu'à 6 questions qu'un citoyen marocain poserait et dont la réponse se trouve EXPLICITEMENT
dans CET article (et pas seulement dans d'autres articles).
Règles :
- Chaque question doit mentionner le sujet précis de l'article (ex: "رخصة السياقة الإلكترونية", "شهادة التسجيل",
  "الفحص الطبي", "مؤسسة تعليم السياقة"...), jamais une question générique ("ما هي الغرامة؟").
- N'invente rien : si l'article ne prévoit ni amende ni points, ne pose pas de question sur une amende ou des points.
- Si l'article est purement procédural ou administratif, pose des questions sur cette procédure.
- 4 questions en arabe courant (sans numéro d'article ; vocabulaire de tous les jours), 1 en darija marocaine, 1 en français.
Réponds UNIQUEMENT par une liste JSON de chaînes."""


GEN_MODELS = ["qwen3.8-27b", "gpt-oss-20b", "qwen2.5-7b-local"]  # separate rate-limit budgets


def generate(aid: str, crumb: str, text: str, model: str = "gpt-oss-120b") -> list[str]:
    raw = chat(model, [{"role": "user", "content": PROMPT.format(aid=aid, crumb=crumb, text=text[:4000])}],
               max_tokens=900, temperature=0.3)
    m = re.search(r"\[.*\]", raw, re.S)
    try:
        qs = json.loads(m.group(0)) if m else []
        return [str(q).strip() for q in qs if str(q).strip()][:8]
    except Exception:
        return []


def filter_questions(min_prob: float = 0.2):
    """Keep a question only if the reranker says its own article answers it."""
    from .retriever import HybridRetriever

    data = json.loads(OUT.read_text(encoding="utf-8"))
    r = HybridRetriever(use_questions=False)
    kept = {}
    dropped = 0
    for key, qs in data.items():
        aid = key.split("#")[0]
        if aid not in r.chunks_of:
            continue
        texts = [r.chunks[ci].text for ci in r.chunks_of[aid]]
        keep = []
        for q in qs:
            p = max(r.reranker.predict([(q, t) for t in texts], show_progress_bar=False))
            if p >= min_prob:
                keep.append(q)
            else:
                dropped += 1
        kept[key] = keep
    config.QUESTIONS_PATH.write_text(json.dumps(kept, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"kept {sum(map(len, kept.values()))} questions, dropped {dropped}")


def _units():
    """One unit per article, or one per chunk for long (multi-chunk) articles.

    Key "184#3" = 4th chunk of article 184. Long enumerations (art. 184/185: 40 offences
    each) need questions per chunk, otherwise most offences get no question at all.
    """
    from .retriever import build_chunks

    arts = load_articles()
    chunks = build_chunks(arts)
    per: dict[str, list] = {}
    for c in chunks:
        per.setdefault(c.article_id, []).append(c)
    by_id = {a.article_id: a for a in arts}
    for aid, cs in per.items():
        a = by_id[aid]
        crumb = a.breadcrumb.split(" > ")[-1]
        if len(cs) == 1:
            yield aid, aid, crumb, a.text
        else:
            for i, c in enumerate(cs):
                yield f"{aid}#{i}", aid, crumb, c.text


def main(workers: int = len(GEN_MODELS)):
    done = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    units = list(_units())
    valid = {u[0] for u in units}
    done = {k: v for k, v in done.items() if k in valid and v}  # drop stale article-level keys of long articles
    todo = [u for u in units if u[0] not in done]
    print(f"{len(done)} cached, {len(todo)} to generate")
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(generate, aid, crumb, text, GEN_MODELS[i % len(GEN_MODELS)]): key
                for i, (key, aid, crumb, text) in enumerate(todo)}
        for i, f in enumerate(as_completed(futs), 1):
            key = futs[f]
            try:
                qs = f.result()
                if qs:
                    done[key] = qs
            except Exception as e:
                print("fail", key, str(e)[:100])
            if i % 25 == 0:
                OUT.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
                print(i, "done")
    OUT.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
    print("total", len(done), "missing", [u[0] for u in units if u[0] not in done])


if __name__ == "__main__":
    import sys

    main()
    if "--no-filter" not in sys.argv:
        filter_questions()
