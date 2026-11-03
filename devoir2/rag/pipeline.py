"""End-to-end RAG pipeline for law 52.05 (Moroccan Code de la Route).

question
  -> LLM query analysis: domain check + rewrite in formal legal Arabic (handles Darija/French)
  -> hybrid retrieval (BM25 + bge-m3, RRF) -> cross-encoder rerank -> top-K articles
  -> out-of-domain gate (LLM verdict + reranker confidence)
  -> grounded prompt with numbered article excerpts
  -> answer in the user's language, citing "المادة N"
  -> citation check: cited articles must belong to the retrieved context
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from . import config
from .llm import chat
from .retriever import Hit, HybridRetriever

REFUSAL = {
    "ar": "عذرا، هذا السؤال خارج نطاق مدونة السير على الطرق (القانون 52.05). يمكنني الإجابة فقط عن الأسئلة المتعلقة بقانون السير المغربي.",
    "fr": "Désolé, cette question sort du champ du Code de la route marocain (loi 52.05). Je ne réponds qu'aux questions sur ce code.",
}

ANALYZE_PROMPT = """You route questions for an assistant that ONLY knows the Moroccan traffic code (law 52.05: driving licence, points, fines, traffic offences, accidents, vehicles, registration, technical inspection, impound, driving schools, road police procedures).

User question (may be Arabic, Moroccan Darija or French):
{q}

Return ONLY a JSON object:
{{"in_domain": true|false, "lang": "ar"|"fr", "query": "<the question rewritten as one formal Modern Standard Arabic question using the vocabulary of law 52.05, e.g. السياقة، رخصة السياقة، خصم النقط، غرامة، يعاقب، المركبة، الضوء الأحمر، حالة سكر، شهادة التسجيل>"}}
"lang" is "fr" if the user wrote in French, otherwise "ar". If not in domain, "query" can be empty."""

SYSTEM_PROMPT = """أنت مساعد قانوني خبير في مدونة السير على الطرق المغربية (القانون رقم 52.05).
قواعد صارمة:
1. أجب فقط اعتمادا على المواد القانونية المقدمة في السياق. لا تستعمل أي معلومة من خارجه ولا تخترع أرقاما.
2. ابدأ بالجواب المباشر (المبلغ، عدد النقط، المدة...) ثم التفاصيل المهمة (حالة العود، العقوبات الإضافية، التخفيض عند الأداء الفوري...) إن وجدت في السياق.
3. اذكر الأرقام كما وردت في النص بالضبط (مثال: من 700 إلى 1400 درهم).
4. استشهد بعد كل معلومة برقم المادة بين معقوفتين هكذا: [المادة 184].
5. إذا كان السياق لا يتضمن الجواب، قل ذلك صراحة ولا تخمن.
6. اكتب بلغة السؤال: بالعربية الفصحى إذا كان السؤال بالعربية أو بالدارجة، وبالفرنسية إذا كان السؤال بالفرنسية (مع الإبقاء على المراجع [المادة N]).
7. كن موجزا وواضحا (من 2 إلى 6 جمل أو نقاط)."""


@dataclass
class RAGResult:
    question: str
    answer: str
    sources: list[dict] = field(default_factory=list)
    in_domain: bool = True
    rewritten: str = ""
    lang: str = "ar"
    model: str = ""
    latency: float = 0.0
    timings: dict = field(default_factory=dict)


def _dedupe_repetition(text: str) -> str:
    """Small models (e.g. allam-2-7b) sometimes loop on one sentence until max_tokens.
    Keep the first occurrence of each line and drop a truncated trailing duplicate."""
    out, seen = [], set()
    for line in text.split("\n"):
        key = re.sub(r"\W+", "", line)
        if key and key in seen:
            continue
        if key and any(k.startswith(key) or key.startswith(k) for k in seen if len(key) > 40 and len(k) > 40):
            continue
        seen.add(key)
        out.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def detect_lang(text: str) -> str:
    latin = len(re.findall(r"[A-Za-zÀ-ÿ]", text))
    arabic = len(re.findall(r"[؀-ۿ]", text))
    return "fr" if latin > arabic else "ar"


def build_context(hits: list[Hit]) -> str:
    parts = []
    for h in hits:
        a = h.article
        body = a.text if len(a.text.split()) <= 220 else "\n…\n".join(h.chunks)
        crumb = f" ({a.breadcrumb.split(' > ')[-1]})" if a.breadcrumb else ""
        parts.append(f"[المادة {a.article_id}]{crumb}\n{body}")
    return "\n\n---\n\n".join(parts)


class RAGPipeline:
    def __init__(self, retriever: HybridRetriever | None = None, rewrite: bool = config.QUERY_REWRITE,
                 utility_model: str = config.UTILITY_MODEL, fallback: bool = True):
        self.retriever = retriever or HybridRetriever()
        self.fallback = fallback  # app: True (always answer); evaluation: False (measure one model)
        self.rewrite = rewrite
        self.utility_model = utility_model

    # ── step 1: domain check + legal rewrite ────────────────────────────────
    def analyze(self, question: str) -> dict:
        default = {"in_domain": True, "lang": detect_lang(question), "query": ""}
        if not self.rewrite:
            return default
        try:
            raw = chat(self.utility_model, [{"role": "user", "content": ANALYZE_PROMPT.format(q=question)}],
                       max_tokens=200, fallback=True)
            m = re.search(r"\{.*\}", raw, re.S)
            data = json.loads(m.group(0)) if m else {}
            return {"in_domain": bool(data.get("in_domain", True)),
                    "lang": data.get("lang") if data.get("lang") in ("ar", "fr") else default["lang"],
                    "query": str(data.get("query") or "").strip()}
        except Exception:
            return default  # never fail the request because of the router

    # ── step 2: retrieval + OOD gate ────────────────────────────────────────
    def retrieve(self, question: str, analysis: dict | None = None, k: int = config.TOP_K):
        analysis = analysis or self.analyze(question)
        hits = self.retriever.search(question, k=k, rewritten=analysis.get("query") or None)
        top = max((h.signals.get("rerank_max") or 0.0) for h in hits) if hits else 0.0
        if self.retriever.reranker is not None:
            confident = top >= config.OOD_RERANK_THRESHOLD
        else:
            confident = (hits[0].signals.get("dense_max", 0) if hits else 0) >= config.OOD_DENSE_THRESHOLD
        # The LLM router vetoes only when retrieval is not clearly confident.
        in_domain = confident and (analysis["in_domain"] or top >= 0.5)
        return hits, in_domain

    # ── step 3: generation ──────────────────────────────────────────────────
    def prepare(self, question: str, k: int = config.TOP_K, analysis: dict | None = None) -> dict:
        """Model-independent part (routing + retrieval); reusable across LLMs."""
        t0 = time.time()
        analysis = analysis or self.analyze(question)
        t1 = time.time()
        hits, in_domain = self.retrieve(question, analysis, k)
        return {"analysis": analysis, "hits": hits, "in_domain": in_domain,
                "timings": {"analyze": round(t1 - t0, 2), "retrieve": round(time.time() - t1, 2)}}

    def answer(self, question: str, model: str = config.DEFAULT_MODEL, k: int = config.TOP_K,
               prepared: dict | None = None) -> RAGResult:
        t0 = time.time()
        prep = prepared or self.prepare(question, k)
        analysis, hits, in_domain = prep["analysis"], prep["hits"], prep["in_domain"]
        timings = dict(prep["timings"])
        lang = analysis["lang"]
        if not in_domain:
            return RAGResult(question, REFUSAL[lang], [], False, analysis.get("query", ""), lang, model,
                             round(sum(timings.values()), 2), timings)
        t2 = time.time()
        context = build_context(hits)
        user = f"السياق (مواد من القانون 52.05):\n\n{context}\n\n=== السؤال ===\n{question}"
        if analysis.get("query") and analysis["query"] != question:
            user += f"\n(صياغة قانونية للسؤال: {analysis['query']})"
        if lang == "fr":
            user += "\n\nRéponds en français."
        try:
            ans = chat(model, [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
                       fallback=self.fallback)
        except Exception as e:
            ans = f"[LLM error] {e}"
        t3 = time.time()
        ans = self._check_citations(_dedupe_repetition(ans), hits)
        sources = [{"article": h.article.article_id, "score": round(h.score, 3),
                    "breadcrumb": h.article.breadcrumb, "page": h.article.page_start} for h in hits]
        timings["generate"] = round(t3 - t2, 2)
        return RAGResult(question, ans, sources, True, analysis.get("query", ""), lang, model,
                         round(sum(timings.values()), 2), timings)

    @staticmethod
    def _check_citations(answer: str, hits: list[Hit]) -> str:
        """Flag citations to articles that were not in the retrieved context (hallucinated refs)."""
        allowed = {h.article.article_id for h in hits}
        cited = set(re.findall(r"المادة\s*([\d]+(?:\s*-\s*\d+)?)", answer))
        bad = {c.replace(" ", "") for c in cited} - allowed
        if bad:
            answer += "\n\n⚠️ " + "، ".join(f"المادة {b}" for b in sorted(bad)) + " غير موجودة في المقتطفات المسترجعة؛ يرجى التحقق منها."
        return answer
