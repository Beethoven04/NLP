# Devoir 2 — RAG assistant for the Moroccan Code de la Route (law 52.05)

Answers natural-language questions — in **Arabic, Moroccan Darija or French** — using only the text of
law 52.05 (consolidated version of 10 July 2024), and cites the articles it used.

## Results

Evaluated on a hand-labelled set of **75 realistic questions** (63 Arabic, 4 Darija, 8 French; gold
article + key facts for each) and **14 off-topic questions** (`eval/gold_questions.py`).

| Metric | Old system | New system |
|---|---|---|
| Retrieval Hit@5 (gold article in top 5, 75 q.) | 44 % ¹ | **92 %** |
| Retrieval Hit@1 | – | 79 % |
| Darija / French questions retrieved | – | 4/4 · 8/8 |
| Off-topic questions refused | – | **14/14** |
| In-domain questions wrongly refused | – | 0/75 |
| **Answer accuracy** (Gemini 2.5 Flash judge, qwen3.8-27b answers) ² | – | **87.9 %** (58/66) |
| Answers containing every key fact (amounts, points, durations) ² | – | **90.9 %** |
| Answers citing the gold article ² | – | 92.4 % |

¹ old system scored leniently: a retrieved *page* counted as a hit if it overlapped the gold article's pages
(`python -m eval.baseline_old`).
² 66 of the 75 questions (63 Arabic + 3 Darija): the Groq free-tier daily cap stopped generation before the
French questions. Raw data: `eval/results/answers.jsonl`, `judgments.jsonl`, `summary.json`.
Of the 8 judged failures, 5 are retrieval misses (art. 95, 109, 219, 259, and art. 184 for wrong-way driving).

## What was wrong before

1. **Corrupted source text.** The devoir-1 CSV came from a broken PDF extraction: Arabic in visual order with
   swapped lam-alef ligatures (`االمانه` for `الأمانة`), numbers moved to the wrong side of words (fines and
   points corrupted), the points table (art. 99) flattened into garbage.
2. **Wrong unit.** The 126 CSV rows were **PDF pages**, not articles (the law has 329 articles incl. *bis*), so
   the retriever could never point at the right article and citations were page numbers.
3. **Weak retrieval.** 300-character chunks cut sentences mid-word, MiniLM embeddings, no lexical search, no reranking;
   everyday words ("القيادة", "الضو الحمر", "permis") never matched legal wording ("السياقة", "الضوء الأحمر").
4. **Toy evaluation.** 5 questions whose gold labels did not match the CSV ids.

## Pipeline

```
PDF ──► rag/extract_pdf.py  glyph-level extraction (PyMuPDF rawdict): RTL ordering, ligature + bidi-number fix,
                            table re-rendering, segmentation into 329 articles + book/part/chapter breadcrumb
    ──► data/articles.jsonl
    ──► rag/enrich.py       doc2query: LLM writes the questions citizens ask for each chunk;
                            kept only if the reranker confirms the chunk answers them (2,282 questions)

question ──► router (LLM): in-domain? + rewrite in formal legal Arabic (handles Darija / French)
         ──► stage 1: BM25 (normalized + light-stemmed Arabic, synonym expansion)
                      + dense bge-m3 over law chunks + dense over doc2query questions
                      → article-level RRF fusion (original + rewritten query)
         ──► stage 2: cross-encoder bge-reranker-v2-m3 on the best chunks of candidate articles;
                      final score = 0.6·rerank + 0.4·RRF
         ──► OOD gate (router verdict + reranker confidence) → polite refusal
         ──► grounded prompt (articles with chapter headings) → LLM answer in the user's language
         ──► citation check: any cited article not in the context is flagged
```

Chunks keep legal structure: each carries its article number and chapter, long enumerations (art. 184/185:
40 offences each) repeat the lead sentence that states the fine, and short articles that only say
"تطبيقا للمادة 94" get the lead of the referenced article as a cross-reference note.

## LLMs

| Name in app | Backend | Notes |
|---|---|---|
| `gpt-oss-120b` | Groq | default; strongest |
| `qwen3.8-27b` | Groq | Qwen3 family |
| `allam-2-7b` | Groq | Arabic-native (SDAIA) |
| `qwen2.5-7b-local` | MLX, on-device | no API key, ~15–25 s/answer on M2 |

If a cloud model runs out of quota the app falls back along `FALLBACK_CHAIN` (config.py) down to the local model.

> **Groq free tier limits** (checked Nov 2026): 8,000 tokens/min and **200,000 tokens/day per model**
> (≈ 60–80 answers/day on gpt-oss-120b). For customers, use a paid Groq tier or run the local model.

## Setup

```bash
cd devoir2
pip install -r requirements.txt
echo "GROQ_API_KEY=gsk_..." > .env          # optional: without it only the local model is used

python -m rag.extract_pdf                   # PDF -> data/articles.jsonl (already committed)
python -m rag.enrich                        # doc2query -> data/article_questions.json (already committed)
python app.py                               # web UI on http://127.0.0.1:7860
python rag_juridique.py "ما هي غرامة عدم احترام الضوء الأحمر؟"   # CLI
```

First run downloads bge-m3 and bge-reranker-v2-m3 (~4.5 GB) and caches the embeddings in `data/cache/`.
Needs ~6 GB free RAM.

## Evaluation

```bash
python -m eval.eval_retrieval --rewrites eval/results/rewrites.json -v   # ablation: bm25 / dense / hybrid / +rerank / +rewrite
python -m eval.eval_answers --models gpt-oss-120b qwen3.8-27b allam-2-7b qwen2.5-7b-local
python -m eval.eval_answers --report-only
python -m eval.baseline_old                                              # original system
```

`eval_answers` is resumable (Groq daily caps): answers → `results/answers.jsonl`, judgments →
`results/judgments.jsonl`, metrics → `results/summary.json`. The judge never grades its own model family and
sees the gold article plus the articles the assistant was given.

## Files

| Path | Purpose |
|---|---|
| `rag/extract_pdf.py` | PDF → clean articles |
| `rag/text.py` | Arabic normalization, tokenization, light stemmer |
| `rag/retriever.py` | chunking, BM25 + dense + doc2query fusion, reranking |
| `rag/enrich.py` | doc2query generation + reranker filter |
| `rag/llm.py` | Groq / MLX backends, retries, quota fallback |
| `rag/pipeline.py` | router, OOD gate, prompt, citation check |
| `rag/config.py` | every tunable (env-overridable) |
| `app.py` | Gradio UI (single answer + 3-model comparison) |
| `rag_juridique.py` | CLI |
| `eval/` | gold set, retrieval + answer evaluation, old-system baseline |
| `data/articles.jsonl` | knowledge base (replaces `export_final.csv`) |
| `devoir2_rag.ipynb`, `export_final.csv` | original submission, kept for reference |

## Limits

* Gold set is 75 questions written by the developer; a held-out set from real users would be a stronger test.
* The answer is only as current as the consolidated PDF (10 July 2024). Not legal advice.
* Retrieval misses left: questions whose wording shares nothing with the article (e.g. "عدم دفع الغرامة" → art. 95),
  and wrong-way driving, where accident articles listing it as an aggravating factor outrank art. 184.
