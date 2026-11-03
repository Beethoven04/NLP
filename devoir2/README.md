# Legal QA Assistant — Moroccan Traffic Code (Law 52.05)

A retrieval-augmented generation system that answers questions about the Moroccan traffic code in Arabic,
Moroccan Darija and French. Every answer is grounded in the text of the law and cites the articles it relies on.
Off-topic questions are refused.

```
Q: شحال الخلاص ديال الضو الحمر؟

A: غرامة عدم احترام الوقوف المفروض بضوء التشوير الأحمر (الضوء الأحمر) هي من سبعمائة (700) إلى ألف وأربعمائة (1.400) درهم [المادة 184].
   تُعد هذه المخالفة من الدرجة الأولى [المادة 184].
   كما يؤدي هذا الفعل إلى خصم 4 نقط من رصيد رخصة السياقة [المادة 99].
```

Verbatim output (`qwen3.8-27b`) from the evaluation run.

## Results

Evaluated on 75 hand-labelled questions (63 Arabic, 4 Darija, 8 French), each with gold articles and key facts,
plus 14 off-topic questions.

| Metric | Baseline | This system |
|---|---|---|
| Retrieval Hit@5 | 44 % | **92 %** |
| Retrieval Hit@1 | – | 79 % |
| Answer accuracy (Gemini 2.5 Flash as judge) | – | **87.9 %** |
| Answers containing every key figure | – | **90.9 %** |
| Answers citing the gold article | – | 92.4 % |
| Off-topic questions refused / in-domain wrongly refused | – | 14 / 14 · 0 / 75 |

Answer metrics cover 66 questions generated with `qwen3.8-27b` before the provider's daily quota was reached.
The baseline is the original pipeline (page-level chunks, MiniLM embeddings), scored leniently at page level.

### Retrieval ablation (Hit@5)

| Configuration | Hit@5 |
|---|---|
| BM25 (normalized, stemmed Arabic) | 83 % |
| Dense (bge-m3) | 85 % |
| Hybrid, reciprocal rank fusion | 88 % |
| + cross-encoder reranking | 91 % |
| + LLM query rewriting | **92–93 %** |

## Architecture

```
                  ┌──────────────────────── offline ────────────────────────┐
 PDF (law 52.05) ─► glyph-level extraction ─► 329 articles ─► chunks ─► BM25 index
                    RTL order, ligatures,      + chapter       + doc2query questions
                    numbers, tables            breadcrumbs     ─► bge-m3 embeddings
                  └─────────────────────────────────────────────────────────┘

 question ─► router (LLM) ─► hybrid retrieval ─► cross-encoder ─► OOD gate ─► grounded prompt ─► answer
             domain check     BM25 + dense +     rerank           refuse if     cited articles      + citation
             legal rewrite    doc2query, RRF     bge-reranker     off-topic     in user language    check
```

## Engineering notes

**Arabic PDF extraction.** Standard extractors return the text in visual order with broken ligatures
(`االمانه` instead of `الأمانة`) and numbers on the wrong side of words, which silently corrupts fines and
point values. The extractor rebuilds each line from glyph coordinates (PyMuPDF `rawdict`), restores logical
order, re-orders left-to-right digit runs, and re-renders the points-deduction table row by row.

**Article-level knowledge base.** The source is segmented on `المادة N` headers into 329 articles, including
*bis* articles such as 61-1, each with its book, part and chapter. Long enumerations (articles 184 and 185 list
about 40 offences each) are chunked with the lead sentence that states the penalty repeated in every chunk.
Short articles that only refer elsewhere ("تطبيقا للمادة 94") carry the lead of the referenced article.

**Bridging the vocabulary gap.** Citizens write "القيادة", "الضو الحمر" or "permis"; the law says "السياقة",
"الضوء الأحمر", "رخصة السياقة". Three mechanisms close the gap:
- an Arabic and French synonym map applied to lexical search;
- an LLM router that rewrites the question in formal legal Arabic;
- doc2query: an LLM writes the questions each chunk answers, and a question is kept only if the reranker
  confirms its chunk answers it (2,282 kept, 263 rejected). These questions are indexed as a separate
  retrieval channel, so they cannot swamp the law text.

**Ranking.** Each channel (BM25, dense law, dense questions; original and rewritten query) is collapsed to
article level and fused with reciprocal rank fusion. The fused head, plus each channel's own top hits, is
rescored by `bge-reranker-v2-m3`. The final score blends reranker probability (0.6) and fused rank (0.4).

**Reliability.** Off-topic questions are refused when the router and the reranker confidence agree. Citations
to articles absent from the context are flagged. Degenerate repetition from small models is removed. When a
cloud model is out of quota, the app falls back along a model chain down to an on-device model.

**Evaluation.** Generation and judging are separate, resumable phases. The judge never grades its own model
family and receives the gold articles plus every article the assistant saw, in full.

## Models

| Name | Provider | Role |
|---|---|---|
| `gpt-oss-120b` | Groq | default answer model |
| `qwen3.8-27b` | Groq | answer model (evaluated) |
| `allam-2-7b` | Groq | Arabic-native comparison model |
| `qwen2.5-7b-local` | MLX, on-device | offline fallback, no API key |
| `gpt-oss-20b` | Groq | query router |
| `gemini-2.5-flash` | Google | evaluation judge |
| `BAAI/bge-m3`, `BAAI/bge-reranker-v2-m3` | local | embeddings, reranking |

## Usage

```bash
pip install -r requirements.txt
cp .env.example .env            # GROQ_API_KEY, optional GEMINI_API_KEY

python app.py                   # web interface on http://127.0.0.1:7860
python rag_juridique.py "ما هي عقوبة السياقة بدون رخصة؟"
```

The knowledge base (`data/articles.jsonl`) and doc2query questions are included. To rebuild them:

```bash
python -m rag.extract_pdf       # PDF -> data/articles.jsonl
python -m rag.enrich            # -> data/article_questions.json
```

The first run downloads the embedding and reranking models (about 4.5 GB) and caches the index in `data/cache/`.
About 6 GB of free memory is required.

## Evaluation

```bash
python -m eval.eval_retrieval --rewrites eval/results/rewrites.json -v
python -m eval.eval_answers --models gpt-oss-120b qwen3.8-27b allam-2-7b qwen2.5-7b-local
python -m eval.eval_answers --report-only
python -m eval.baseline_old
```

Outputs are written to `eval/results/` (`answers.jsonl`, `judgments.jsonl`, `summary.json`).

## Project structure

```
rag/
  extract_pdf.py   PDF to structured articles
  text.py          Arabic normalization, tokenization, light stemming
  retriever.py     chunking, BM25 / dense / doc2query fusion, reranking
  enrich.py        doc2query generation and filtering
  pipeline.py      routing, out-of-domain gate, prompting, citation check
  llm.py           Groq, Gemini and MLX backends with retries and quota fallback
  config.py        all parameters, overridable by environment variables
eval/              gold set, retrieval and answer evaluation, baseline
data/              knowledge base and doc2query questions
app.py             Gradio interface
rag_juridique.py   command-line interface
```

## Limitations

- The gold set was written by the author; a held-out set of real user questions would be a stronger test.
- Remaining errors are mostly retrieval misses where the question shares little vocabulary with the article
  (for example, unpaid fines and article 95).
- Answers reflect the consolidated text of 10 July 2024. The assistant provides information, not legal advice.
- The free Groq tier allows about 200k tokens per model per day; production use needs a paid tier or the
  on-device model.
