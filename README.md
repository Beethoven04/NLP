# Arabic NLP and Retrieval-Augmented Generation

Natural language processing projects on Arabic legal text and network-security data, built during the
Master IASD (FST Tanger, 2026). The flagship project is a retrieval-augmented assistant that answers questions
about the Moroccan traffic code in Arabic, Moroccan Darija and French, with article-level citations.

## Highlights

**[Legal QA assistant for the Moroccan traffic code](devoir2/)** — law 52.05, 329 articles

| Metric | Baseline | Final |
|---|---|---|
| Retrieval Hit@5 | 44 % | 92 % |
| Answer accuracy (LLM-judged) | – | 87.9 % |
| Answers with every key figure correct | – | 90.9 % |
| Off-topic questions refused | – | 14 / 14 |

Glyph-level PDF extraction for right-to-left Arabic, hybrid BM25 + dense retrieval with cross-encoder
reranking, doc2query enrichment, LLM query rewriting for dialect and French, and a 75-question
hand-labelled evaluation set.

## Projects

| Project | Topic | Techniques |
|---|---|---|
| [devoir0](devoir0/) | Arabic text preprocessing | Normalization, tokenization, stopwords, TF-IDF, cosine similarity |
| [devoir1](devoir1/) | Structuring the traffic code | PDF parsing, rule-based extraction (articles, fines, points), TF-IDF clustering, NER |
| [devoir2](devoir2/) | Legal RAG assistant | Hybrid retrieval, reranking, doc2query, query rewriting, LLM evaluation, Gradio |
| [devoir3](devoir3/) | Snort rule generation | Seven RAG architectures compared: classic, rerank, hybrid, multi-hop, graph, agentic |

## Stack

Python · PyMuPDF · sentence-transformers (bge-m3, bge-reranker-v2-m3) · rank-bm25 · FAISS ·
Groq (gpt-oss, Qwen3, ALLaM) · MLX (on-device Qwen2.5) · Gemini (evaluation) · Gradio · NetworkX

## Getting started

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # all projects
pip install -r devoir2/requirements.txt  # or a single project
python verify_environment.py
```

Each project folder has its own README with run instructions. API keys (Groq, Gemini) go in a local `.env`
file, which is gitignored.

## Repository layout

```
devoir0/   Arabic NLP fundamentals (notebook)
devoir1/   Traffic-code extraction pipeline (notebook, source PDF)
devoir2/   Legal RAG assistant (Python package, app, evaluation)
devoir3/   Snort RAG architecture comparison (notebook, datasets, figures)
```
