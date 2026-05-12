# Devoir 3 — Snort rule generator: comparing 7 RAG architectures

Generates Snort IDS rules from natural-language attack descriptions, comparing seven retrieval architectures on the same task with the same local LLM. Independent from devoirs 0–2.

## Architectures compared

| # | Architecture | One-line description |
|---|---|---|
| 0 | LLM-only (baseline) | No retrieval. Pure hallucination test. |
| 1 | Classic RAG | Dense retrieval (FAISS, MiniLM) + LLM. |
| 2 | Re-ranking RAG | Classic + CrossEncoder re-scores candidates. |
| 3 | Hybrid RAG | BM25 (sparse) + FAISS (dense), fused with Reciprocal Rank Fusion. |
| 4 | Multi-hop RAG | Two-step retrieval: original query + LLM-reformulated query. |
| 5 | Graph RAG | NetworkX co-occurrence graph traversal for context expansion. |
| 6 | Agentic RAG | Self-evaluation loop: generate → score → re-retrieve if low → regenerate. |

## Stack

- **LLM**: `mlx-community/Mistral-7B-Instruct-v0.3-4bit` via MLX (Apple Silicon).
  - Linux/Intel fallback: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` via HuggingFace transformers (CPU).
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384-d).
- **Sparse retrieval**: `rank_bm25`.
- **Vector index**: FAISS `IndexFlatIP` (cosine on normalized vectors).
- **Re-ranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- **Metrics**: BLEU, ROUGE-L, Snort syntax validity, faithfulness, latency.
- **UI**: Gradio (gated behind `LAUNCH_UI=1`).

## Files

| File | Purpose |
|---|---|
| `snort_rag_devoir3.ipynb` | Main notebook with all 7 architectures + evaluation + Gradio. |
| `dataset/custom_snort_rag_dataset_500.{csv,json}` | 500-rule training corpus. |
| `dataset/snort_dataset.{csv,json}` | Smaller curated set (200 rules). |
| `dataset/generate_custom_snort_dataset.py` | Script that built the 500-rule dataset. |
| `results_comparison.csv` | Final evaluation table for the 7 architectures. |
| `*.png` | Figures (t-SNE, radar chart, metrics heatmap, knowledge graph, dataset distribution). |
| `requirements.txt` | Points to `../requirements.txt`. |

## Run

```bash
cd devoir3
jupyter notebook snort_rag_devoir3.ipynb
```

To launch the Gradio UI:

```bash
LAUNCH_UI=1 jupyter notebook snort_rag_devoir3.ipynb
```

## Results headline

Agentic RAG wins on the global score (0.5017), Multi-hop wins on BLEU (0.4131), Hybrid is the best performance/latency trade-off. LLM-without-RAG has faithfulness = 0.0 (every output is ungrounded). See `results_comparison.csv` and the figures.
