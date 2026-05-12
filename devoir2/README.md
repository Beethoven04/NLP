# Devoir 2 — RAG over Moroccan traffic law (code de la route)

RAG system that answers natural-language questions about the Moroccan code de la route, using the `export_final.csv` produced by devoir 1.

Two parallel implementations are committed — pick one for grading. Both follow the same pipeline (load CSV → normalize → chunk → embed → FAISS → retrieve top-K → LLM with the retrieved context → Gradio UI). They differ only in the LLM backend.

| Notebook | LLM backend | Models | Recommended for |
|---|---|---|---|
| `rag_juridique_submission.ipynb` | **Groq** API | llama-3.1-8b, llama-3.3-70b, gemma2-9b | Reliable free tier, faster latency |
| `devoir2_rag.ipynb` | **HuggingFace Inference** API | Qwen3-8B, Mistral-7B-Instruct, Phi-3.5-mini | Matches the assignment's literal model list (mistral, qwen3, GPT) |

Per `ASSIGNMENT.txt`, the HF version maps more directly to "mistral, qwen3, GPT". The Groq version is more reliable in practice (HF Inference API often returns 503s on the free tier in 2025).

## Setup

```bash
cd devoir2
cp .env.example .env  # then edit and add your real key
```

`.env` (this file is gitignored):

```
# For rag_juridique_submission.ipynb (Groq):
GROQ_API_KEY=gsk_your_key_here

# For devoir2_rag.ipynb (HF Inference):
HF_TOKEN=hf_your_token_here
```

## Run

```bash
jupyter notebook rag_juridique_submission.ipynb     # Groq version
# or
jupyter notebook devoir2_rag.ipynb                  # HF version
```

To launch the Gradio web UI from inside the notebook, set the env var:

```bash
LAUNCH_UI=1 jupyter notebook rag_juridique_submission.ipynb
```

By default the Gradio launch is skipped so that `nbconvert --execute` doesn't hang forever.

## Files

| File | Purpose |
|---|---|
| `rag_juridique_submission.ipynb` | Primary submission (Groq). |
| `devoir2_rag.ipynb` | Alternative submission (HF Inference). |
| `rag_juridique.py` | Standalone Python version of the Groq notebook. |
| `export_final.csv` | Knowledge base — 126 articles from devoir 1. |
| `ASSIGNMENT.txt` | Professor's brief (original spec). |
| `requirements.txt` | Points to `../requirements.txt`. |

The other files in this folder (`devoir2_rag_kaggle.ipynb`, `nlp-pipeline-week-3.ipynb`, `WhatsApp AI Agent Architecture.html`, `create_devoir2_notebook.py`, `devoir2`) are obsolete iterations or unrelated. **Delete them before pushing.**

## Architecture

```
question
  → normalize_arabic()
  → embed (paraphrase-multilingual-MiniLM-L12-v2)
  → FAISS IndexFlatIP (cosine, top-K)
  → if max_sim < OOD_THRESHOLD: refuse politely
  → build prompt with article references in arabic
  → Groq LLM (or HF Inference) → answer
  → return answer + cited article ids
```

Evaluation: precision@5 / recall@5 on a 5-question handcrafted set; OOD detection threshold tuned at 0.30 cosine.
