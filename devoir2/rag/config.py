"""Central configuration. Every value can be overridden with an environment variable."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
ARTICLES_PATH = Path(os.getenv("ARTICLES_PATH", DATA_DIR / "articles.jsonl"))
PDF_PATH = Path(os.getenv("PDF_PATH", ROOT.parent / "devoir1" / "code de la route MA52_05.pdf"))

try:  # optional .env (GROQ_API_KEY, ...)
    from dotenv import load_dotenv

    for _p in (ROOT / ".env", ROOT.parent / ".env"):
        if _p.exists():
            load_dotenv(_p)
            break
except ImportError:
    pass

# ── Retrieval ────────────────────────────────────────────────────────────────
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
USE_RERANKER = os.getenv("USE_RERANKER", "1") == "1"
CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", "140"))
CANDIDATES = int(os.getenv("CANDIDATES", "30"))      # articles per ranking before RRF fusion
RERANK_ARTICLES = int(os.getenv("RERANK_ARTICLES", "8"))    # RRF head sent to the cross-encoder
PER_CHANNEL_TOP = int(os.getenv("PER_CHANNEL_TOP", "4"))    # + each channel's own top-N articles
CHUNKS_PER_ARTICLE = int(os.getenv("CHUNKS_PER_ARTICLE", "3"))
RERANK_WEIGHT = float(os.getenv("RERANK_WEIGHT", "0.6"))    # final = w*rerank + (1-w)*rrf_norm
MMR_SIM = float(os.getenv("MMR_SIM", "0.85"))          # cosine above which articles count as near-duplicates
MMR_PENALTY = float(os.getenv("MMR_PENALTY", "0"))
USE_DOC2QUERY = os.getenv("USE_DOC2QUERY", "1") == "1"
QUESTIONS_PATH = DATA_DIR / "article_questions.json"
TOP_K = int(os.getenv("TOP_K", "5"))                 # articles returned to the LLM
RRF_K = 60

# Out-of-domain gate: reranker probability of the best chunk (tuned in eval/eval_answers.py)
OOD_RERANK_THRESHOLD = float(os.getenv("OOD_RERANK_THRESHOLD", "0.02"))
OOD_DENSE_THRESHOLD = float(os.getenv("OOD_DENSE_THRESHOLD", "0.35"))  # used when reranker is off

# ── Generation ───────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# Models available on Groq (checked Nov 2026). Three families: GPT, Qwen3, ALLaM (Arabic-native).
GROQ_MODELS = {
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "qwen3.8-27b": "qwen/qwen3.8-27b",
    "allam-2-7b": "allam-2-7b",
    "gpt-oss-20b": "openai/gpt-oss-20b",
}
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-oss-120b")
UTILITY_MODEL = os.getenv("UTILITY_MODEL", "gpt-oss-20b")   # query rewriting (fast)
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-oss-120b")      # evaluation only
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")   # used as independent eval judge ("gemini")
MLX_MODEL = os.getenv("MLX_MODEL", "mlx-community/Qwen2.5-7B-Instruct-4bit")
LOCAL_MODELS = {"qwen2.5-7b-local": MLX_MODEL}
# When a cloud model is out of daily quota, the app falls back along this chain.
FALLBACK_CHAIN = os.getenv("FALLBACK_CHAIN", "gpt-oss-120b,qwen3.8-27b,gpt-oss-20b,qwen2.5-7b-local").split(",")
QUERY_REWRITE = os.getenv("QUERY_REWRITE", "1") == "1"
