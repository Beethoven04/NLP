"""
RAG System — Code de la Route (Moroccan Traffic Law)
Master IASD 2026 — Devoir 2
Université Abdelmalek Essaadi

Requirements:
    pip install pandas sentence-transformers faiss-cpu groq gradio unicodedata2 tqdm scikit-learn

LLMs used (Groq free tier):
    - llama3.1-8b-8192
    - mixtral-8x7b-32768
    - gemma2-9b-it

Set your Groq API key:
    export GROQ_API_KEY="your_key_here"
    Or pass it directly in the GROQ_API_KEY variable below.
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. CONFIG
# ─────────────────────────────────────────────────────────────────────────────

import os
import re
import json
import time
import socket
import warnings
warnings.filterwarnings("ignore")

# Load .env if present
try:
    from dotenv import load_dotenv
    from pathlib import Path as _P
    for _c in (_P('.env'), _P('..') / '.env'):
        if _c.exists():
            load_dotenv(_c); break
except ImportError:
    pass

GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    print("⚠ GROQ_API_KEY not set. Put it in a .env file before running LLM cells.")
CSV_PATH       = os.getenv("CSV_PATH", "export_final.csv")
EMBED_MODEL    = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K          = 5                          # documents returned per query
CHUNK_SIZE     = 300                        # characters per chunk
CHUNK_OVERLAP  = 50
OOD_THRESHOLD  = 0.30                      # below this → out-of-domain

LLMS = {
    "llama3.1-8b"  : "llama-3.1-8b-instant",
    "llama3.3-70b" : "llama-3.3-70b-versatile",
    "gemma2-9b"    : "gemma2-9b-it",
}  # mixtral-8x7b removed: deprecated on Groq

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import unicodedata

def normalize_arabic(text: str) -> str:
    """Light normalization: strip diacritics, normalize whitespace."""
    if not isinstance(text, str):
        return ""
    # Remove Arabic diacritics (tashkeel)
    text = re.sub(r'[\u0617-\u061A\u064B-\u065F]', '', text)
    # Normalize alef variants → ا
    text = re.sub(r'[أإآ]', 'ا', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    print(f"[Data] Loaded {len(df)} rows, columns: {list(df.columns)}")

    # Make this script robust to different Devoir1 CSV schemas.
    # Preferred full text: texte_complet; fallback: infraction_desc + sanction_desc + mots_cles.
    if 'article_id' not in df.columns:
        raise ValueError("Column 'article_id' is required in CSV.")

    for col in ['infraction_desc', 'sanction_desc', 'mots_cles', 'localisation', 'type_infraction']:
        if col not in df.columns:
            df[col] = ''
        df[col] = df[col].fillna('')

    if 'chapitre' not in df.columns:
        # Fallback metadata column for UI/source display
        if 'type_infraction' in df.columns:
            df['chapitre'] = df['type_infraction'].fillna('')
        elif 'localisation' in df.columns:
            df['chapitre'] = df['localisation'].fillna('')
        else:
            df['chapitre'] = ''
    else:
        df['chapitre'] = df['chapitre'].fillna('')

    if 'texte_complet' in df.columns:
        df['texte_complet'] = df['texte_complet'].fillna('')
    else:
        df['texte_complet'] = df.apply(
            lambda r: " | ".join(
                x for x in [
                    str(r.get('infraction_desc', '')).strip(),
                    str(r.get('sanction_desc', '')).strip(),
                    str(r.get('mots_cles', '')).strip(),
                ] if x
            ),
            axis=1
        )

    df['article_id'] = df['article_id'].astype(str).str.strip()

    # Combine rich text field
    df['text_clean'] = df.apply(
        lambda r: normalize_arabic(
            f"المادة {r['article_id']} | {r['chapitre']} | {r['texte_complet']}"
        ), axis=1
    )
    df = df[df['text_clean'].str.len() > 10].reset_index(drop=True)
    print(f"[Data] After cleaning: {len(df)} rows")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. CHUNKING
# ─────────────────────────────────────────────────────────────────────────────

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split text into overlapping character-level chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def build_chunks(df: pd.DataFrame):
    """Return list of dicts: {chunk, article_id, chapitre, amende_fixe, points_retrait}"""
    records = []
    for _, row in df.iterrows():
        for ch in chunk_text(row['text_clean']):
            if len(ch.strip()) < 20:
                continue
            records.append({
                "chunk"          : ch,
                "article_id"     : row['article_id'],
                "chapitre"       : row['chapitre'],
                "amende_fixe"    : row.get('amende_fixe', ''),
                "points_retrait" : row.get('points_retrait', ''),
                "sanction_type"  : row.get('sanction_type', ''),
            })
    print(f"[Chunks] Total chunks: {len(records)}")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# 3. EMBEDDINGS & FAISS INDEX
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(self, model_name: str = EMBED_MODEL):
        print(f"[Embedder] Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.index  = None
        self.chunks = []

    def build(self, records: list):
        self.chunks = records
        texts = [r['chunk'] for r in records]
        print(f"[Embedder] Encoding {len(texts)} chunks…")
        embeddings = self.model.encode(texts, batch_size=64, show_progress_bar=True,
                                       normalize_embeddings=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)      # Inner-product = cosine (normalized)
        self.index.add(embeddings.astype('float32'))
        print(f"[FAISS] Index built: {self.index.ntotal} vectors, dim={dim}")

    def search(self, query: str, k: int = TOP_K):
        q_emb = self.model.encode([query], normalize_embeddings=True).astype('float32')
        scores, idxs = self.index.search(q_emb, k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            results.append({**self.chunks[idx], "score": float(score)})
        return results

    def max_similarity(self, query: str) -> float:
        results = self.search(query, k=1)
        return results[0]['score'] if results else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. GROQ LLM CLIENT
# ─────────────────────────────────────────────────────────────────────────────

from groq import Groq

groq_client = Groq(api_key=GROQ_API_KEY)


def build_prompt(question: str, docs: list) -> str:
    context_parts = []
    for i, d in enumerate(docs, 1):
        meta = f"[المادة {d['article_id']} | {d['chapitre']}]"
        context_parts.append(f"{meta}\n{d['chunk']}")
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""أنت مساعد قانوني متخصص في قانون المرور المغربي.
استخدم فقط المقتطفات القانونية التالية للإجابة على السؤال.
إذا كانت المعلومات غير كافية، قل ذلك بوضوح.
أذكر دائماً رقم المادة المستخدمة في إجابتك.

=== المقتطفات القانونية ===
{context}

=== السؤال ===
{question}

=== الإجابة ==="""
    return prompt


def query_llm(question: str, docs: list, model_key: str = "llama3.1-8b") -> dict:
    model_id = LLMS[model_key]
    prompt   = build_prompt(question, docs)
    t0 = time.time()
    try:
        response = groq_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=600,
            timeout=30,
        )
        answer = response.choices[0].message.content.strip()
        latency = round(time.time() - t0, 2)
    except Exception as e:
        answer  = f"[Error] {e}"
        latency = -1
    return {"model": model_key, "answer": answer, "latency": latency}


# ─────────────────────────────────────────────────────────────────────────────
# 5. FULL RAG PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class RAGPipeline:
    def __init__(self, vector_store: VectorStore):
        self.vs = vector_store

    def is_out_of_domain(self, question: str) -> bool:
        return self.vs.max_similarity(question) < OOD_THRESHOLD

    def run(self, question: str, model_key: str = "llama3.1-8b") -> dict:
        if self.is_out_of_domain(question):
            return {
                "question"  : question,
                "answer"    : "❌ هذا السؤال خارج نطاق قانون المرور المغربي.",
                "sources"   : [],
                "model"     : model_key,
                "ood"       : True,
                "latency"   : 0,
            }
        docs   = self.vs.search(question, k=TOP_K)
        result = query_llm(question, docs, model_key)
        return {
            "question" : question,
            "answer"   : result["answer"],
            "sources"  : [{"article": d["article_id"],
                           "chapitre": d["chapitre"],
                           "score": round(d["score"], 3)} for d in docs],
            "model"    : model_key,
            "ood"      : False,
            "latency"  : result["latency"],
        }

    def compare_llms(self, question: str) -> list:
        """Run the same question through all 3 LLMs."""
        if self.is_out_of_domain(question):
            return [{"model": k, "answer": "❌ خارج النطاق", "latency": 0}
                    for k in LLMS]
        docs = self.vs.search(question, k=TOP_K)
        return [query_llm(question, docs, k) for k in LLMS]


# ─────────────────────────────────────────────────────────────────────────────
# 6. EVALUATION  (precision / recall)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(pipeline: RAGPipeline, test_set: list) -> dict:
    """
    test_set: list of {"question": str, "relevant_articles": ["1","2",...]}
    Computes retrieval precision & recall @TOP_K
    """
    precisions, recalls = [], []
    for item in test_set:
        q          = item["question"]
        relevant   = set(str(a) for a in item["relevant_articles"])
        docs       = pipeline.vs.search(q, k=TOP_K)
        retrieved  = set(d["article_id"] for d in docs)
        tp         = len(relevant & retrieved)
        precisions.append(tp / len(retrieved) if retrieved else 0)
        recalls.append(tp / len(relevant)    if relevant  else 0)

    avg_p = round(sum(precisions) / len(precisions), 4) if precisions else 0
    avg_r = round(sum(recalls)    / len(recalls),    4) if recalls    else 0
    f1    = round(2*avg_p*avg_r / (avg_p+avg_r), 4)     if (avg_p+avg_r) else 0
    print(f"\n[Eval] Precision@{TOP_K}: {avg_p}  |  Recall@{TOP_K}: {avg_r}  |  F1: {f1}")
    return {"precision": avg_p, "recall": avg_r, "f1": f1}


# ─────────────────────────────────────────────────────────────────────────────
# 7. GRADIO WEB INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

import gradio as gr


def build_ui(pipeline: RAGPipeline):
    # ── Single model query ──────────────────────────────────────────────────
    def single_query(question, model_key):
        if not question.strip():
            return "الرجاء إدخال سؤال.", ""
        res = pipeline.run(question.strip(), model_key)
        answer = res["answer"]
        if res["ood"]:
            sources_text = ""
        else:
            lines = [f"• المادة {s['article']} — {s['chapitre']} (score: {s['score']})"
                     for s in res["sources"]]
            sources_text = "\n".join(lines)
            answer += f"\n\n⏱ {res['latency']}s"
        return answer, sources_text

    # ── Compare 3 LLMs ──────────────────────────────────────────────────────
    def compare_query(question):
        if not question.strip():
            return "", "", ""
        results = pipeline.compare_llms(question.strip())
        outputs = []
        for r in results:
            outputs.append(f"**{r['model']}** (⏱ {r['latency']}s)\n\n{r['answer']}")
        while len(outputs) < 3:
            outputs.append("")
        return outputs[0], outputs[1], outputs[2]

    def use_example(example_text):
        return example_text

    custom_css = """
    .gradio-container {
        background:
            radial-gradient(circle at 12% 20%, rgba(255, 182, 72, 0.18), transparent 32%),
            radial-gradient(circle at 88% 10%, rgba(24, 161, 251, 0.16), transparent 34%),
            linear-gradient(135deg, #f7f9fc 0%, #eef3f9 100%);
    }
    .rtl {direction: rtl; text-align: right;}
    .hero-card {
        border: 1px solid #d9e2ec;
        border-radius: 16px;
        padding: 14px 18px;
        background: #ffffffcc;
        box-shadow: 0 8px 26px rgba(15, 23, 42, 0.08);
    }
    .panel-card {
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 10px;
        background: #ffffff;
    }
    .examples .gr-button {
        background: #eaf3ff !important;
        color: #0f3d73 !important;
        border: 1px solid #c6dcfb !important;
    }
    """

    # ── UI Layout ────────────────────────────────────────────────────────────
    with gr.Blocks(
        title="RAG — قانون المرور المغربي",
        theme=gr.themes.Base(primary_hue="amber", secondary_hue="blue", neutral_hue="slate"),
        css=custom_css,
    ) as demo:

        with gr.Row():
            with gr.Column():
                gr.Markdown(
                    """
                    <div class="hero-card rtl">
                      <h1 style="margin:0;color:#0b3b66;">🚦 مساعد قانون السير المغربي</h1>
                      <p style="margin:8px 0 0;color:#1f2937;">
                        اسأل عن المخالفات، الغرامات، النقاط، أو الرخصة.
                        النظام يعتمد فقط على نصوص قاعدة قانون السير المسترجعة.
                      </p>
                    </div>
                    """
                )

        with gr.Tab("📌 الاستعلام القانوني"):
            with gr.Row():
                with gr.Column(scale=6):
                    with gr.Group(elem_classes="panel-card"):
                        q1 = gr.Textbox(
                            label="السؤال",
                            placeholder="مثال: ما هي عقوبة قيادة بدون رخصة؟",
                            lines=3,
                            elem_classes="rtl",
                        )
                        model_dd = gr.Dropdown(
                            choices=list(LLMS.keys()),
                            value="llama3.1-8b",
                            label="النموذج",
                        )
                        with gr.Row():
                            btn1 = gr.Button("🔎 تحليل السؤال", variant="primary")
                            clear_btn = gr.Button("🧹 مسح")

                with gr.Column(scale=4):
                    with gr.Group(elem_classes="panel-card"):
                        gr.Markdown("### أمثلة جاهزة", elem_classes="rtl")
                        with gr.Row(elem_classes="examples"):
                            ex1 = gr.Button("عقوبة بدون رخصة")
                            ex2 = gr.Button("غرامة الإشارة الحمراء")
                        with gr.Row(elem_classes="examples"):
                            ex3 = gr.Button("نقاط الهاتف أثناء السياقة")
                            ex4 = gr.Button("سحب الرخصة نهائياً")

            with gr.Row():
                with gr.Column(scale=6):
                    ans1 = gr.Textbox(label="الجواب", lines=10, elem_classes="rtl")
                with gr.Column(scale=4):
                    src1 = gr.Textbox(label="المراجع القانونية المستعملة", lines=10, elem_classes="rtl")

            btn1.click(single_query, inputs=[q1, model_dd], outputs=[ans1, src1])
            clear_btn.click(lambda: ("", "", ""), outputs=[q1, ans1, src1])
            ex1.click(lambda: "ما هي عقوبة قيادة السيارة بدون رخصة؟", outputs=[q1])
            ex2.click(lambda: "ما هي غرامة تجاوز الإشارة الحمراء؟", outputs=[q1])
            ex3.click(lambda: "كم نقطة تُخصم عند استخدام الهاتف أثناء القيادة؟", outputs=[q1])
            ex4.click(lambda: "متى يتم سحب رخصة السياقة نهائياً؟", outputs=[q1])

        with gr.Tab("🧪 ساحة مقارنة النماذج"):
            q2 = gr.Textbox(
                label="السؤال نفسه يُرسل لكل النماذج",
                placeholder="مثال: ما هو الحد الأقصى للسرعة داخل المدن؟",
                lines=2,
                elem_classes="rtl",
            )
            btn2 = gr.Button("⚖️ شغّل المقارنة", variant="primary")
            with gr.Row():
                out_llama = gr.Markdown("### LLaMA", elem_classes="rtl")
                out_mixtral = gr.Markdown("### Mixtral", elem_classes="rtl")
                out_gemma = gr.Markdown("### Gemma", elem_classes="rtl")
            btn2.click(compare_query, inputs=[q2], outputs=[out_llama, out_mixtral, out_gemma])

        with gr.Tab("ℹ️ ملاحظات"):
            gr.Markdown(
                """
                <div class="hero-card rtl">
                  <h3 style="margin-top:0;">كيف يعمل هذا النظام؟</h3>
                  <p>1) يسترجع مقاطع قانونية مشابهة من الفهرس.</p>
                  <p>2) يرسل السؤال + المقاطع إلى النموذج اللغوي.</p>
                  <p>3) يعرض الجواب مع المواد القانونية المستخدمة.</p>
                  <p style="margin-bottom:0;">إذا كان السؤال خارج المجال، سيظهر تنبيه خارج النطاق.</p>
                </div>
                """,
            )

        gr.Markdown("**Master IASD 2026 — Université Abdelmalek Essaadi**")

    return demo


def find_free_port(start: int = 7860, end: int = 7890) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise OSError(f"No free port found in range {start}-{end}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Load & process data ──────────────────────────────────────────────────
    df      = load_and_clean(CSV_PATH)
    records = build_chunks(df)

    # ── Build vector store ───────────────────────────────────────────────────
    vs = VectorStore()
    vs.build(records)

    # ── Create pipeline ──────────────────────────────────────────────────────
    rag = RAGPipeline(vs)

    # ── Quick sanity check ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("[Test] Single query…")
    r = rag.run("ما هي عقوبة قيادة السيارة بدون رخصة؟", "llama3.1-8b")
    print(f"Answer: {r['answer'][:200]}")
    print(f"Sources: {r['sources']}")

    # ── Evaluation with a small handcrafted test set ─────────────────────────
    TEST_SET = [
        {"question": "ما هي عقوبة قيادة السيارة بدون رخصة؟",
         "relevant_articles": ["1"]},
        {"question": "ما هو الحد الأقصى للسرعة داخل المدن؟",
         "relevant_articles": ["50", "51"]},
        {"question": "ما هي غرامة تجاوز الإشارة الحمراء؟",
         "relevant_articles": ["44", "45"]},
        {"question": "كم نقطة تُخصم عند استخدام الهاتف أثناء القيادة؟",
         "relevant_articles": ["75", "76"]},
        {"question": "متى يتم سحب رخصة السياقة نهائياً؟",
         "relevant_articles": ["20", "21"]},
    ]
    eval_results = evaluate(rag, TEST_SET)

    # ── Out-of-domain test ───────────────────────────────────────────────────
    print("\n[OOD Test] Non-traffic question:")
    ood = rag.run("ما هي عاصمة المغرب؟", "llama3.1-8b")
    print(f"  OOD={ood['ood']}, Answer: {ood['answer']}")

    # ── Launch Gradio ────────────────────────────────────────────────────────
    ui = build_ui(rag)
    # Local launch is more reliable on restricted networks.
    port = find_free_port(7860, 7890)
    print(f"[UI] Launching Gradio on http://127.0.0.1:{port}")
    ui.launch(share=False, server_port=port)
