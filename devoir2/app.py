"""Gradio web UI for the Code de la Route RAG assistant.

    python app.py                      # http://127.0.0.1:7860
    SHARE=1 python app.py              # public gradio link (demo only)
"""
from __future__ import annotations

import os
import socket

import gradio as gr

from rag import config
from rag.llm import available_models
from rag.pipeline import RAGPipeline

PIPE = RAGPipeline()
MODELS = available_models()
DEFAULT = config.DEFAULT_MODEL if config.DEFAULT_MODEL in MODELS else (MODELS[0] if MODELS else "")

EXAMPLES = [
    "ما هي غرامة عدم احترام الضوء الأحمر؟",
    "كم نقطة تخصم عند استعمال الهاتف أثناء السياقة؟",
    "ما عقوبة السياقة في حالة سكر؟",
    "شحال الخلاص ديال الضو الحمر؟",
    "Quel est l'âge minimum pour passer le permis B ?",
    "كيف يمكنني الطعن في مخالفة سير؟",
]

CSS = """
.rtl textarea, .rtl .prose {direction: rtl; text-align: right;}
footer {display: none !important;}
"""


def _sources_md(res) -> str:
    if not res.in_domain:
        return "_لا توجد مراجع: السؤال خارج النطاق._"
    lines = []
    for s in res.sources:
        crumb = s["breadcrumb"].split(" > ")[-1] if s["breadcrumb"] else ""
        lines.append(f"- **المادة {s['article']}** — {crumb} (ص {s['page']}, score {s['score']})")
    if res.rewritten and res.rewritten != res.question:
        lines.append(f"\n<sub>صياغة البحث: {res.rewritten}</sub>")
    t = res.timings
    lines.append(f"\n<sub>⏱ {res.latency}s — analyse {t.get('analyze', 0)}s · recherche {t.get('retrieve', 0)}s · "
                 f"génération {t.get('generate', 0)}s · {res.model}</sub>")
    return "\n".join(lines)


def ask(question: str, model: str):
    if not question or not question.strip():
        return "الرجاء كتابة سؤال.", ""
    res = PIPE.answer(question.strip(), model=model or DEFAULT)
    return res.answer, _sources_md(res)


def compare(question: str):
    if not question or not question.strip():
        return [""] * 3
    prep = PIPE.prepare(question.strip())
    outs = []
    for m in [m for m in MODELS if m != "gpt-oss-20b"][:3]:
        r = PIPE.answer(question.strip(), model=m, prepared=prep)
        outs.append(f"### {m}\n<sub>⏱ {r.timings.get('generate', 0)}s</sub>\n\n{r.answer}")
    return outs + [""] * (3 - len(outs))


with gr.Blocks(title="مساعد مدونة السير — Code de la route", css=CSS, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚦 مساعد مدونة السير على الطرق (القانون 52.05)\n"
                "أسئلة بالعربية، بالدارجة أو بالفرنسية. الأجوبة مبنية حصريا على نص القانون مع ذكر المواد.")
    with gr.Tab("💬 سؤال"):
        with gr.Row():
            with gr.Column(scale=3):
                q = gr.Textbox(label="السؤال / Question", lines=2, elem_classes="rtl",
                               placeholder="مثال: ما هي عقوبة السياقة بدون رخصة؟")
                with gr.Row():
                    model = gr.Dropdown(MODELS, value=DEFAULT, label="LLM")
                    btn = gr.Button("إرسال", variant="primary")
                gr.Examples(EXAMPLES, inputs=q)
            with gr.Column(scale=4):
                ans = gr.Markdown(elem_classes="rtl", label="الجواب")
                src = gr.Markdown(elem_classes="rtl")
        btn.click(ask, [q, model], [ans, src])
        q.submit(ask, [q, model], [ans, src])
    with gr.Tab("⚖️ مقارنة النماذج"):
        q2 = gr.Textbox(label="نفس السؤال لكل النماذج", lines=2, elem_classes="rtl")
        btn2 = gr.Button("قارن", variant="primary")
        with gr.Row():
            o1, o2, o3 = gr.Markdown(elem_classes="rtl"), gr.Markdown(elem_classes="rtl"), gr.Markdown(elem_classes="rtl")
        btn2.click(compare, [q2], [o1, o2, o3])
    gr.Markdown("<sub>⚠️ أداة معلوماتية وليست استشارة قانونية. النص المرجعي: الصيغة الموطدة للقانون 52.05 بتاريخ 10 يوليو 2024.</sub>")


def _free_port(start=7860, end=7890) -> int:
    for p in range(start, end):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise OSError("no free port")


if __name__ == "__main__":
    demo.queue().launch(server_name=os.getenv("HOST", "127.0.0.1"), server_port=_free_port(),
                        share=os.getenv("SHARE") == "1")
