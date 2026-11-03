"""Command-line entry point for the Code de la Route RAG assistant.

    python rag_juridique.py "ما هي غرامة عدم احترام الضوء الأحمر؟"
    python rag_juridique.py --model qwen2.5-7b-local "Que risque-t-on sans permis ?"
    python rag_juridique.py            # interactive mode

The implementation lives in the `rag/` package (extraction, retrieval, pipeline);
the web UI is `app.py`, the evaluation is in `eval/`.
"""
from __future__ import annotations

import argparse

from rag import config
from rag.llm import available_models
from rag.pipeline import RAGPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="*")
    ap.add_argument("--model", default=config.DEFAULT_MODEL, choices=available_models() or None)
    args = ap.parse_args()
    pipe = RAGPipeline()

    def run(q: str):
        r = pipe.answer(q, model=args.model)
        print("\n" + r.answer)
        if r.sources:
            print("\nالمراجع: " + "، ".join(f"المادة {s['article']}" for s in r.sources))
        print(f"⏱ {r.latency}s {r.timings}\n")

    if args.question:
        run(" ".join(args.question))
        return
    while True:
        try:
            q = input("سؤال> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q:
            run(q)


if __name__ == "__main__":
    main()
