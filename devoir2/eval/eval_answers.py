"""End-to-end answer evaluation (retrieval + generation) and LLM comparison.

Three resumable phases (Groq free tier has daily token caps, so runs may need to resume):
  1. generate : answers for every (model, gold question)      -> results/answers.jsonl
  2. judge    : LLM judge per answer, never by its own family   -> results/judgments.jsonl
  3. report   : metrics                                         -> results/summary.json

Metrics per model:
  * answer_accuracy : LLM judge says correct (reference = gold articles + retrieved law text)
  * facts_match     : every key fact (amount, points, duration...) appears verbatim (deterministic)
  * cites_gold      : the answer cites a gold article
Retrieval: Hit@5, OOD refusal rate (14 off-topic questions), false refusals (75 in-domain).

    python -m eval.eval_answers --models gpt-oss-120b qwen3.8-27b allam-2-7b qwen2.5-7b-local
    python -m eval.eval_answers --report-only
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from rag import config
from rag.llm import QuotaExceeded, available_models, chat
from rag.pipeline import RAGPipeline
from rag.text import normalize

from .gold_questions import GOLD, OOD

OUT = Path(__file__).parent / "results"
ANSWERS = OUT / "answers.jsonl"
JUDGMENTS = OUT / "judgments.jsonl"
JUDGE_CHAIN = ["gemini", "gpt-oss-120b", "gpt-oss-20b", "qwen3.8-27b"]

JUDGE_PROMPT = """You grade answers of a legal assistant about the Moroccan traffic code (law 52.05).

Question: {q}

Authoritative law text (the gold article(s) first, then other articles the assistant was given):
{ref}

Key facts a correct answer must contain: {facts}

Assistant answer:
{ans}

The answer is CORRECT if it answers the question, its key figures (amounts, points, durations, ages) match the law text,
and it states nothing that contradicts the law text. Extra details are fine if the law text above supports them.
Language (Arabic or French) does not matter.
Return ONLY JSON: {{"correct": true|false, "reason": "<short>"}}"""


def _norm_num(text: str) -> str:
    t = normalize(text)
    return re.sub(r"(?<=\d)[.,\s  ](?=\d{3}\b)", "", t)


def facts_ok(answer: str, facts: list[str]) -> bool:
    a = _norm_num(answer).lower()
    return all(any(_norm_num(alt).lower() in a for alt in f.split("|")) for f in facts)


def _family(model: str) -> str:
    return model.split("-")[0].replace("qwen2.5", "qwen").replace("qwen3.8", "qwen")


def _load(path: Path) -> dict:
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[(r["model"], r["question"])] = r
    return out


def _append(path: Path, row: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ── phase 1 ──────────────────────────────────────────────────────────────────
def generate(models: list[str], analysis_cache: dict):
    pipe = RAGPipeline(fallback=False)  # measure each model on its own
    done = _load(ANSWERS)
    prepared = {}
    for _, q, *_ in GOLD:
        prepared[q] = pipe.prepare(q, analysis=analysis_cache.get(q))
    ood = {q: pipe.prepare(q, analysis=analysis_cache.get(q)) for _, q in OOD}
    retrieval = {
        "hit@5": round(sum(any(h.article.article_id in g for h in prepared[q]["hits"][:5]) for _, q, g, _ in GOLD) / len(GOLD), 3),
        "ood_refusal_rate": round(sum(not p["in_domain"] for p in ood.values()) / len(OOD), 3),
        "false_refusals": [q for q, p in prepared.items() if not p["in_domain"]],
    }
    (OUT / "retrieval_summary.json").write_text(json.dumps(retrieval, ensure_ascii=False, indent=2), encoding="utf-8")
    print("retrieval:", retrieval, flush=True)
    for model in models:
        for lang, q, g, facts in GOLD:
            if (model, q) in done and not done[(model, q)]["answer"].startswith("[LLM error]"):
                continue
            r = pipe.answer(q, model=model, prepared=prepared[q])
            if "QuotaExceeded" in r.answer or "tokens per day" in r.answer:
                print(f"[{model}] daily quota exhausted — resume later", flush=True)
                break
            row = {"model": model, "lang": lang, "question": q, "gold": g, "facts": facts, "rewritten": r.rewritten,
                   "retrieved": [s["article"] for s in r.sources], "in_domain": r.in_domain, "answer": r.answer,
                   "generate_s": r.timings.get("generate", 0)}
            _append(ANSWERS, row)
            print(f"[{model}] facts={int(facts_ok(r.answer, facts))} {q[:60]}", flush=True)


# ── phase 2 ──────────────────────────────────────────────────────────────────
def judge_all(by_id: dict, only: list[str] | None = None):
    answers, done = _load(ANSWERS), _load(JUDGMENTS)
    if only:
        answers = {k: v for k, v in answers.items() if k[0] in only}
    exhausted: set[str] = set()
    for key, a in answers.items():
        if key in done:
            continue
        if not a["in_domain"]:
            _append(JUDGMENTS, {"model": a["model"], "question": a["question"], "correct": False, "reason": "refused", "judge": None})
            continue
        # Gold articles in FULL (art. 99's points table and art. 185 item 36 sit far past any prefix),
        # then every other article the assistant was given, also in full (answers may cite them).
        others = [i for i in dict.fromkeys(a["retrieved"]) if i not in a["gold"]]
        ref = "\n\n".join([f"[المادة {i}]\n{by_id[i].text}" for i in a["gold"] if i in by_id] +
                           [f"[المادة {i}]\n{by_id[i].text}" for i in others if i in by_id])
        prompt = JUDGE_PROMPT.format(q=a["question"], ref=ref, facts=", ".join(a["facts"]) or "(none)", ans=a["answer"])
        verdict = None
        for jm in JUDGE_CHAIN:
            if jm in exhausted or _family(jm) == _family(a["model"]):
                continue
            try:
                raw = chat(jm, [{"role": "user", "content": prompt}], max_tokens=200)
            except QuotaExceeded:
                exhausted.add(jm)
                continue
            m = re.search(r"\{.*\}", raw, re.S)
            try:
                d = json.loads(m.group(0))
                verdict = {"correct": bool(d.get("correct")), "reason": str(d.get("reason", "")), "judge": jm}
            except Exception:
                verdict = {"correct": False, "reason": f"unparseable: {raw[:80]}", "judge": jm}
            break
        if verdict is None:
            print("all judges out of quota — resume later", flush=True)
            return
        _append(JUDGMENTS, {"model": a["model"], "question": a["question"], **verdict})
        print(f"[{a['model']}] {'✓' if verdict['correct'] else '✗'} {a['question'][:60]}"
              + ("" if verdict["correct"] else f"  <- {verdict['reason'][:100]}"), flush=True)


# ── phase 3 ──────────────────────────────────────────────────────────────────
def report() -> dict:
    answers, judgments = _load(ANSWERS), _load(JUDGMENTS)
    summary = {}
    for model in dict.fromkeys(m for m, _ in answers):
        rows = [a for (m, _), a in answers.items() if m == model]
        js = [judgments[(model, a["question"])] for a in rows if (model, a["question"]) in judgments]
        cites = 0
        for a in rows:
            cited = {c.replace(" ", "") for c in re.findall(r"المادة\s*(\d+(?:\s*-\s*\d+)?)", a["answer"])}
            cites += bool(cited & set(a["gold"]))
        summary[model] = {
            "n_answers": len(rows),
            "answer_accuracy": round(sum(j["correct"] for j in js) / len(js), 3) if js else None,
            "n_judged": len(js),
            "facts_match": round(sum(facts_ok(a["answer"], a["facts"]) for a in rows) / len(rows), 3),
            "cites_gold_article": round(cites / len(rows), 3),
            "avg_generation_s": round(sum(a["generate_s"] for a in rows) / len(rows), 2),
        }
    rs = OUT / "retrieval_summary.json"
    if rs.exists():
        summary["_retrieval"] = json.loads(rs.read_text(encoding="utf-8"))
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--judge-only", nargs="*", default=None, help="only judge existing answers of these models")
    ap.add_argument("--rejudge-failed", action="store_true", help="drop previous negative verdicts first")
    ap.add_argument("--fresh", action="store_true", help="delete previous answers/judgments first")
    ap.add_argument("--analysis-cache", default=str(OUT / "analysis_cache.json"))
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.fresh:
        for p in (ANSWERS, JUDGMENTS):
            p.unlink(missing_ok=True)
    if args.rejudge_failed and JUDGMENTS.exists():
        keep = [l for l in JUDGMENTS.read_text(encoding="utf-8").splitlines() if l.strip() and json.loads(l)["correct"]]
        JUDGMENTS.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8")
    if args.judge_only is not None:
        from rag.retriever import load_articles

        judge_all({a.article_id: a for a in load_articles()}, args.judge_only or None)
    elif not args.report_only:
        cache = json.loads(Path(args.analysis_cache).read_text(encoding="utf-8")) if Path(args.analysis_cache).exists() else {}
        generate(args.models or available_models(), cache)
        if not args.no_judge:
            from rag.retriever import load_articles

            judge_all({a.article_id: a for a in load_articles()})
    report()


if __name__ == "__main__":
    main()
