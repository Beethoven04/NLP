"""Extract the Moroccan Code de la Route (law 52.05) PDF into one record per article.

Why this exists
---------------
The devoir-1 CSV was built from a broken extraction:
  * text came out in visual (reversed) order with swapped lam-alef ligatures
    ("االمانه" instead of "الأمانة"),
  * numbers drifted to the wrong side of words (fines / points were corrupted),
  * rows were PAGES (126) instead of ARTICLES (~330), so retrieval could not
    point at the right article.

Here we rebuild every line from glyph coordinates (PyMuPDF rawdict):
  1. sort glyphs right-to-left by x position (zero-width ligature parts after their base),
  2. re-reverse left-to-right runs (digits, latin, "52.05", "1.10.07"),
then segment the text on "المادة N" headers and attach the book/part/chapter/section
headings that precede each article.

Usage:  python -m rag.extract_pdf [pdf_path] [out_jsonl]
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pymupdf

LTR_CHAR = re.compile(r"[0-9A-Za-z]")
LTR_JOINERS = "./,-:"
PAGE_HEADER = "الأمانة العامة للحكومة"

# "المادة 61-1", "المادة 1 - 61", "المادة 137 المكررة", "المادة 12 مكرر"
ARTICLE_HEADER = re.compile(
    r"^المادة\s*(\d+)(?:\s*-\s*(\d+))?(?:\s*(مكرر(?:ة)?|المكرر(?:ة)?))?\s*$"
)
HEADING = re.compile(r"^(الكتاب|القسم|الباب|الفرع|الجزء)\s+\S+")
HEADING_LEVELS = ["الكتاب", "القسم", "الباب", "الفرع", "الجزء"]
AMENDMENT_NOTE = re.compile(r"^\(?\s*(غيرت|غير|تممت|أضيفت|نسخت|نسخ)\b.*?(القانون رقم|ج\.ر)")


def _line_text(line: dict) -> str:
    chars = [c for s in line["spans"] for c in s["chars"]]
    chars.sort(key=lambda c: (-round(c["bbox"][2], 1), (c["bbox"][2] - c["bbox"][0]) < 0.05))
    seq = [c["c"] for c in chars]
    out: list[str] = []
    i = 0
    while i < len(seq):
        if LTR_CHAR.match(seq[i]):
            j = i
            while j + 1 < len(seq) and (
                LTR_CHAR.match(seq[j + 1])
                or (seq[j + 1] in LTR_JOINERS and j + 2 < len(seq) and LTR_CHAR.match(seq[j + 2]))
            ):
                j += 1
            out.extend(reversed(seq[i : j + 1]))
            i = j + 1
        else:
            out.append(seq[i])
            i += 1
    return fix_glyph_artifacts(re.sub(r"\s+", " ", "".join(out)).strip())


# Some composite glyphs in the PDF font decompose into an extra letter
# ("الأقظصى" -> "الأقصى", "بمقتغضى" -> "بمقتضى", "الآشيء المقغضي" -> "الشيء المقضي").
_GLYPH_FIXES = [("قظص", "قص"), ("تغض", "تض"), ("قغض", "قض"), ("الآشيء", "الشيء"), ("ألأقصى", "الأقصى"), ("مغضي", "مضي"), ("حظص", "حص")]


def fix_glyph_artifacts(text: str) -> str:
    for bad, good in _GLYPH_FIXES:
        text = text.replace(bad, good)
    return text


def _cell_text(page, bbox) -> str:
    if not bbox:
        return ""
    rd = page.get_text("rawdict", clip=pymupdf.Rect(bbox))
    return " ".join(_line_text(l) for b in rd["blocks"] for l in b.get("lines", [])).strip()


def _table_lines(page, table) -> list[str]:
    """Render a points table (points | offence | n°) as one readable line per row."""
    rows = [[_cell_text(page, c) for c in r.cells] for r in table.rows]
    if not rows:
        return []
    header = rows[0]
    kind = header[1] if len(header) > 1 else ""
    out = [f"جدول النقط الواجب خصمها من رصيد رخصة السياقة ({kind}) :"]
    for row in rows[1:]:
        if len(row) < 2 or not row[1]:
            continue
        points = re.sub(r"\D", "", row[0])
        out.append(f"• {row[1]} : خصم {points} نقط")  # no row number: LLMs mistook it for an article number
    return out


def extract_lines(pdf_path: str | Path) -> list[tuple[int, str]]:
    """Return (page_number, line) pairs, page headers / page numbers removed.

    Tables are detected with PyMuPDF and re-rendered row by row, because the plain
    line stream interleaves the cells (points end up glued to the offence text).
    """
    doc = pymupdf.open(str(pdf_path))
    lines: list[tuple[int, str]] = []
    for pno, page in enumerate(doc, start=1):
        tables = page.find_tables().tables
        emitted = set()
        for block in page.get_text("rawdict")["blocks"]:
            for line in block.get("lines", []):
                x0, y0, x1, y1 = line["bbox"]
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                hit = next((t for t in tables if pymupdf.Rect(t.bbox).contains(pymupdf.Point(cx, cy))), None)
                if hit is not None:
                    if id(hit) not in emitted:
                        emitted.add(id(hit))
                        lines.extend((pno, t) for t in _table_lines(page, hit))
                    continue
                txt = _line_text(line)
                if not txt or txt == PAGE_HEADER or re.fullmatch(r"\d{1,3}", txt):
                    continue
                lines.append((pno, txt))
    return lines


@dataclass
class Article:
    article_id: str
    number: int
    text: str
    page_start: int
    page_end: int
    headings: dict = field(default_factory=dict)
    amendments: list = field(default_factory=list)

    @property
    def breadcrumb(self) -> str:
        return " > ".join(v for v in self.headings.values() if v)


def segment_articles(lines: list[tuple[int, str]]) -> list[Article]:
    articles: list[Article] = []
    headings: dict[str, str] = {}
    current: Article | None = None
    body: list[str] = []
    pending_heading: str | None = None  # heading keyword line waiting for its title line

    def close():
        nonlocal current, body
        if current is not None:
            text, notes = [], []
            in_note = False
            for ln in body:
                if AMENDMENT_NOTE.match(ln):
                    notes.append(ln)
                    in_note = not re.search(r"ص\s*:?\s*\d+\s*\)\s*$", ln)
                elif in_note:
                    notes[-1] += " " + ln
                    in_note = not re.search(r"\)\s*$", ln)
                else:
                    text.append(ln)
            current.text = _join_lines(text)
            current.amendments = notes
            articles.append(current)
        current, body = None, []

    for pno, ln in lines:
        m = ARTICLE_HEADER.match(ln)
        if m:
            close()
            num, sub, bis = m.group(1), m.group(2), m.group(3)
            if sub and int(sub) > int(num):  # "1-166" printed for 166-1
                num, sub = sub, num
            aid = num + (f"-{sub}" if sub else "") + (" مكرر" if bis else "")
            current = Article(aid, int(num), "", pno, pno, dict(headings))
            pending_heading = None
            continue
        hm = HEADING.match(ln)
        if hm and len(ln.split()) <= 4:
            level = hm.group(1)
            idx = HEADING_LEVELS.index(level)
            for lower in HEADING_LEVELS[idx:]:
                headings.pop(lower, None)
            headings[level] = ln
            pending_heading = level
            close()  # an article never spans a new heading
            continue
        if pending_heading and current is None:
            headings[pending_heading] += " : " + ln
            pending_heading = None
            continue
        if current is not None:
            body.append(ln)
            current.page_end = pno
    close()
    return _dedupe(articles)


def _join_lines(lines: list[str]) -> str:
    """Join wrapped lines; keep a newline after sentence ends and before enumerations."""
    out = ""
    for ln in lines:
        if not out:
            out = ln
        elif re.search(r"[.:؛;]$", out) or re.match(r"^(\d+\s*-|[أ-ي]\s*[)(]|-|•|جدول)", ln):
            out += "\n" + ln
        else:
            out += " " + ln
    return out.strip()


def _dedupe(articles: list[Article]) -> list[Article]:
    """Some article ids appear twice (e.g. abrogated + replaced); merge their text."""
    seen: dict[str, Article] = {}
    for a in articles:
        if a.article_id in seen:
            prev = seen[a.article_id]
            prev.text = (prev.text + "\n" + a.text).strip()
            prev.page_end = max(prev.page_end, a.page_end)
            prev.amendments += a.amendments
        else:
            seen[a.article_id] = a
    return list(seen.values())


def main(pdf: str, out: str) -> list[Article]:
    arts = segment_articles(extract_lines(pdf))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for a in arts:
            rec = asdict(a)
            rec["breadcrumb"] = a.breadcrumb
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"{len(arts)} articles -> {out}")
    return arts


if __name__ == "__main__":
    here = Path(__file__).resolve().parent.parent
    pdf = sys.argv[1] if len(sys.argv) > 1 else str(here.parent / "devoir1" / "code de la route MA52_05.pdf")
    out = sys.argv[2] if len(sys.argv) > 2 else str(here / "data" / "articles.jsonl")
    main(pdf, out)
