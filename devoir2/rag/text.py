"""Arabic text normalization, tokenization and light stemming for lexical retrieval."""
from __future__ import annotations

import re

_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
_TATWEEL = "ـ"
_NON_WORD = re.compile(r"[^\w\s]|_")

# Words that carry no retrieval signal in legal questions.
STOPWORDS = set(
    """
    في من على إلى الى عن مع أو او و ثم أن ان إن لا لم لن ما ماذا هل كم كيف متى أين اين لماذا
    هو هي هم هن هذا هذه ذلك تلك الذي التي الذين اللذين اللتين كل بعض أي اي غير بين عند حتى
    قد كان كانت يكون تكون ليس به بها له لها لهم فيه فيها منه منها عليه عليها اليه إليه إليها
    أعلاه بعده أدناه المادة المواد هذا القانون قانون نص النصوص الصادرة لتطبيقه السير مدونة
    يجب يمكن يجوز حالة الحالات وفق طبقا أحكام احكام بموجب المشار المنصوص المذكورة إليها اليها
    هي ما هو هل يا أنا انا نحن انت أنت المغرب المغربي المغربية
    """.split()
)

_PREFIXES = ("وبال", "وال", "بال", "كال", "فال", "لل", "ال", "و", "ب", "ف", "ل")
_SUFFIXES = ("ات", "ون", "ين", "ان", "ها", "هم", "هن", "كم", "نا", "ية", "يه", "ة", "ه", "ي")


def normalize(text: str) -> str:
    """Canonical Arabic form used by every retriever (query and documents alike)."""
    if not isinstance(text, str):
        return ""
    text = _DIACRITICS.sub("", text).replace(_TATWEEL, "")
    text = re.sub("[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي").replace("ة", "ه")
    # Arabic-Indic digits -> ASCII, thousands dot "1.400" -> "1400"
    text = text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    text = re.sub(r"(?<=\d)\.(?=\d{3}\b)", "", text)
    return re.sub(r"\s+", " ", text).strip()


def light_stem(token: str) -> str:
    """Light10-style stemmer: strip one common prefix and one common suffix."""
    if token.isdigit() or len(token) <= 3:
        return token
    for p in _PREFIXES:
        if token.startswith(p) and len(token) - len(p) >= 3:
            token = token[len(p):]
            break
    for s in _SUFFIXES:
        if token.endswith(s) and len(token) - len(s) >= 3:
            token = token[: -len(s)]
            break
    return token


def tokenize(text: str, stem: bool = True) -> list[str]:
    text = _NON_WORD.sub(" ", normalize(text).lower())
    toks = [t for t in text.split() if t not in STOPWORDS and len(t) > 1]
    return [light_stem(t) for t in toks] if stem else toks
