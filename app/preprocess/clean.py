import html
import re

import pandas as pd

HTML_TAG = re.compile(r"<[^>]+>", re.IGNORECASE)
URL = re.compile(
    r"https?://\S+|www\.\S+|t\.me/\S+|vk\.ru/\S+",
    re.IGNORECASE,
)
EMOJI = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U0001F600-\U0001F64F"
    "\U00002600-\U000026FF"
    "]+",
    flags=re.UNICODE,
)
MEDIA_PLACEHOLDER = re.compile(
    r"\[(?:фото(?:графия)?|видео|изображение|photo|image|video)[^\]]*\]",
    re.IGNORECASE,
)
VK_BBCODE = re.compile(
    r"\[(?:club|id|public)\d+[^\]]*\]",
    re.IGNORECASE,
)
WHITESPACE = re.compile(r"\s+")
REPEATED_PUNCT = re.compile(r"([!?.,:;—-])\1{2,}")
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SURROUNDING_QUOTES = re.compile(r"^['\"`«»]+|['\"`«»]+$")
SPAM_LINE = re.compile(
    r"^(?:.*\b(?:подпишись|подписывайтесь)\b.*|.*@\w[\w.]*.*|.*\bt\.me/\S+.*|.*\bvk\.ru/\S+.*)$",
    re.IGNORECASE,
)

TEXT_COLUMN = "incident_text"
EMPTY_VALUES = frozenset({"", "nan", "<na>", "none"})


def clean_text(text: str | None) -> str | None:
    if text is None or pd.isna(text):
        return None

    cleaned = str(text).strip()
    if cleaned.lower() in EMPTY_VALUES:
        return None

    cleaned = SURROUNDING_QUOTES.sub("", cleaned)
    cleaned = html.unescape(cleaned)
    cleaned = HTML_TAG.sub(" ", cleaned)
    cleaned = URL.sub(" ", cleaned)
    cleaned = EMOJI.sub("", cleaned)
    cleaned = MEDIA_PLACEHOLDER.sub(" ", cleaned)
    cleaned = VK_BBCODE.sub(" ", cleaned)
    cleaned = CONTROL_CHARS.sub("", cleaned)

    lines = []
    for line in cleaned.splitlines():
        line = WHITESPACE.sub(" ", line).strip()
        if not line or SPAM_LINE.match(line):
            continue
        lines.append(line)
    cleaned = " ".join(lines)

    cleaned = REPEATED_PUNCT.sub(r"\1\1", cleaned)
    cleaned = WHITESPACE.sub(" ", cleaned).strip()

    return cleaned or None


def clean_incidents(
    df: pd.DataFrame,
    *,
    text_column: str = TEXT_COLUMN,
    min_length: int | None = 20,
    keep_raw: bool = True,
    drop_empty: bool = True,
) -> pd.DataFrame:
    if text_column not in df.columns:
        raise KeyError(f"Колонка не найдена: {text_column}")

    out = df.copy()
    if keep_raw and "incident_text_raw" not in out.columns:
        out["incident_text_raw"] = out[text_column]

    out[text_column] = out[text_column].map(clean_text)

    if drop_empty:
        out = out[out[text_column].notna()]

    if min_length is not None:
        out = out[out[text_column].str.len() >= min_length]

    return out.reset_index(drop=True)
