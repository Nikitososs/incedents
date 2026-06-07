import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.config.paths import ANCHORS_PATH, get_main_file, get_training_file, get_training_path
from app.io.load_training import load_training_set

LEVEL_META: dict[int, tuple[str, str]] = {
    0: ("Не инцидент", "Инфо / спам / благодарность / консультация"),
    1: ("Низкая тяжесть", "Эстетика / локальный дискомфорт, угрозы нет"),
    2: ("Средняя тяжесть", "Нарушение нормативов, мешает жить"),
    3: ("Высокая тяжесть", "Угроза имуществу или комфорту многих"),
    4: ("Критическая / ЧС", "Угроза здоровью и жизни"),
}

RULES: dict[int, list[re.Pattern[str]]] = {
    4: [
        re.compile(p, re.IGNORECASE)
        for p in (
            r"запах\s+газ",
            r"(?:утеч|запах|взрыв|запах).*газ|газ.*(?:утеч|запах|взрыв)",
            r"\bпожар\b",
            r"обруш",
            r"угарн",
            r"(?:огол|гол).*провод",
            r"провод.*(?:детск|площад)",
            r"\bвзрыв",
            r"угроз.*(?:жизн|здоров)",
            r"\bэвакуац",
            r"ч\s*\.?\s*с\b",
        )
    ],
    3: [
        re.compile(p, re.IGNORECASE)
        for p in (
            r"отключ.*(?:вод|элект|отоп|газ)",
            r"нет\s+(?:холод|горяч|газ|отоплен|вод)",
            r"затопл",
            r"прор(?:ыв|в)",
            r"кипяток",
            r"(?:открыт|открытый).*люк|люк.*(?:открыт|без\s+ограж)",
            r"(?:не\s+работает|сломан).*(?:оба|ни\s+один).*лифт|(?:оба|ни\s+один).*лифт",
            r"без\s+отоплен",
            r"авар.*(?:тепл|вод|канал)",
            r"подвал.*(?:зал|затоп|залит)",
            r"стояк.*(?:прор|зал)",
            r"канализац.*(?:ль|теч|хлещ)",
            r"(?:весь|всём|всем).*дом.*(?:без|нет).*(?:вод|отоплен|электр)",
        )
    ],
    0: [
        re.compile(p, re.IGNORECASE)
        for p in (
            r"\bспасибо\b",
            r"\bблагодар",
            r"подскажите",
            r"как\s+(?:запис|получ|оформ)",
            r"часы\s+работ",
            r"консультац",
            r"закройте\s+обращ",
            r"дублиру",
            r"ложн.*обращ",
            r"претензий\s+нет",
            r"проблема\s+(?:решена|устранена)",
            r"информ.*справ",
        )
    ],
    1: [
        re.compile(p, re.IGNORECASE)
        for p in (
            r"трав[аы]",
            r"(?:не\s+)?(?:скошен|косил).*трав",
            r"лавоч|скамей",
            r"урн",
            r"граффит",
            r"сорняк|клумб",
            r"облупил.*краск",
            r"эстетик",
            r"фасад.*(?:космет|трещ)",
            r"объявлен.*(?:стенд|доск)",
            r"озеленен",
        )
    ],
}

PROBLEM_HINTS = re.compile(
    r"жалоб|проблем|авар|убрать|исправ|принять\s+мер|не\s+(?:убира|работ|горит|включ)",
    re.IGNORECASE,
)

MIN_ANCHOR_LEN = 60
MAX_ANCHOR_LEN = 400
ANCHORS_PER_LEVEL = 15


def _matches(text: str, patterns: list[re.Pattern[str]]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def assign_severity(text: str) -> int:
    if _matches(text, RULES[4]):
        return 4
    if _matches(text, RULES[3]):
        return 3
    if _matches(text, RULES[0]) and not _matches(text, RULES[3] + RULES[4]):
        if not PROBLEM_HINTS.search(text) or _matches(text, RULES[0]):
            return 0
    if _matches(text, RULES[1]) and not _matches(text, RULES[3] + RULES[4]):
        return 1
    return 2


def _normalize_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())[:100]


def _truncate(text: str, limit: int = MAX_ANCHOR_LEN) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    cut = text[: limit - 1].rsplit(" ", 1)[0]
    return cut + "…"


def select_anchors(df: pd.DataFrame, level: int, count: int = ANCHORS_PER_LEVEL) -> list[str]:
    subset = df.loc[df["severity"] == level, ["theme", "incident_text"]].copy()
    subset["len"] = subset["incident_text"].str.len()
    subset = subset[(subset["len"] >= MIN_ANCHOR_LEN) & (subset["len"] <= 800)]
    if subset.empty:
        return []

    anchors: list[str] = []
    seen: set[str] = set()

    for _, row in subset.sort_values("len").groupby("theme", sort=False).head(1).iterrows():
        text = _truncate(str(row["incident_text"]))
        key = _normalize_key(text)
        if key in seen:
            continue
        seen.add(key)
        anchors.append(text)
        if len(anchors) >= count:
            return anchors

    for _, row in subset.sort_values("len", ascending=False).iterrows():
        text = _truncate(str(row["incident_text"]))
        key = _normalize_key(text)
        if key in seen:
            continue
        seen.add(key)
        anchors.append(text)
        if len(anchors) >= count:
            break

    return anchors


def load_for_anchors() -> tuple[pd.DataFrame, str]:
    training_path = get_training_file()
    if training_path is not None:
        print(f"Обучающая выборка: {training_path.name}")
        return load_training_set(training_path), "training"

    from app.io.load import load_incidents
    from app.preprocess.clean import clean_incidents

    main_path = get_main_file()
    print(f"Обучающая выборка не найдена ({get_training_path().name})")
    print(f"Bootstrap по правилам из: {main_path.name}")

    df = clean_incidents(load_incidents(main_path))
    df["severity"] = df["incident_text"].map(assign_severity)
    return df, "rules"


def build_anchors(*, anchors_per_level: int = ANCHORS_PER_LEVEL) -> dict:
    df, labeling_source = load_for_anchors()
    source_file = (get_training_file() or get_training_path()).name

    print(f"Строк: {len(df)}")
    print(f"Источник разметки: {labeling_source}")
    print(df["severity"].value_counts().sort_index().to_string())

    levels: dict[str, dict] = {}
    for level, (label, description) in LEVEL_META.items():
        anchors = select_anchors(df, level, count=anchors_per_level)
        if len(anchors) < anchors_per_level:
            print(f"  уровень {level}: только {len(anchors)} якорей")
        levels[str(level)] = {
            "level": level,
            "label": label,
            "description": description,
            "anchors": anchors,
        }

    return {
        "source_file": source_file,
        "labeling_source": labeling_source,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows_used": len(df),
        "anchors_per_level": anchors_per_level,
        "levels": levels,
    }


def save_anchors(data: dict, path: Path = ANCHORS_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    payload = build_anchors()
    out = save_anchors(payload)
    print(f"\nСохранено: {out}")
    for level in sorted(payload["levels"], key=int):
        print(f"  [{level}] якорей: {len(payload['levels'][level]['anchors'])}")
