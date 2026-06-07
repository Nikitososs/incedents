import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.config.paths import ANCHORS_PATH


@dataclass(frozen=True)
class SeverityLevel:
    level: int
    label: str
    description: str
    anchors: tuple[str, ...]


@lru_cache(maxsize=1)
def get_severity_levels(path: Path | None = None) -> dict[int, SeverityLevel]:
    file_path = path or ANCHORS_PATH
    if not file_path.exists():
        raise FileNotFoundError(
            f"Файл эталонов не найден: {file_path}\n"
            "Сначала соберите якоря:\n"
            "  python -m app.severity.anchor_builder"
        )

    data = json.loads(file_path.read_text(encoding="utf-8"))
    levels: dict[int, SeverityLevel] = {}

    for key, meta in data["levels"].items():
        level = int(key)
        anchors = tuple(meta["anchors"])
        if not anchors:
            raise ValueError(f"Для уровня {level} нет эталонных фраз")
        levels[level] = SeverityLevel(
            level=level,
            label=meta["label"],
            description=meta["description"],
            anchors=anchors,
        )

    return levels


def get_severity_labels() -> dict[int, str]:
    return {level: meta.label for level, meta in get_severity_levels().items()}
