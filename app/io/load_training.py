"""Загрузка обучающей выборки (размеченный основной файл)."""

from pathlib import Path
from typing import Union

import pandas as pd

from app.config.paths import get_training_file, get_training_path
from app.config.schema import (
    IS_PROBLEM_COLUMN,
    SEVERITY_COLUMN,
    SEVERITY_CONFIDENCE_COLUMN,
    SEVERITY_LABEL_COLUMN,
    SEVERITY_SOURCE_COLUMN,
    apply_severity_column,
    has_severity,
)

_RU_TO_INTERNAL = {
    "Степень тяжести": SEVERITY_COLUMN,
    "Уровень тяжести": SEVERITY_LABEL_COLUMN,
    "Уверенность": SEVERITY_CONFIDENCE_COLUMN,
    "Проблема": IS_PROBLEM_COLUMN,
    "Источник разметки": SEVERITY_SOURCE_COLUMN,
    "Дата создания": "created_at",
    "Дата окончания": "closed_at",
    "Группа тем": "group",
    "Тема": "theme",
    "Регион": "region",
    "Муниципалитет": "municipality",
    "Населённый пункт": "settlement",
    "Текст инцидента": "incident_text",
}


def load_training_set(path: Union[str, Path] | None = None) -> pd.DataFrame:
    """Загружает обучающую выборку — только для якорей и дообучения."""
    file_path = Path(path) if path else get_training_file()
    if file_path is None:
        expected = get_training_path()
        raise FileNotFoundError(
            f"Обучающая выборка не найдена: {expected}\n"
            "Сначала разметьте основной файл:\n"
            "  python -m app.labeling.training_set"
        )

    df = pd.read_excel(file_path, engine="openpyxl")
    df = df.rename(columns={
        ru: internal for ru, internal in _RU_TO_INTERNAL.items() if ru in df.columns
    })

    if not has_severity(df):
        raise ValueError(f"В обучающей выборке нет колонки «{SEVERITY_COLUMN}»")

    df = apply_severity_column(df)
    return df[df[SEVERITY_COLUMN].notna()].reset_index(drop=True)
