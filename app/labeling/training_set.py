"""Создание обучающей выборки из основного файла."""

from pathlib import Path

import pandas as pd

from app.config.paths import get_main_file, get_training_path
from app.config.schema import (
    EXPORT_HEADERS_RU,
    IS_PROBLEM_COLUMN,
    SEVERITY_COLUMN,
    SEVERITY_CONFIDENCE_COLUMN,
    SEVERITY_LABEL_COLUMN,
    SEVERITY_SOURCE_COLUMN,
)
from app.io.load import load_incidents
from app.preprocess.clean import clean_incidents
from app.severity.classifier import SeverityClassifier

OUTPUT_COLUMNS = (
    "created_at",
    "closed_at",
    "group",
    "theme",
    "region",
    "municipality",
    "settlement",
    "incident_text",
    SEVERITY_COLUMN,
    SEVERITY_LABEL_COLUMN,
    SEVERITY_CONFIDENCE_COLUMN,
    IS_PROBLEM_COLUMN,
    SEVERITY_SOURCE_COLUMN,
)


def label_incidents(
    df: pd.DataFrame,
    classifier: SeverityClassifier | None = None,
    *,
    chunk_size: int = 2000,
    source: str = "model",
) -> pd.DataFrame:
    classifier = classifier or SeverityClassifier()
    parts: list[pd.DataFrame] = []

    for start in range(0, len(df), chunk_size):
        chunk = df.iloc[start : start + chunk_size]
        labeled = classifier.classify_dataframe(chunk)
        labeled[SEVERITY_SOURCE_COLUMN] = source
        parts.append(labeled)

    return pd.concat(parts, ignore_index=True)


def build_training_set(
    *,
    nrows: int | None = None,
    chunk_size: int = 2000,
    russian_headers: bool = True,
) -> Path:
    main_path = get_main_file()
    output_path = get_training_path()

    print(f"Основной файл: {main_path.name}")
    print(f"Обучающая выборка → {output_path.name}")

    df = clean_incidents(load_incidents(main_path, nrows=nrows))
    print(f"После очистки: {len(df)} строк")

    print("Предварительная разметка моделью (можно править вручную в Excel)...")
    labeled = label_incidents(df, chunk_size=chunk_size)

    export = labeled[[col for col in OUTPUT_COLUMNS if col in labeled.columns]].copy()
    if russian_headers:
        export = export.rename(columns=EXPORT_HEADERS_RU)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    export.to_excel(output_path, index=False, engine="openpyxl")

    print(f"Сохранено: {output_path.name}")
    print(labeled[SEVERITY_COLUMN].value_counts().sort_index().to_string())
    print("\nДальше:")
    print("  python -m app.severity.anchor_builder")
    print("  python -m app.training.export")
    return output_path


if __name__ == "__main__":
    import sys

    nrows = int(sys.argv[1]) if len(sys.argv) > 1 else None
    build_training_set(nrows=nrows)
