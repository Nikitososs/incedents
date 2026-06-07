from pathlib import Path
from typing import Union

import pandas as pd
from openpyxl.utils import column_index_from_string

COLUMN_MAP: dict[str, str] = {
    "R": "created_at",
    "S": "closed_at",
    "T": "group",
    "U": "theme",
    "V": "region",
    "W": "municipality",
    "X": "settlement",
    "AI": "incident_text",
}

DATE_COLUMNS = ("created_at", "closed_at")
TEXT_COLUMNS = ("group", "theme", "region", "municipality", "settlement", "incident_text")


def _column_indices() -> list[int]:
    return [column_index_from_string(letter) - 1 for letter in COLUMN_MAP]


def load_incidents(
    path: Union[str, Path],
    *,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Загружает столбцы R, S, T, U, V, W, X, AI из Excel-файла обращений."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    if path.suffix.lower() != ".xlsx":
        raise ValueError(f"Ожидается .xlsx, получено: {path.suffix}")

    df = pd.read_excel(
        path,
        usecols=_column_indices(),
        nrows=nrows,
        engine="openpyxl",
    )
    df.columns = list(COLUMN_MAP.values())

    for col in DATE_COLUMNS:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in TEXT_COLUMNS:
        df[col] = df[col].astype("string").str.strip()

    return df.reset_index(drop=True)
