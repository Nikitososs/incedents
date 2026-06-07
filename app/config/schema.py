import pandas as pd

SEVERITY_COLUMN = "severity"
SEVERITY_LABEL_COLUMN = "severity_label"
SEVERITY_CONFIDENCE_COLUMN = "severity_confidence"
IS_PROBLEM_COLUMN = "is_problem"
SEVERITY_SOURCE_COLUMN = "severity_source"

SEVERITY_COLUMNS = (
    SEVERITY_COLUMN,
    SEVERITY_LABEL_COLUMN,
    SEVERITY_CONFIDENCE_COLUMN,
    IS_PROBLEM_COLUMN,
)

VALID_SEVERITY = frozenset(range(5))

EXPORT_HEADERS_RU: dict[str, str] = {
    "created_at": "Дата создания",
    "closed_at": "Дата окончания",
    "group": "Группа тем",
    "theme": "Тема",
    "region": "Регион",
    "municipality": "Муниципалитет",
    "settlement": "Населённый пункт",
    SEVERITY_COLUMN: "Степень тяжести",
    SEVERITY_LABEL_COLUMN: "Уровень тяжести",
    SEVERITY_CONFIDENCE_COLUMN: "Уверенность",
    IS_PROBLEM_COLUMN: "Проблема",
    SEVERITY_SOURCE_COLUMN: "Источник разметки",
    "incident_text": "Текст инцидента",
}


def has_severity(df: pd.DataFrame) -> bool:
    return SEVERITY_COLUMN in df.columns and df[SEVERITY_COLUMN].notna().any()


def normalize_severity(value) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        level = int(value)
    except (TypeError, ValueError):
        return None
    return level if level in VALID_SEVERITY else None


def apply_severity_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[SEVERITY_COLUMN] = out[SEVERITY_COLUMN].map(normalize_severity)
    return out
