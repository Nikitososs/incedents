"""Экспорт обучающей выборки для дообучения модели."""

import json
from pathlib import Path

import pandas as pd

from app.config.paths import FINETUNING_PATH
from app.config.schema import SEVERITY_LABEL_COLUMN
from app.io.load_training import load_training_set
from app.severity.anchors import get_severity_labels


def compose_text(row: pd.Series) -> str:
    parts = [
        str(row[col]).strip()
        for col in ("group", "theme", "incident_text")
        if col in row and pd.notna(row[col]) and str(row[col]).strip()
    ]
    return ". ".join(parts)


def export_finetuning_dataset(output_path: Path = FINETUNING_PATH) -> Path:
    df = load_training_set()
    labels = get_severity_labels()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as file:
        for _, row in df.iterrows():
            text = compose_text(row)
            if not text:
                continue
            record = {
                "text": text,
                "label": int(row["severity"]),
                "label_name": row.get(SEVERITY_LABEL_COLUMN) or labels[int(row["severity"])],
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"Экспортировано {count} примеров → {output_path.name}")
    print(df["severity"].value_counts().sort_index().to_string())
    return output_path


if __name__ == "__main__":
    export_finetuning_dataset()
