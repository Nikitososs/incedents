from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from app.severity.anchors import get_severity_labels, get_severity_levels

DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
TEXT_COLUMN = "incident_text"


@dataclass(frozen=True)
class SeverityResult:
    severity: int
    label: str
    confidence: float
    is_problem: bool
    scores: dict[int, float]


class SeverityClassifier:
    """Классификатор тяжести через cosine similarity к эталонным фразам."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        device: str | None = None,
        batch_size: int = 64,
        temperature: float = 0.05,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self.temperature = temperature
        self._levels = get_severity_levels()
        self._labels = get_severity_labels()
        self._model = SentenceTransformer(model_name, device=device)
        self._prototypes = self._build_prototypes()

    def _encode(self, texts: Iterable[str]) -> np.ndarray:
        embeddings = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def _build_prototypes(self) -> np.ndarray:
        levels = sorted(self._levels)
        prototypes = []
        for level in levels:
            anchors = self._levels[level].anchors
            anchor_embeddings = self._encode(anchors)
            centroid = anchor_embeddings.mean(axis=0)
            centroid /= np.linalg.norm(centroid) + 1e-12
            prototypes.append(centroid)
        return np.vstack(prototypes)

    def _compose_text(
        self,
        text: str,
        *,
        group: str | None = None,
        theme: str | None = None,
    ) -> str:
        parts = [part.strip() for part in (group, theme, text) if part and str(part).strip()]
        return ". ".join(parts)

    def _predict_from_embeddings(self, embeddings: np.ndarray) -> list[SeverityResult]:
        levels = sorted(self._levels)
        similarities = embeddings @ self._prototypes.T
        scaled = similarities / self.temperature
        scaled -= scaled.max(axis=1, keepdims=True)
        probabilities = np.exp(scaled)
        probabilities /= probabilities.sum(axis=1, keepdims=True)

        results: list[SeverityResult] = []
        for idx in range(len(embeddings)):
            severity = int(levels[int(np.argmax(probabilities[idx]))])
            scores = {
                level: float(probabilities[idx][level_idx])
                for level_idx, level in enumerate(levels)
            }
            results.append(
                SeverityResult(
                    severity=severity,
                    label=self._labels[severity],
                    confidence=float(scores[severity]),
                    is_problem=severity >= 1,
                    scores=scores,
                )
            )
        return results

    def classify_text(
        self,
        text: str,
        *,
        group: str | None = None,
        theme: str | None = None,
    ) -> SeverityResult:
        composed = self._compose_text(text, group=group, theme=theme)
        embedding = self._encode([composed])
        return self._predict_from_embeddings(embedding)[0]

    def classify_texts(
        self,
        texts: list[str],
        *,
        groups: list[str | None] | None = None,
        themes: list[str | None] | None = None,
    ) -> list[SeverityResult]:
        if groups is None:
            groups = [None] * len(texts)
        if themes is None:
            themes = [None] * len(texts)

        composed = [
            self._compose_text(text, group=group, theme=theme)
            for text, group, theme in zip(texts, groups, themes)
        ]
        embeddings = self._encode(composed)
        return self._predict_from_embeddings(embeddings)

    def classify_dataframe(
        self,
        df: pd.DataFrame,
        *,
        text_column: str = TEXT_COLUMN,
        group_column: str = "group",
        theme_column: str = "theme",
    ) -> pd.DataFrame:
        if text_column not in df.columns:
            raise KeyError(f"Колонка не найдена: {text_column}")

        texts = df[text_column].astype(str).tolist()
        groups = df[group_column].tolist() if group_column in df.columns else None
        themes = df[theme_column].tolist() if theme_column in df.columns else None

        results = self.classify_texts(texts, groups=groups, themes=themes)

        out = df.copy()
        out["severity"] = [item.severity for item in results]
        out["severity_label"] = [item.label for item in results]
        out["severity_confidence"] = [item.confidence for item in results]
        out["is_problem"] = [item.is_problem for item in results]
        return out


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from app.config.paths import get_test_file
    from app.io.load import load_incidents
    from app.preprocess.clean import clean_incidents

    file_path = Path(sys.argv[1]) if len(sys.argv) > 1 else get_test_file()
    sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else 200

    df = clean_incidents(load_incidents(file_path)).head(sample_size)
    classifier = SeverityClassifier()
    labeled = classifier.classify_dataframe(df)

    print(f"Файл: {file_path.name}")
    print(f"Классифицировано: {len(labeled)}")
    print(labeled["severity"].value_counts().sort_index().to_string())
    print()
    for severity in sorted(labeled["severity"].unique()):
        row = labeled.loc[labeled["severity"] == severity].iloc[0]
        print(f"[{severity}] {row['severity_label']} ({row['severity_confidence']:.2f})")
        print(str(row["incident_text"])[:220])
        print()
