"""
Пути к файлам данных.

Роли файлов:
  data/основной файл.xlsx              — сырьё для обучения
  data/основной файл_размеченный.xlsx  — обучающая выборка (якоря + дообучение)
  data/тестовый файл.xlsx              — только проверка модели
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ANCHORS_PATH = DATA_DIR / "severity_anchors.json"
FINETUNING_PATH = DATA_DIR / "severity_finetuning.jsonl"

TRAINING_SUFFIX = "_размеченный"


def _raw_xlsx_files() -> list[Path]:
    return [
        path
        for path in DATA_DIR.glob("*.xlsx")
        if TRAINING_SUFFIX not in path.stem
    ]


def get_main_file() -> Path:
    files = _raw_xlsx_files()
    if not files:
        raise FileNotFoundError(f"Нет исходных .xlsx в {DATA_DIR}")
    return max(files, key=lambda path: path.stat().st_size)


def get_test_file() -> Path:
    files = _raw_xlsx_files()
    if len(files) < 2:
        raise FileNotFoundError(f"Нужны два исходных .xlsx в {DATA_DIR}")
    return min(files, key=lambda path: path.stat().st_size)


def get_training_path() -> Path:
    main = get_main_file()
    return main.with_name(f"{main.stem}{TRAINING_SUFFIX}{main.suffix}")


def get_training_file() -> Path | None:
    path = get_training_path()
    return path if path.exists() else None


def assert_main_file(path: Path) -> None:
    if path.resolve() == get_test_file().resolve():
        raise ValueError(
            "Тестовый файл нельзя использовать для обучения.\n"
            "Разметка делается только на основном файле:\n"
            f"  {get_main_file().name}"
        )
