from pathlib import Path

packages = [
    "backend",
    "backend/app",
    "backend/app/domain",
    "backend/app/application",
    "backend/app/ports",
    "backend/app/adapters",
    "backend/app/adapters/api",
    "backend/app/adapters/database",
    "backend/app/adapters/storage",
    "backend/app/adapters/inference",
    "backend/app/pipeline",
    "ml",
    "ml/datasets",
    "ml/training",
    "ml/evaluation",
]

for folder in packages:
    path = Path(folder)
    path.mkdir(parents=True, exist_ok=True)
    (path / "__init__.py").touch(exist_ok=True)

for folder in [
    "frontend",
    "contracts",
    "tests",
    "notebooks",
    "ml/configs",
    "data/raw",
    "data/processed",
    "artifacts/models",
    "artifacts/metrics",
]:
    path = Path(folder)
    path.mkdir(parents=True, exist_ok=True)
    (path / ".gitkeep").touch(exist_ok=True)

for filename in [
    "README.md",
    ".env.example",
    "requirements.txt",
    "contracts/api.md",
]:
    Path(filename).touch(exist_ok=True)

print("Готово! Структура проекта создана.")
