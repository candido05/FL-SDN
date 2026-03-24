"""
Gerenciamento do diretorio de saida para logs e graficos.

Cada execucao cria um subdiretorio em output/ com timestamp:
    output/2026-03-21_14-30-00_xgboost_bagging/
        ├── resultados.csv
        ├── sdn_metricas.csv
        ├── metricas_fl_sdn.png
        ├── duracao_por_round.png
        └── ...

O plot_resultados.py tambem pode receber --run-dir para apontar
para um diretorio especifico.
"""

import os
from datetime import datetime

_BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")


def create_run_dir(model: str, strategy: str, exp_name: str = "") -> str:
    """
    Cria e retorna o caminho de um diretorio de execucao com timestamp.

    Formato: output/YYYY-MM-DD_HH-MM-SS_<model>_<strategy>[_<exp>]/

    Args:
        model: Nome do modelo (xgboost, lightgbm, catboost).
        strategy: Nome da estrategia (bagging, cycling, etc.).
        exp_name: Nome do experimento (opcional, ex: "com_sdn").

    Returns:
        Caminho absoluto do diretorio criado.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parts = [timestamp, model, strategy]
    if exp_name:
        parts.append(exp_name)
    dirname = "_".join(parts)

    run_dir = os.path.join(_BASE_DIR, dirname)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def get_latest_run_dir() -> str:
    """Retorna o diretorio de execucao mais recente em output/."""
    if not os.path.exists(_BASE_DIR):
        return ""
    dirs = [
        d for d in os.listdir(_BASE_DIR)
        if os.path.isdir(os.path.join(_BASE_DIR, d))
    ]
    if not dirs:
        return ""
    dirs.sort(reverse=True)
    return os.path.join(_BASE_DIR, dirs[0])


def list_run_dirs() -> list:
    """Lista todos os diretorios de execucao (mais recente primeiro)."""
    if not os.path.exists(_BASE_DIR):
        return []
    dirs = [
        d for d in os.listdir(_BASE_DIR)
        if os.path.isdir(os.path.join(_BASE_DIR, d))
    ]
    dirs.sort(reverse=True)
    return [os.path.join(_BASE_DIR, d) for d in dirs]
