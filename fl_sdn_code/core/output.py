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

import json
import os
import sys
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


# ---------------------------------------------------------------------------
# Registro da execucao do servidor
# ---------------------------------------------------------------------------

RUN_INFO_FILE = "execucao.json"
SERVER_LOG_FILE = "servidor.log"


def write_run_info(run_dir: str, info: dict) -> str:
    """
    Grava {run_dir}/execucao.json de forma atomica (tmp + rename).

    E o contrato com os scripts do laboratorio (sdn_v1/scripts/clientes_fl.sh
    e validar_fl.sh): quem conecta os clientes le daqui modelo, dataset e
    epocas locais do servidor em execucao, em vez de repeti-los a mao.
    """
    path = os.path.join(run_dir, RUN_INFO_FILE)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, path)
    return path


class _Tee:
    """Escreve em um stream e num arquivo ao mesmo tempo."""

    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh

    def write(self, data):
        self._stream.write(data)
        self._fh.write(data)
        return len(data)

    def flush(self):
        self._stream.flush()
        self._fh.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def tee_output(run_dir: str) -> str:
    """
    Duplica stdout/stderr em {run_dir}/servidor.log. O servidor roda no
    terminal do operador (acompanhamento ao vivo); o arquivo e o que a
    validacao da rodada le depois.
    """
    path = os.path.join(run_dir, SERVER_LOG_FILE)
    fh = open(path, "a", buffering=1, encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, fh)
    sys.stderr = _Tee(sys.stderr, fh)
    return path
