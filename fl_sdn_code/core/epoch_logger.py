"""
Logger de epocas locais para Federated Learning.

Registra a curva de perda (boosting loss) a cada iteracao local
de cada cliente em cada round de treinamento.

Gera {exp}_epocas_locais.csv em formato tidy (long-format):
  - Uma linha por (round × cliente × epoca_local)
  - Facil de plotar com pandas/matplotlib/seaborn

Schema:
  round, client_id, model_type, dataset, local_epoch,
  train_logloss, val_logloss, elapsed_sec
"""

import csv
import os
import time
from typing import Dict, List, Optional


# Campos do CSV de epocas locais
EPOCH_LOG_FIELDS = [
    "round", "client_id", "model_type", "dataset",
    "local_epoch", "total_epochs",
    "train_logloss", "val_logloss",
    "elapsed_sec",
]


class EpochLogger:
    """
    Registra metricas de treinamento em cada epoch (boosting iteration)
    para cada cliente em cada round.

    Uso em client.py:
        logger = EpochLogger(run_dir, exp_name)
        ...
        model, history = ModelFactory.train(..., epoch_logger=logger)

    O EpochLogger e passado para os callbacks dos modelos, que chamam
    log_epoch() apos cada iteracao de boosting.
    """

    def __init__(self, run_dir: str, exp_name: str = None):
        self._exp_name = exp_name or os.environ.get("EXP", "experimento")
        self._run_dir = run_dir
        self._log_file = os.path.join(run_dir, f"{self._exp_name}_epocas_locais.csv")
        self._rows: List[Dict] = []
        self._t_round_start: Optional[float] = None
        os.makedirs(run_dir, exist_ok=True)

    @property
    def log_file(self) -> str:
        return self._log_file

    def start_round(self) -> None:
        """Inicia cronometro para o round atual."""
        self._t_round_start = time.time()

    def log_epoch(
        self,
        server_round: int,
        client_id: int,
        model_type: str,
        dataset: str,
        local_epoch: int,
        total_epochs: int,
        train_logloss: float,
        val_logloss: Optional[float] = None,
    ) -> None:
        """
        Registra metricas de uma unica epoch de boosting.

        Args:
            server_round:  Round FL atual.
            client_id:     ID do cliente.
            model_type:    "xgboost", "lightgbm" ou "catboost".
            dataset:       Nome do dataset.
            local_epoch:   Numero da iteracao local (1-indexed).
            total_epochs:  Total de epocas planejadas.
            train_logloss: Log-loss no conjunto de treino/validacao interna.
            val_logloss:   Log-loss no conjunto de validacao (None se nao disponivel).
        """
        elapsed = round(time.time() - self._t_round_start, 3) if self._t_round_start else 0.0
        row = {
            "round":        server_round,
            "client_id":    client_id,
            "model_type":   model_type,
            "dataset":      dataset,
            "local_epoch":  local_epoch,
            "total_epochs": total_epochs,
            "train_logloss": round(float(train_logloss), 6),
            "val_logloss":   round(float(val_logloss), 6) if val_logloss is not None else "",
            "elapsed_sec":  elapsed,
        }
        self._rows.append(row)

    def flush(self) -> None:
        """Salva todas as linhas acumuladas no CSV (incremental, append-safe)."""
        if not self._rows:
            return
        # Reescreve inteiro a cada flush para garantir legibilidade apos crash
        with open(self._log_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=EPOCH_LOG_FIELDS)
            writer.writeheader()
            writer.writerows(self._rows)

    def __repr__(self) -> str:
        return f"EpochLogger(file={self._log_file!r}, rows={len(self._rows)})"
