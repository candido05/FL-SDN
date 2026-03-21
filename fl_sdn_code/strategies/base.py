"""
Template Method Pattern para estrategias FL.

BaseStrategy implementa:
  - evaluate()       — Template Method: prediz, computa metricas, loga
  - configure_evaluate() / aggregate_evaluate() — noop (avaliacao e server-side)
  - initialize_parameters() — retorna None (clientes treinam do zero no round 1)

Subclasses devem implementar:
  - _predict(X_test)     — retorna (y_prob, y_pred) ou (None, None)
  - configure_fit(...)   — configura quais clientes treinam
  - aggregate_fit(...)   — agrega resultados dos clientes
  - _eval_label(round)   — label para impressao (ex: "ENSEMBLE Round 3/20")
"""

import sys
from abc import abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np

from flwr.common import Parameters, Scalar
from flwr.server.strategy import Strategy
from flwr.server.client_manager import ClientManager

from core.metrics import compute_all_metrics, print_metrics_table


class BaseStrategy(Strategy):
    """Classe base com Template Method para evaluate()."""

    def __init__(self, num_clients: int, X_test: np.ndarray, y_test: np.ndarray):
        self.num_clients = num_clients
        self.X_test = X_test
        self.y_test = y_test
        self._logger = None  # CSVLogger, injetado via set_logger()

    def set_logger(self, logger) -> None:
        """Injeta o CSVLogger do server.py. Repassa run_dir para subclasses."""
        self._logger = logger
        self._on_logger_set(logger)

    def _on_logger_set(self, logger) -> None:
        """Hook para subclasses reagirem ao logger (ex: criar SDNMetricsLogger)."""
        pass

    # ------------------------------------------------------------------
    # Template Method: evaluate()
    # ------------------------------------------------------------------

    def evaluate(self, server_round: int, parameters: Parameters):
        """
        Template Method: prediz → computa metricas → loga → retorna loss.

        Subclasses customizam via _predict() e _eval_label().
        """
        y_prob, y_pred = self._predict(self.X_test)
        if y_prob is None:
            return None

        metrics = compute_all_metrics(self.y_test, y_pred, y_prob)

        print(f"\n  {self._eval_label(server_round)}")
        print_metrics_table("Avaliacao no conjunto de teste:", metrics)

        if self._logger:
            self._logger.log_round(server_round, metrics)

        # Retorna dict sem chaves internas (_tp, _fp, etc.)
        public_metrics = {k: v for k, v in metrics.items() if not k.startswith("_")}
        return float(1 - metrics["accuracy"]), public_metrics

    @abstractmethod
    def _predict(self, X_test) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Retorna (y_prob, y_pred) ou (None, None) se nao ha modelo."""
        ...

    @abstractmethod
    def _eval_label(self, server_round: int) -> str:
        """Label para impressao no evaluate()."""
        ...

    # ------------------------------------------------------------------
    # Metodos com implementacao padrao (noop)
    # ------------------------------------------------------------------

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        return None

    def configure_evaluate(self, server_round, parameters, client_manager):
        return []

    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}
