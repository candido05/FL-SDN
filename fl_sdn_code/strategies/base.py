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
        self._last_resource_metrics: Dict = {}
        self._last_network_metrics: Dict = {}

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
            self._logger.log_round(
                server_round, metrics,
                resource_metrics=self._last_resource_metrics,
                network_metrics=self._last_network_metrics,
            )

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

    # ------------------------------------------------------------------
    # Helper: agrega metricas de recursos dos clientes
    # ------------------------------------------------------------------

    def _aggregate_resource_metrics(self, results) -> Dict:
        """
        Agrega metricas de recursos dos FitRes recebidos dos clientes.

        Chamado pelas subclasses no aggregate_fit() para popular
        self._last_resource_metrics antes do evaluate().
        """
        times, sizes, cpus, rams, ram_peaks = [], [], [], [], []
        for _, fit_res in results:
            m = fit_res.metrics
            times.append(m.get("training_time", 0))
            sizes.append(m.get("model_size_kb", 0))
            cpus.append(m.get("cpu_percent", 0))
            rams.append(m.get("ram_mb", 0))
            ram_peaks.append(m.get("ram_peak_mb", 0))

        def _avg(lst):
            return sum(lst) / len(lst) if lst else 0.0

        self._last_resource_metrics = {
            "training_time_avg": _avg(times),
            "model_size_kb_avg": _avg(sizes),
            "cpu_percent_avg": _avg(cpus),
            "ram_mb_avg": _avg(rams),
            "ram_peak_mb_max": max(ram_peaks) if ram_peaks else 0.0,
        }
        return self._last_resource_metrics

    def _aggregate_network_metrics(
        self,
        selected_ids: list,
        net_metrics: Dict,
        scores: Dict,
    ) -> Dict:
        """
        Agrega metricas de rede dos clientes selecionados no round.

        Chamado pelas SDN strategies no configure_fit().
        """
        bws, lats, losses, jitters, eff_scores = [], [], [], [], []
        for cid in selected_ids:
            m = net_metrics.get(cid, {})
            bws.append(m.get("bandwidth_mbps", 0))
            lats.append(m.get("latency_ms", 0))
            losses.append(m.get("packet_loss", 0))
            jitters.append(m.get("jitter_ms", 0))
            eff_scores.append(scores.get(cid, 0))

        def _avg(lst):
            return sum(lst) / len(lst) if lst else 0.0

        self._last_network_metrics = {
            "bandwidth_mbps_avg": _avg(bws),
            "latency_ms_avg": _avg(lats),
            "packet_loss_avg": _avg(losses),
            "jitter_ms_avg": _avg(jitters),
            "efficiency_score_avg": _avg(eff_scores),
        }
        return self._last_network_metrics
