"""
Logger CSV para metricas por round.

Gera dois tipos de CSV dentro do diretorio de execucao (run_dir):
  - resultados.csv         — metricas de modelo por round
  - sdn_metricas.csv       — metricas de rede por cliente por round

O CSV e reescrito completo a cada round para ser legivel mesmo se
o experimento travar no meio.
"""

import csv
import os
import sys
import time
from typing import Dict, List

from core.metrics import CSV_METRIC_FIELDS

# Campos de recursos e rede agregados por round
RESOURCE_FIELDS = [
    "training_time_avg", "model_size_kb_avg",
    "cpu_percent_avg", "ram_mb_avg", "ram_peak_mb_max",
]

NETWORK_FIELDS = [
    "bandwidth_mbps_avg", "latency_ms_avg", "packet_loss_avg",
    "jitter_ms_avg", "efficiency_score_avg",
]


class CSVLogger:
    """Logger de metricas por round em CSV incremental."""

    _CSV_FIELDS = (
        ["round", "elapsed_sec"]
        + CSV_METRIC_FIELDS
        + RESOURCE_FIELDS
        + NETWORK_FIELDS
    )

    def __init__(self, run_dir: str, exp_name: str = None):
        self._exp_name = exp_name or os.environ.get("EXP", "experimento")
        self._run_dir = run_dir
        self._log_file = os.path.join(run_dir, f"{self._exp_name}_resultados.csv")
        self._t_start = 0.0
        self._rows: List[Dict] = []

    @property
    def log_file(self) -> str:
        return self._log_file

    @property
    def exp_name(self) -> str:
        return self._exp_name

    @property
    def run_dir(self) -> str:
        return self._run_dir

    def start_timer(self) -> None:
        """Inicia o cronometro. Chamar antes de start_server()."""
        self._t_start = time.time()

    def log_round(
        self,
        server_round: int,
        metrics: Dict,
        resource_metrics: Dict = None,
        network_metrics: Dict = None,
    ) -> None:
        """
        Registra metricas do round atual no CSV e imprime resumo.

        Args:
            metrics: Metricas de modelo (accuracy, f1, auc, ...).
            resource_metrics: Metricas de recursos agregadas dos clientes
                (training_time_avg, cpu_percent_avg, ram_mb_avg, ...).
            network_metrics: Metricas de rede agregadas
                (bandwidth_mbps_avg, latency_ms_avg, ...).
        """
        elapsed = round(time.time() - self._t_start, 2)
        row = {"round": server_round, "elapsed_sec": elapsed}
        for field in CSV_METRIC_FIELDS:
            row[field] = round(metrics.get(field, 0.0), 4)

        # Recursos
        rm = resource_metrics or {}
        for field in RESOURCE_FIELDS:
            row[field] = round(rm.get(field, 0.0), 2)

        # Rede
        nm = network_metrics or {}
        for field in NETWORK_FIELDS:
            row[field] = round(nm.get(field, 0.0), 4)

        self._rows.append(row)

        print(f"  [LOG] Round {server_round:2d} | {elapsed:7.1f}s | "
              f"acc={row['accuracy']:.4f} | f1={row['f1']:.4f} | "
              f"auc={row['auc']:.4f} | mcc={row['mcc']:.4f} | "
              f"kappa={row['cohen_kappa']:.4f}")
        if rm:
            print(f"        CPU={row['cpu_percent_avg']:.1f}% | "
                  f"RAM={row['ram_mb_avg']:.1f}MB | "
                  f"Tempo={row['training_time_avg']:.1f}s")
        if nm:
            print(f"        BW={row['bandwidth_mbps_avg']:.1f}Mbps | "
                  f"Lat={row['latency_ms_avg']:.1f}ms | "
                  f"Loss={row['packet_loss_avg']:.4f}")
        sys.stdout.flush()

        with open(self._log_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._CSV_FIELDS)
            writer.writeheader()
            writer.writerows(self._rows)

    def total_elapsed(self) -> float:
        return round(time.time() - self._t_start, 1)


class SDNMetricsLogger:
    """Logger de metricas de rede SDN por cliente por round."""

    def __init__(self, run_dir: str, exp_name: str = None):
        self._exp_name = exp_name or os.environ.get("EXP", "experimento")
        self._run_dir = run_dir
        self._log_file = os.path.join(run_dir, f"{self._exp_name}_sdn_metricas.csv")
        self._health_file = os.path.join(run_dir, f"{self._exp_name}_health_scores.csv")
        self._rows: List[Dict] = []
        self._health_rows: List[Dict] = []

    @property
    def log_file(self) -> str:
        return self._log_file

    def log_round(
        self,
        server_round: int,
        selected_clients: List[int],
        all_metrics: Dict[int, Dict[str, float]],
        scores: Dict[int, float],
        adapted_epochs: Dict[int, int],
        default_epochs: int,
    ) -> None:
        """Registra metricas de rede por cliente por round."""
        for cid in selected_clients:
            m = all_metrics.get(cid, {})
            row = {
                "round": server_round,
                "client_id": cid,
                "bandwidth_mbps": round(m.get("bandwidth_mbps", 0), 2),
                "latency_ms": round(m.get("latency_ms", 0), 2),
                "packet_loss": round(m.get("packet_loss", 0), 4),
                "jitter_ms": round(m.get("jitter_ms", 0), 2),
                "efficiency_score": round(scores.get(cid, 0), 4),
                "adapted_epochs": adapted_epochs.get(cid, default_epochs),
                "selected": True,
            }
            self._rows.append(row)

        if self._rows:
            with open(self._log_file, "w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=list(self._rows[0].keys()),
                )
                writer.writeheader()
                writer.writerows(self._rows)

    _HEALTH_FIELDS = [
        "round", "client_id", "health_score",
        "contribution_score", "resource_score", "network_score",
        "excluded",
    ]

    def log_health_scores(
        self,
        server_round: int,
        scores: Dict[int, Dict],
    ) -> None:
        """Registra health scores por cliente por round."""
        for cid, info in sorted(scores.items()):
            row = {
                "round": server_round,
                "client_id": cid,
                "health_score": round(info.get("health_score", 0), 4),
                "contribution_score": round(info.get("contribution_score", 0), 4),
                "resource_score": round(info.get("resource_score", 0), 4),
                "network_score": round(info.get("network_score", 0), 4),
                "excluded": info.get("excluded", False),
            }
            self._health_rows.append(row)

        if self._health_rows:
            with open(self._health_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._HEALTH_FIELDS)
                writer.writeheader()
                writer.writerows(self._health_rows)
