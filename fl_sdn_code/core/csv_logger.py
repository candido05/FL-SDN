"""
Logger CSV para metricas de Federated Learning.

Gera os seguintes CSVs dentro do diretorio de execucao (run_dir):
  - {exp}_resultados.csv         — 12 metricas globais por round
  - {exp}_sdn_metricas.csv       — metricas de rede por cliente por round
  - {exp}_health_scores.csv      — health score por cliente por round
  - {exp}_clientes_round.csv     — 12 metricas + confusion matrix por cliente por round
  - {exp}_convergencia.csv       — delta das metricas entre rounds consecutivos

Todos os CSVs sao reescritos completos a cada round — seguros a crash.
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
            print(f"        Tempo={row['training_time_avg']:.1f}s | "
                  f"Modelo={row['model_size_kb_avg']:.1f}KB")
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


# ======================================================================
# ClientRoundLogger — 12 metricas + confusion matrix por cliente por round
# ======================================================================

# Campos completos do CSV por cliente por round
CLIENT_ROUND_FIELDS = [
    "round", "client_id", "model_type", "dataset",
    "local_epochs", "training_time_sec", "model_size_kb",
    "warm_start",
    # 12 metricas de classificacao
    "accuracy", "balanced_accuracy", "precision", "recall",
    "specificity", "f1", "auc", "pr_auc",
    "log_loss", "brier_score", "mcc", "cohen_kappa",
    # Confusion matrix
    "tp", "fp", "tn", "fn",
]


class ClientRoundLogger:
    """
    Registra todas as 12 metricas de classificacao mais metadados por
    cliente por round. Gera {exp}_clientes_round.csv.

    Uso em client.py apos ModelFactory.train():
        logger = ClientRoundLogger(run_dir, exp_name)
        logger.log(server_round, client_id, model_type, dataset,
                   local_epochs, elapsed, model_size_kb, metrics,
                   warm_start=True)
    """

    def __init__(self, run_dir: str, exp_name: str = None):
        self._exp_name = exp_name or os.environ.get("EXP", "experimento")
        self._run_dir = run_dir
        self._log_file = os.path.join(run_dir, f"{self._exp_name}_clientes_round.csv")
        self._rows: List[Dict] = []
        os.makedirs(run_dir, exist_ok=True)

    @property
    def log_file(self) -> str:
        return self._log_file

    def log(
        self,
        server_round: int,
        client_id: int,
        model_type: str,
        dataset: str,
        local_epochs: int,
        training_time_sec: float,
        model_size_kb: float,
        metrics: Dict,
        warm_start: bool = False,
    ) -> None:
        """
        Registra todas as metricas de um cliente em um round.

        Args:
            metrics: Dicionario retornado por compute_all_metrics()
                     (inclui _tp, _fp, _tn, _fn).
        """
        row = {
            "round":            server_round,
            "client_id":        client_id,
            "model_type":       model_type,
            "dataset":          dataset,
            "local_epochs":     local_epochs,
            "training_time_sec": round(training_time_sec, 3),
            "model_size_kb":    round(model_size_kb, 2),
            "warm_start":       warm_start,
        }
        for field in CSV_METRIC_FIELDS:
            row[field] = round(float(metrics.get(field, 0.0)), 6)
        row["tp"] = int(metrics.get("_tp", 0))
        row["fp"] = int(metrics.get("_fp", 0))
        row["tn"] = int(metrics.get("_tn", 0))
        row["fn"] = int(metrics.get("_fn", 0))

        self._rows.append(row)
        self._flush()

        print(
            f"    [LOG-CLIENTE] Round {server_round} | Cliente {client_id} | "
            f"acc={row['accuracy']:.4f} | f1={row['f1']:.4f} | "
            f"auc={row['auc']:.4f} | pr_auc={row['pr_auc']:.4f} | "
            f"mcc={row['mcc']:.4f} | recall={row['recall']:.4f} | "
            f"TP={row['tp']} FP={row['fp']} TN={row['tn']} FN={row['fn']}"
        )
        sys.stdout.flush()

    def _flush(self) -> None:
        with open(self._log_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CLIENT_ROUND_FIELDS)
            writer.writeheader()
            writer.writerows(self._rows)


# ======================================================================
# ConvergenceLogger — delta das metricas entre rounds consecutivos
# ======================================================================

CONVERGENCE_FIELDS = (
    ["round", "elapsed_sec"]
    + CSV_METRIC_FIELDS
    + [f"{m}_delta" for m in CSV_METRIC_FIELDS]
    + ["is_best_accuracy", "is_best_auc", "is_best_f1", "is_best_pr_auc"]
)


class ConvergenceLogger:
    """
    Rastreia a convergencia do modelo global round a round.

    Para cada round registra:
      - As 12 metricas absolutas
      - O delta (variacao) em relacao ao round anterior
      - Flags de melhor resultado historico (is_best_*)

    Gera {exp}_convergencia.csv.

    Uso em strategies/base.py dentro de evaluate():
        self._convergence_logger.log_round(server_round, metrics, elapsed_sec)
    """

    def __init__(self, run_dir: str, exp_name: str = None):
        self._exp_name = exp_name or os.environ.get("EXP", "experimento")
        self._run_dir = run_dir
        self._log_file = os.path.join(
            run_dir, f"{self._exp_name}_convergencia.csv"
        )
        self._rows: List[Dict] = []
        self._prev_metrics: Dict = {}
        self._best: Dict = {m: None for m in ["accuracy", "auc", "f1", "pr_auc"]}
        os.makedirs(run_dir, exist_ok=True)

    @property
    def log_file(self) -> str:
        return self._log_file

    def log_round(
        self,
        server_round: int,
        metrics: Dict,
        elapsed_sec: float = 0.0,
    ) -> None:
        """
        Registra as metricas do round atual e calcula deltas.

        Args:
            metrics: Dicionario retornado por compute_all_metrics().
            elapsed_sec: Tempo acumulado desde o inicio do experimento.
        """
        row: Dict = {"round": server_round, "elapsed_sec": round(elapsed_sec, 2)}

        for m in CSV_METRIC_FIELDS:
            val = float(metrics.get(m, 0.0))
            row[m] = round(val, 6)
            prev = self._prev_metrics.get(m)
            row[f"{m}_delta"] = round(val - prev, 6) if prev is not None else ""

        # Flags de melhor resultado
        for key in ["accuracy", "auc", "f1", "pr_auc"]:
            cur = row[key]
            best = self._best[key]
            is_best = (best is None) or (cur > best)
            row[f"is_best_{key}"] = is_best
            if is_best:
                self._best[key] = cur

        self._rows.append(row)
        self._prev_metrics = {m: row[m] for m in CSV_METRIC_FIELDS}
        self._flush()

        # Imprime deltas relevantes
        delta_acc = row.get("accuracy_delta", "")
        delta_auc = row.get("auc_delta", "")
        delta_f1  = row.get("f1_delta", "")
        delta_pra = row.get("pr_auc_delta", "")
        if delta_acc != "":
            sign = lambda d: "+" if d >= 0 else ""
            print(
                f"  [CONVERGENCIA] Round {server_round} | "
                f"acc_delta={sign(delta_acc)}{delta_acc:.4f} | "
                f"auc_delta={sign(delta_auc)}{delta_auc:.4f} | "
                f"f1_delta={sign(delta_f1)}{delta_f1:.4f} | "
                f"pr_auc_delta={sign(delta_pra)}{delta_pra:.4f}"
            )
        if row.get("is_best_auc"):
            print(f"  [CONVERGENCIA] ** Novo melhor AUC: {row['auc']:.4f} **")
        sys.stdout.flush()

    def _flush(self) -> None:
        with open(self._log_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CONVERGENCE_FIELDS)
            writer.writeheader()
            writer.writerows(self._rows)
