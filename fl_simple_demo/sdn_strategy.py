"""
Estrategias FL integradas com SDN para eficiencia de recursos.

SDNBagging  — Bagging com selecao de clientes baseada em metricas de rede
SDNCycling  — Cycling com selecao adaptativa do proximo cliente

Ambas consultam o ODL (ou mock) antes de cada round para:
  1. Obter metricas de rede de cada cliente
  2. Filtrar clientes com rede insuficiente
  3. Selecionar os mais eficientes via efficiency_score
  4. Aplicar QoS para os clientes selecionados
  5. Adaptar epocas locais conforme condicao de rede
  6. Logar metricas de rede no CSV

Uso:
    python server.py --model xgboost --strategy sdn-bagging
    python server.py --model xgboost --strategy sdn-cycling
"""

import csv
import os
import pickle
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    log_loss, matthews_corrcoef, balanced_accuracy_score, cohen_kappa_score,
    brier_score_loss, average_precision_score, confusion_matrix,
)

import flwr as fl
from flwr.common import Parameters, Scalar, FitRes, FitIns
from flwr.server.strategy import Strategy
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager

from config import (
    NUM_CLIENTS, NUM_ROUNDS, LOCAL_EPOCHS,
    LOCAL_EPOCHS_BY_CAT, CLIENT_CATEGORIES,
    SDN_ADAPTIVE_EPOCHS,
)
from sdn_utils import (
    get_network_metrics_from_sdn,
    filter_eligible_clients,
    apply_qos_policy_via_sdn,
    remove_qos_policies,
    adapt_local_epochs,
)


# ---------------------------------------------------------------------------
# Logging SDN — CSV separado para metricas de rede
# ---------------------------------------------------------------------------

_EXP_NAME = os.environ.get("EXP", "experimento")
_SDN_LOG_FILE = f"{_EXP_NAME}_sdn_metricas.csv"
_sdn_log_rows: List[Dict] = []


def _log_sdn_round(
    server_round: int,
    selected_clients: List[int],
    all_metrics: Dict[int, Dict[str, float]],
    scores: Dict[int, float],
    adapted_epochs: Dict[int, int],
) -> None:
    """Registra metricas de rede por cliente por round em CSV separado."""
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
            "adapted_epochs": adapted_epochs.get(cid, LOCAL_EPOCHS),
            "selected": True,
        }
        _sdn_log_rows.append(row)

    # Escreve CSV completo (resiliente a crashes)
    if _sdn_log_rows:
        with open(_SDN_LOG_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(_sdn_log_rows[0].keys()))
            writer.writeheader()
            writer.writerows(_sdn_log_rows)


# ---------------------------------------------------------------------------
# Utilidades compartilhadas
# ---------------------------------------------------------------------------

def deserialize_model(raw_bytes: bytes):
    return pickle.loads(raw_bytes)


def print_metrics(prefix: str, y_true, y_pred, y_prob):
    acc    = accuracy_score(y_true, y_pred)
    prec   = precision_score(y_true, y_pred, zero_division=0)
    rec    = recall_score(y_true, y_pred, zero_division=0)
    f1     = f1_score(y_true, y_pred, zero_division=0)
    auc    = roc_auc_score(y_true, y_prob)
    loglss = log_loss(y_true, y_prob)
    mcc    = matthews_corrcoef(y_true, y_pred)
    bal_ac = balanced_accuracy_score(y_true, y_pred)
    kappa  = cohen_kappa_score(y_true, y_pred)
    brier  = brier_score_loss(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec   = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print(f"  {prefix}")
    print(f"    Accuracy          = {acc:.4f}")
    print(f"    Balanced Accuracy = {bal_ac:.4f}")
    print(f"    Precision         = {prec:.4f}")
    print(f"    Recall            = {rec:.4f}")
    print(f"    Specificity       = {spec:.4f}")
    print(f"    F1-Score          = {f1:.4f}")
    print(f"    AUC-ROC           = {auc:.4f}")
    print(f"    PR-AUC            = {pr_auc:.4f}")
    print(f"    Log Loss          = {loglss:.4f}")
    print(f"    Brier Score       = {brier:.4f}")
    print(f"    MCC               = {mcc:.4f}")
    print(f"    Cohen Kappa       = {kappa:.4f}")
    sys.stdout.flush()

    return {
        "accuracy": acc, "balanced_accuracy": bal_ac,
        "precision": prec, "recall": rec, "specificity": spec,
        "f1": f1, "auc": auc, "pr_auc": pr_auc,
        "log_loss": loglss, "brier_score": brier,
        "mcc": mcc, "cohen_kappa": kappa,
    }


# Referencia ao _log_round do server.py (injetado em runtime)
_log_round_fn = None


def set_log_round_fn(fn):
    """Permite que server.py injete sua funcao _log_round."""
    global _log_round_fn
    _log_round_fn = fn


# ---------------------------------------------------------------------------
# SDN Bagging
# ---------------------------------------------------------------------------

class SDNBagging(Strategy):
    """
    Bagging com selecao de clientes baseada em metricas de rede.

    Antes de cada round:
    1. Consulta metricas de rede de todos os clientes via SDN
    2. Filtra clientes com rede insuficiente
    3. Seleciona os N mais eficientes
    4. Aplica QoS para os selecionados
    5. Adapta epocas locais conforme rede
    """

    def __init__(self, num_clients: int, X_test: np.ndarray, y_test: np.ndarray):
        self.num_clients   = num_clients
        self.X_test        = X_test
        self.y_test        = y_test
        self.client_models: Dict[int, object] = {}
        self.best_model    = None

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        return None

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        has_model = self.best_model is not None

        print(f"\n{'#'*60}")
        print(f"# SDN-BAGGING - Round {server_round}/{NUM_ROUNDS}")
        print(f"# Consultando metricas de rede via SDN...")
        print(f"{'#'*60}")
        sys.stdout.flush()

        # Obtem todos os clientes conectados
        clients = client_manager.sample(
            num_clients=self.num_clients,
            min_num_clients=self.num_clients,
        )

        # Mapeia indice sequencial → ClientProxy
        # Usa indice sequencial (0, 1, 2...) para evitar colisoes de hash
        # e problemas com CIDs que nao sao numericos
        client_map = {}
        for i, c in enumerate(clients):
            client_map[i] = c

        # 1. Consulta metricas de rede
        all_client_ids = list(client_map.keys())
        net_metrics = get_network_metrics_from_sdn(all_client_ids)

        # 2. Filtra e calcula scores
        eligible = filter_eligible_clients(net_metrics)

        if not eligible:
            print(f"  [SDN] AVISO: Nenhum cliente elegivel! Usando todos.")
            eligible = {cid: 0.5 for cid in all_client_ids}

        # 3. Seleciona os melhores (todos os elegiveis no Bagging)
        sorted_clients = sorted(eligible.items(), key=lambda x: x[1], reverse=True)
        selected_ids = [cid for cid, _ in sorted_clients]

        print(f"\n  [SDN] Clientes selecionados: {selected_ids}")
        print(f"  [SDN] Warm start: {has_model}")

        # 4. Aplica QoS
        for i, cid in enumerate(selected_ids):
            priority = 1 if i < len(selected_ids) // 2 else 2
            apply_qos_policy_via_sdn(cid, priority)

        # 5. Adapta epocas locais
        adapted_epochs = {}
        for cid in selected_ids:
            cat = CLIENT_CATEGORIES.get(cid, "cat1")
            base_epochs = LOCAL_EPOCHS_BY_CAT.get(cat, LOCAL_EPOCHS)
            if SDN_ADAPTIVE_EPOCHS:
                score = eligible.get(cid, 0.5)
                adapted = adapt_local_epochs(base_epochs, net_metrics.get(cid, {}), score)
            else:
                adapted = base_epochs
            adapted_epochs[cid] = adapted
            print(f"  [SDN] Cliente {cid} ({cat}): {base_epochs} → {adapted} epocas")

        # 6. Log SDN
        _log_sdn_round(server_round, selected_ids, net_metrics, eligible, adapted_epochs)

        sys.stdout.flush()

        # Serializa modelo uma unica vez (reutiliza para todos os clientes)
        if has_model:
            model_bytes = pickle.dumps(self.best_model)
            fit_params = Parameters(tensors=[model_bytes], tensor_type="pickle")
        else:
            model_bytes = None
            fit_params = Parameters(tensors=[], tensor_type="")

        # Prepara FitIns para cada cliente selecionado
        fit_configs = []
        for cid in selected_ids:
            proxy = client_map.get(cid)
            if proxy is None:
                continue

            config = {
                "server_round": server_round,
                "warm_start": has_model,
                "adapted_epochs": adapted_epochs.get(cid, LOCAL_EPOCHS),
                "efficiency_score": eligible.get(cid, 0.5),
            }

            fit_configs.append((proxy, FitIns(parameters=fit_params, config=config)))

        return fit_configs

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        print(f"\n  [SDN-Bagging] Agregando modelos de {len(results)} clientes...")

        self.client_models = {}
        best_acc = -1
        best_cid = -1

        for _, fit_res in results:
            cid   = int(fit_res.metrics.get("client_id", 0))
            model = deserialize_model(fit_res.parameters.tensors[0])
            self.client_models[cid] = model

            t   = fit_res.metrics.get("training_time", 0)
            acc = fit_res.metrics.get("accuracy",      0)
            f1  = fit_res.metrics.get("f1",            0)
            sz  = fit_res.metrics.get("model_size_kb", 0)
            ep  = fit_res.metrics.get("local_epochs",  0)
            print(f"    Cliente {cid}: Acc={acc:.4f} F1={f1:.4f} "
                  f"Tempo={t:.1f}s Modelo={sz:.1f}KB Epocas={ep}")

            if acc > best_acc:
                best_acc = acc
                best_cid = cid

        if best_cid >= 0:
            self.best_model = self.client_models[best_cid]
            print(f"  [SDN-Bagging] Melhor modelo: Cliente {best_cid} (Acc={best_acc:.4f})")

        if failures:
            print(f"  [SDN-Bagging] AVISO: {len(failures)} falha(s)")

        # Remove QoS apos agregacao
        remove_qos_policies(list(self.client_models.keys()))

        sys.stdout.flush()
        return None, {"num_models": len(self.client_models)}

    def configure_evaluate(self, server_round, parameters, client_manager):
        return []

    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}

    def evaluate(self, server_round: int, parameters: Parameters):
        if not self.client_models:
            return None

        preds  = [m.predict_proba(self.X_test)[:, 1] for m in self.client_models.values()]
        y_prob = np.mean(preds, axis=0)
        y_pred = (y_prob >= 0.5).astype(int)

        print(f"\n  [SDN-Bagging] METRICAS ENSEMBLE Round {server_round}/{NUM_ROUNDS} "
              f"({len(self.client_models)} modelos agregados):")
        metrics = print_metrics(
            "Avaliacao no conjunto de teste:", self.y_test, y_pred, y_prob,
        )

        if _log_round_fn:
            _log_round_fn(server_round, metrics)

        return float(1 - metrics["accuracy"]), metrics


# ---------------------------------------------------------------------------
# SDN Cycling
# ---------------------------------------------------------------------------

class SDNCycling(Strategy):
    """
    Cycling com selecao adaptativa do proximo cliente baseada em rede.

    Em vez de round-robin fixo, seleciona o proximo cliente com melhor
    condicao de rede dentre os que ainda nao treinaram neste ciclo.
    """

    def __init__(self, num_clients: int, X_test: np.ndarray, y_test: np.ndarray):
        self.num_clients   = num_clients
        self.X_test        = X_test
        self.y_test        = y_test
        self.current_model = None
        self.trained_this_cycle: List[int] = []  # rastreia quem ja treinou no ciclo

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        return None

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        has_model = self.current_model is not None

        print(f"\n{'#'*60}")
        print(f"# SDN-CYCLING - Round {server_round}/{NUM_ROUNDS}")
        print(f"# Consultando metricas de rede via SDN...")
        print(f"{'#'*60}")
        sys.stdout.flush()

        clients = client_manager.sample(
            num_clients=self.num_clients,
            min_num_clients=self.num_clients,
        )
        clients = sorted(clients, key=lambda c: c.cid)

        # Mapeia indice sequencial → (posicao, proxy)
        client_map = {}
        for i, c in enumerate(clients):
            client_map[i] = (i, c)

        # Reset do ciclo quando todos treinaram
        if len(self.trained_this_cycle) >= self.num_clients:
            print(f"  [SDN-Cycling] Novo ciclo — todos os clientes ja treinaram")
            self.trained_this_cycle = []

        # Candidatos: quem ainda nao treinou neste ciclo
        candidates = [cid for cid in client_map if cid not in self.trained_this_cycle]
        if not candidates:
            candidates = list(client_map.keys())

        # 1. Consulta metricas de rede dos candidatos
        net_metrics = get_network_metrics_from_sdn(candidates)

        # 2. Filtra e calcula scores
        eligible = filter_eligible_clients(net_metrics)

        if not eligible:
            # Fallback: usa o primeiro candidato
            selected_cid = candidates[0]
            score = 0.5
            print(f"  [SDN-Cycling] Nenhum elegivel, fallback para cliente {selected_cid}")
        else:
            # 3. Seleciona o melhor
            selected_cid = max(eligible, key=eligible.get)
            score = eligible[selected_cid]

        self.trained_this_cycle.append(selected_cid)

        print(f"\n  [SDN-Cycling] Cliente selecionado: {selected_cid} (score={score:.4f})")
        print(f"  [SDN-Cycling] Warm start: {has_model}")
        print(f"  [SDN-Cycling] Treinados neste ciclo: {self.trained_this_cycle}")

        # 4. Aplica QoS
        apply_qos_policy_via_sdn(selected_cid, priority_level=1)

        # 5. Adapta epocas
        cat = CLIENT_CATEGORIES.get(selected_cid, "cat1")
        base_epochs = LOCAL_EPOCHS_BY_CAT.get(cat, LOCAL_EPOCHS)
        if SDN_ADAPTIVE_EPOCHS:
            adapted = adapt_local_epochs(base_epochs, net_metrics.get(selected_cid, {}), score)
        else:
            adapted = base_epochs
        print(f"  [SDN-Cycling] Cliente {selected_cid} ({cat}): {base_epochs} → {adapted} epocas")

        # 6. Log SDN
        _log_sdn_round(
            server_round, [selected_cid], net_metrics,
            {selected_cid: score}, {selected_cid: adapted},
        )

        sys.stdout.flush()

        # Prepara FitIns
        idx, proxy = client_map.get(selected_cid, (0, clients[0]))
        config = {
            "server_round": server_round,
            "cycling_position": idx,
            "warm_start": has_model,
            "adapted_epochs": adapted,
            "efficiency_score": score,
        }

        if has_model:
            model_bytes = pickle.dumps(self.current_model)
            fit_params = Parameters(tensors=[model_bytes], tensor_type="pickle")
        else:
            fit_params = Parameters(tensors=[], tensor_type="")

        return [(proxy, FitIns(parameters=fit_params, config=config))]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            print(f"  [SDN-Cycling] AVISO: Nenhum resultado recebido!")
            return None, {}

        _, fit_res         = results[0]
        cid                = int(fit_res.metrics.get("client_id", 0))
        self.current_model = deserialize_model(fit_res.parameters.tensors[0])

        t   = fit_res.metrics.get("training_time", 0)
        acc = fit_res.metrics.get("accuracy",      0)
        f1  = fit_res.metrics.get("f1",            0)
        sz  = fit_res.metrics.get("model_size_kb", 0)
        ep  = fit_res.metrics.get("local_epochs",  0)

        print(f"\n  [SDN-Cycling] Modelo recebido (client_id={cid}):")
        print(f"    Acc={acc:.4f} F1={f1:.4f} Tempo={t:.1f}s "
              f"Modelo={sz:.1f}KB Epocas={ep}")
        sys.stdout.flush()

        # Remove QoS
        remove_qos_policies([cid])

        return None, {"trained_client": cid}

    def configure_evaluate(self, server_round, parameters, client_manager):
        return []

    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}

    def evaluate(self, server_round: int, parameters: Parameters):
        if self.current_model is None:
            return None

        y_prob = self.current_model.predict_proba(self.X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        print(f"\n  [SDN-Cycling] METRICAS Round {server_round}/{NUM_ROUNDS}:")
        metrics = print_metrics(
            "Avaliacao no conjunto de teste:", self.y_test, y_pred, y_prob,
        )

        if _log_round_fn:
            _log_round_fn(server_round, metrics)

        return float(1 - metrics["accuracy"]), metrics
