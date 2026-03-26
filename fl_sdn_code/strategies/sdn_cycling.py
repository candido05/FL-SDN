"""Estrategia SDN-Cycling: selecao adaptativa do proximo cliente por rede."""

import pickle
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

from flwr.common import Parameters, Scalar, FitRes, FitIns
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager

from config import (
    NUM_ROUNDS, LOCAL_EPOCHS,
    CLIENT_CATEGORIES,
    SDN_ADAPTIVE_EPOCHS,
    HEALTH_SCORE_ENABLED, HEALTH_SCORE_PROFILE,
    HEALTH_SCORE_CUSTOM_WEIGHTS, HEALTH_SCORE_MAX_EXCLUDE,
    HEALTH_SCORE_MIN_ROUNDS, HEALTH_SCORE_THRESHOLD,
)
from core.serialization import deserialize_model
from core.csv_logger import CSVLogger, SDNMetricsLogger
from core.health_score import ClientHealthTracker
from sdn.network import get_network_metrics, filter_eligible_clients, adapt_local_epochs
from sdn.qos import apply_qos_policy, remove_qos_policies
from sdn.controller import fl_round_start, fl_round_stop
from strategies.base import BaseStrategy


class SDNCycling(BaseStrategy):
    """
    Cycling com selecao adaptativa do proximo cliente baseada em rede.

    Em vez de round-robin fixo, seleciona o proximo cliente com melhor
    condicao de rede dentre os que ainda nao treinaram neste ciclo.
    """

    def __init__(self, num_clients: int, X_test: np.ndarray, y_test: np.ndarray):
        super().__init__(num_clients, X_test, y_test)
        self.current_model = None
        self.trained_this_cycle: List[int] = []
        self._sdn_logger = None
        self._last_net_metrics: Dict = {}  # reutilizado no aggregate_fit (evita double-fetch)

        # Health Score tracker
        if HEALTH_SCORE_ENABLED:
            self._health_tracker = ClientHealthTracker(
                profile=HEALTH_SCORE_PROFILE,
                custom_weights=HEALTH_SCORE_CUSTOM_WEIGHTS,
                max_exclude=HEALTH_SCORE_MAX_EXCLUDE,
                min_rounds_before_exclude=HEALTH_SCORE_MIN_ROUNDS,
                exclude_threshold=HEALTH_SCORE_THRESHOLD,
            )
            print(f"  [Health Score] Perfil: {HEALTH_SCORE_PROFILE} | "
                  f"Pesos: {self._health_tracker.weights}")
        else:
            self._health_tracker = None

    def _on_logger_set(self, logger: CSVLogger) -> None:
        """Cria SDNMetricsLogger usando run_dir do CSVLogger."""
        self._sdn_logger = SDNMetricsLogger(logger.run_dir, logger.exp_name)

    # -- Template Method hooks --

    def _predict(self, X_test):
        if self.current_model is None:
            return None, None
        y_prob = self.current_model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        return y_prob, y_pred

    def _eval_label(self, server_round: int) -> str:
        return f"[SDN-Cycling] METRICAS Round {server_round}/{NUM_ROUNDS}:"

    # -- Flower interface --

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        has_model = self.current_model is not None

        print(f"\n{'#'*60}")
        print(f"# SDN-CYCLING - Round {server_round}/{NUM_ROUNDS}")
        print(f"# Aguardando {self.num_clients} cliente(s)...")
        print(f"{'#'*60}")
        sys.stdout.flush()

        fl_round_start(server_round)

        clients = client_manager.sample(
            num_clients=self.num_clients,
            min_num_clients=self.num_clients,
        )
        clients = sorted(clients, key=lambda c: c.cid)

        client_map = {i: (i, c) for i, c in enumerate(clients)}

        # Reset do ciclo
        if len(self.trained_this_cycle) >= self.num_clients:
            print(f"  [SDN-Cycling] Novo ciclo — todos os clientes ja treinaram")
            self.trained_this_cycle = []

        # Candidatos: quem nao treinou neste ciclo
        candidates = [cid for cid in client_map if cid not in self.trained_this_cycle]
        if not candidates:
            candidates = list(client_map.keys())

        print(f"# Consultando metricas de rede via SDN...")
        sys.stdout.flush()

        # 1. Consulta metricas de rede
        net_metrics = get_network_metrics(candidates)
        self._last_net_metrics = net_metrics  # cache para aggregate_fit

        # 2. Filtra e calcula scores
        eligible = filter_eligible_clients(net_metrics)

        # Exclui clientes via Health Score (se habilitado)
        excluded_ids = []
        if self._health_tracker and eligible:
            excluded_ids = self._health_tracker.get_excluded_clients(
                list(eligible.keys()),
            )
            if excluded_ids:
                print(f"  [Health Score] Excluindo clientes: {excluded_ids}")
                for cid in excluded_ids:
                    eligible.pop(cid, None)

        if not eligible:
            selected_cid = candidates[0]
            score = 0.5
            print(f"  [SDN-Cycling] Nenhum elegivel, fallback para cliente {selected_cid}")
        else:
            selected_cid = max(eligible, key=eligible.get)
            score = eligible[selected_cid]

        self.trained_this_cycle.append(selected_cid)

        print(f"\n  [SDN-Cycling] Cliente selecionado: {selected_cid} (score={score:.4f})")
        print(f"  [SDN-Cycling] Warm start: {has_model}")
        print(f"  [SDN-Cycling] Treinados neste ciclo: {self.trained_this_cycle}")

        # 4. Aplica QoS
        apply_qos_policy(selected_cid, priority_level=1)

        # 5. Adapta epocas
        cat = CLIENT_CATEGORIES.get(selected_cid, "cat1")
        if SDN_ADAPTIVE_EPOCHS:
            adapted = adapt_local_epochs(
                LOCAL_EPOCHS, net_metrics.get(selected_cid, {}), score,
            )
            print(f"  [SDN-Cycling] Cliente {selected_cid} ({cat}): {LOCAL_EPOCHS} → {adapted} epocas")
        else:
            adapted = 0
            print(f"  [SDN-Cycling] Cliente {selected_cid} ({cat}): {LOCAL_EPOCHS} epocas")

        # 6. Agrega metricas de rede para o CSV principal
        self._last_network_metrics = self._aggregate_network_metrics(
            [selected_cid], net_metrics, {selected_cid: score},
        )

        # 7. Log SDN
        if self._sdn_logger:
            self._sdn_logger.log_round(
                server_round, [selected_cid], net_metrics,
                {selected_cid: score}, {selected_cid: adapted}, LOCAL_EPOCHS,
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

        _, fit_res = results[0]
        cid = int(fit_res.metrics.get("client_id", 0))
        self.current_model = deserialize_model(fit_res.parameters.tensors[0])

        t = fit_res.metrics.get("training_time", 0)
        acc = fit_res.metrics.get("accuracy", 0)
        f1 = fit_res.metrics.get("f1", 0)
        sz = fit_res.metrics.get("model_size_kb", 0)
        ep = fit_res.metrics.get("local_epochs", 0)

        print(f"\n  [SDN-Cycling] Modelo recebido (client_id={cid}):")
        print(f"    Acc={acc:.4f} F1={f1:.4f} Tempo={t:.1f}s "
              f"Modelo={sz:.1f}KB Epocas={ep}")
        sys.stdout.flush()

        remove_qos_policies([cid])

        self._aggregate_resource_metrics(results)

        # Health Score: atualiza tracker com resultados do round
        if self._health_tracker:
            client_results = {
                cid: {
                    "accuracy": acc,
                    "f1": f1,
                    "training_time": t,
                    "model_size_kb": sz,
                },
            }

            # Reutiliza metricas de rede ja coletadas no configure_fit (evita timeout duplo)
            net_metrics = {cid: self._last_net_metrics[cid]} if cid in self._last_net_metrics else get_network_metrics([cid])
            net_scores = filter_eligible_clients(net_metrics)

            scores = self._health_tracker.update_round(
                server_round, client_results, net_metrics, net_scores,
            )

            for c, info in scores.items():
                status = "EXCLUIDO" if info["excluded"] else "OK"
                print(f"    [Health Score] Cliente {c}: health={info['health_score']:.4f} "
                      f"(C={info['contribution_score']:.2f} "
                      f"R={info['resource_score']:.2f} "
                      f"N={info['network_score']:.2f}) [{status}]")

            if self._sdn_logger:
                self._sdn_logger.log_health_scores(server_round, scores)

        fl_round_stop()
        return None, {"trained_client": cid}
