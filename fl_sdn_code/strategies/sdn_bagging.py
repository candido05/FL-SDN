"""Estrategia SDN-Bagging: selecao de clientes por metricas de rede."""

import pickle
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

from flwr.common import Parameters, Scalar, FitRes, FitIns
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager

from config import (
    NUM_ROUNDS, LOCAL_EPOCHS,
    LOCAL_EPOCHS_BY_CAT, CLIENT_CATEGORIES,
    SDN_ADAPTIVE_EPOCHS,
    HEALTH_SCORE_ENABLED, HEALTH_SCORE_PROFILE,
    HEALTH_SCORE_CUSTOM_WEIGHTS, HEALTH_SCORE_MAX_EXCLUDE,
    HEALTH_SCORE_MIN_ROUNDS, HEALTH_SCORE_THRESHOLD,
)
from core.serialization import deserialize_model
from core.csv_logger import CSVLogger, SDNMetricsLogger
from core.health_score import ClientHealthTracker, compute_leave_one_out
from sdn.network import get_network_metrics, filter_eligible_clients, adapt_local_epochs
from sdn.qos import apply_qos_policy, remove_qos_policies
from strategies.base import BaseStrategy


class SDNBagging(BaseStrategy):
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
        super().__init__(num_clients, X_test, y_test)
        self.client_models: Dict[int, object] = {}
        self.best_model = None
        self._sdn_logger = None

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
        if not self.client_models:
            return None, None
        preds = [m.predict_proba(X_test)[:, 1] for m in self.client_models.values()]
        y_prob = np.mean(preds, axis=0)
        y_pred = (y_prob >= 0.5).astype(int)
        return y_prob, y_pred

    def _eval_label(self, server_round: int) -> str:
        return (f"[SDN-Bagging] METRICAS ENSEMBLE Round {server_round}/{NUM_ROUNDS} "
                f"({len(self.client_models)} modelos agregados):")

    # -- Flower interface --

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

        clients = client_manager.sample(
            num_clients=self.num_clients,
            min_num_clients=self.num_clients,
        )

        # Mapeia indice sequencial para evitar colisoes de hash
        client_map = {i: c for i, c in enumerate(clients)}
        all_client_ids = list(client_map.keys())

        # 1. Consulta metricas de rede
        net_metrics = get_network_metrics(all_client_ids)

        # 2. Filtra e calcula scores
        eligible = filter_eligible_clients(net_metrics)
        if not eligible:
            print(f"  [SDN] AVISO: Nenhum cliente elegivel! Usando todos.")
            eligible = {cid: 0.5 for cid in all_client_ids}

        # 3. Exclui clientes via Health Score (se habilitado)
        excluded_ids = []
        if self._health_tracker:
            excluded_ids = self._health_tracker.get_excluded_clients(
                list(eligible.keys()),
            )
            if excluded_ids:
                print(f"  [Health Score] Excluindo clientes: {excluded_ids}")
                for cid in excluded_ids:
                    eligible.pop(cid, None)

        # 4. Seleciona os melhores dentre os elegiveis restantes
        sorted_clients = sorted(eligible.items(), key=lambda x: x[1], reverse=True)
        selected_ids = [cid for cid, _ in sorted_clients]

        print(f"\n  [SDN] Clientes selecionados: {selected_ids}")
        if excluded_ids:
            print(f"  [SDN] Clientes excluidos por Health Score: {excluded_ids}")
        print(f"  [SDN] Warm start: {has_model}")

        # 4. Aplica QoS baseado na categoria do cliente
        # cat1 → EF(46): modelos pequenos, trafego frequente, prioridade alta
        # cat2 → AF31(26): modelos medios, prioridade media
        # cat3 → BE(0): modelos grandes, tolerantes a atraso
        _CAT_TO_PRIORITY = {"cat1": 1, "cat2": 2, "cat3": 3}
        for cid in selected_ids:
            cat = CLIENT_CATEGORIES.get(cid, "cat1")
            priority = _CAT_TO_PRIORITY.get(cat, 2)
            apply_qos_policy(cid, priority)

        # 5. Adapta epocas locais
        # NOTA: quando SDN_ADAPTIVE_EPOCHS=False, enviamos adapted_epochs=0
        # para que o CLIENTE use suas proprias epocas por categoria.
        # Isso evita o bug de mapeamento posicao→client_id do client_manager.sample().
        adapted_epochs = {}
        for cid in selected_ids:
            cat = CLIENT_CATEGORIES.get(cid, "cat1")
            base_epochs = LOCAL_EPOCHS_BY_CAT.get(cat, LOCAL_EPOCHS)
            if SDN_ADAPTIVE_EPOCHS:
                score = eligible.get(cid, 0.5)
                adapted = adapt_local_epochs(
                    base_epochs, net_metrics.get(cid, {}), score,
                )
                adapted_epochs[cid] = adapted
                print(f"  [SDN] Cliente {cid} ({cat}): {base_epochs} → {adapted} epocas")
            else:
                # Nao adapta: cliente usara suas proprias epocas por categoria
                adapted_epochs[cid] = 0
                print(f"  [SDN] Cliente {cid} ({cat}): {base_epochs} epocas (definido pelo cliente)")

        # 6. Agrega metricas de rede para o CSV principal
        self._last_network_metrics = self._aggregate_network_metrics(
            selected_ids, net_metrics, eligible,
        )

        # 7. Log SDN
        if self._sdn_logger is None:
            print("  [SDN-Bagging] AVISO: SDNMetricsLogger nao configurado (set_logger nao chamado)")
        if self._sdn_logger:
            self._sdn_logger.log_round(
                server_round, selected_ids, net_metrics,
                eligible, adapted_epochs, LOCAL_EPOCHS,
            )
        sys.stdout.flush()

        # Serializa modelo uma unica vez
        if has_model:
            model_bytes = pickle.dumps(self.best_model)
            fit_params = Parameters(tensors=[model_bytes], tensor_type="pickle")
        else:
            fit_params = Parameters(tensors=[], tensor_type="")

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
            cid = int(fit_res.metrics.get("client_id", 0))
            model = deserialize_model(fit_res.parameters.tensors[0])
            self.client_models[cid] = model

            t = fit_res.metrics.get("training_time", 0)
            acc = fit_res.metrics.get("accuracy", 0)
            f1 = fit_res.metrics.get("f1", 0)
            sz = fit_res.metrics.get("model_size_kb", 0)
            ep = fit_res.metrics.get("local_epochs", 0)
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

        remove_qos_policies(list(self.client_models.keys()))

        self._aggregate_resource_metrics(results)

        # Health Score: atualiza tracker com resultados do round
        if self._health_tracker and self.client_models:
            # Coleta metricas por cliente a partir dos FitRes
            client_results = {}
            for _, fit_res in results:
                cid = int(fit_res.metrics.get("client_id", 0))
                client_results[cid] = {
                    "accuracy": fit_res.metrics.get("accuracy", 0),
                    "f1": fit_res.metrics.get("f1", 0),
                    "training_time": fit_res.metrics.get("training_time", 0),
                    "cpu_percent": fit_res.metrics.get("cpu_percent", 0),
                    "ram_mb": fit_res.metrics.get("ram_mb", 0),
                    "model_size_kb": fit_res.metrics.get("model_size_kb", 0),
                }

            # Leave-one-out para medir contribuicao de cada cliente
            ensemble_acc, contributions = compute_leave_one_out(
                self.client_models, self.X_test, self.y_test,
            )

            # Recupera metricas de rede do configure_fit
            net_metrics = get_network_metrics(list(client_results.keys()))
            net_scores = filter_eligible_clients(net_metrics)

            scores = self._health_tracker.update_round(
                server_round, client_results, net_metrics, net_scores,
                ensemble_acc, contributions,
            )

            # Log dos scores
            print(f"\n  [Health Score] Round {server_round} — Perfil: {self._health_tracker.profile_name}")
            for cid, info in sorted(scores.items()):
                status = "EXCLUIDO" if info["excluded"] else "OK"
                print(f"    Cliente {cid}: health={info['health_score']:.4f} "
                      f"(C={info['contribution_score']:.2f} "
                      f"R={info['resource_score']:.2f} "
                      f"N={info['network_score']:.2f}) [{status}]")

            # Log health score no SDNMetricsLogger
            if self._sdn_logger:
                self._sdn_logger.log_health_scores(server_round, scores)

        sys.stdout.flush()
        return None, {"num_models": len(self.client_models)}
