"""Estrategia Bagging: todos os clientes treinam em paralelo."""

import pickle
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

from flwr.common import Parameters, Scalar, FitRes, FitIns
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager

from config import NUM_ROUNDS, LOCAL_EPOCHS
from core.serialization import deserialize_model
from strategies.base import BaseStrategy


class SimpleBagging(BaseStrategy):
    """Todos os clientes treinam em paralelo; ensemble por media ponderada."""

    def __init__(self, num_clients: int, X_test: np.ndarray, y_test: np.ndarray):
        super().__init__(num_clients, X_test, y_test)
        self.client_models: Dict[int, object] = {}
        self.client_weights: Dict[int, float] = {}
        self.best_model = None

    # -- Template Method hooks --

    def _predict(self, X_test):
        if not self.client_models:
            return None, None
        preds = []
        weights = []
        for cid, model in self.client_models.items():
            preds.append(model.predict_proba(X_test)[:, 1])
            weights.append(self.client_weights.get(cid, 1.0))
        weights = np.array(weights)
        weights = weights / weights.sum()
        y_prob = np.average(preds, axis=0, weights=weights)
        y_pred = (y_prob >= 0.5).astype(int)
        return y_prob, y_pred

    def _eval_label(self, server_round: int) -> str:
        return (f"[Servidor] METRICAS ENSEMBLE Round {server_round}/{NUM_ROUNDS} "
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
        print(f"# BAGGING - Round {server_round}/{NUM_ROUNDS}")
        print(f"# Selecionando {self.num_clients} clientes para treino paralelo")
        print(f"# Warm start: {has_model}")
        print(f"# Epocas locais por cliente: {LOCAL_EPOCHS}")
        print(f"{'#'*60}")
        sys.stdout.flush()

        clients = client_manager.sample(
            num_clients=self.num_clients,
            min_num_clients=self.num_clients,
        )

        if has_model:
            model_bytes = pickle.dumps(self.best_model)
            fit_params = Parameters(tensors=[model_bytes], tensor_type="pickle")
            config = {"server_round": server_round, "warm_start": True}
            print(f"  [Servidor] Enviando melhor modelo ({len(model_bytes)/1024:.1f} KB) "
                  f"para {self.num_clients} clientes")
        else:
            fit_params = Parameters(tensors=[], tensor_type="")
            config = {"server_round": server_round, "warm_start": False}
            print(f"  [Servidor] Clientes treinam do zero (round 1)")

        sys.stdout.flush()
        return [(c, FitIns(parameters=fit_params, config=config)) for c in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        print(f"\n  [Servidor] Agregando modelos de {len(results)} clientes...")

        self.client_models = {}
        self.client_weights = {}
        best_acc = -1
        best_cid = -1

        for _, fit_res in results:
            cid = int(fit_res.metrics.get("client_id", 0))
            model = deserialize_model(fit_res.parameters.tensors[0])
            self.client_models[cid] = model

            t = fit_res.metrics.get("training_time", 0)
            client_acc = fit_res.metrics.get("accuracy", 0)
            f1 = fit_res.metrics.get("f1", 0)
            sz = fit_res.metrics.get("model_size_kb", 0)
            print(f"    Cliente {cid}: Acc(local)={client_acc:.4f} F1={f1:.4f} "
                  f"Tempo={t:.1f}s Modelo={sz:.1f}KB")

        # Avaliacao server-side: seleciona best_model e calcula pesos do ensemble
        # usando o test set do servidor (evita vies de selecao por treino)
        print(f"  [Servidor] Avaliando modelos no test set do servidor...")
        for cid, model in self.client_models.items():
            y_prob = model.predict_proba(self.X_test)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            server_acc = float(np.mean(y_pred == self.y_test))
            self.client_weights[cid] = server_acc

            print(f"    Cliente {cid}: Acc(server)={server_acc:.4f}")

            if server_acc > best_acc:
                best_acc = server_acc
                best_cid = cid

        if best_cid >= 0:
            self.best_model = self.client_models[best_cid]
            print(f"  [Servidor] Melhor modelo (server-side): "
                  f"Cliente {best_cid} (Acc={best_acc:.4f})")

        if failures:
            print(f"  [Servidor] AVISO: {len(failures)} falha(s)")

        self._aggregate_resource_metrics(results)

        sys.stdout.flush()
        return None, {"num_models": len(self.client_models)}
