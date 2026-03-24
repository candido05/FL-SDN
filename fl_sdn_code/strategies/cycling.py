"""Estrategia Cycling: um cliente por rodada, round-robin sequencial."""

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


class SimpleCycling(BaseStrategy):
    """Um cliente por rodada (round-robin); modelo passa sequencialmente."""

    def __init__(self, num_clients: int, X_test: np.ndarray, y_test: np.ndarray):
        super().__init__(num_clients, X_test, y_test)
        self.current_idx = 0
        self.current_model = None

    # -- Template Method hooks --

    def _predict(self, X_test):
        if self.current_model is None:
            return None, None
        y_prob = self.current_model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        return y_prob, y_pred

    def _eval_label(self, server_round: int) -> str:
        trained_client = (self.current_idx - 1) % self.num_clients
        return (f"[Servidor] METRICAS Round {server_round}/{NUM_ROUNDS} "
                f"(modelo do cliente {trained_client}):")

    # -- Flower interface --

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        has_model = self.current_model is not None

        print(f"\n{'#'*60}")
        print(f"# CYCLING - Round {server_round}/{NUM_ROUNDS}")
        print(f"# Cliente selecionado: {self.current_idx}")
        print(f"# Warm start: {has_model}")
        print(f"# Epocas locais: {LOCAL_EPOCHS}")
        print(f"{'#'*60}")
        sys.stdout.flush()

        clients = client_manager.sample(
            num_clients=self.num_clients,
            min_num_clients=self.num_clients,
        )
        clients = sorted(clients, key=lambda c: c.cid)
        selected = clients[self.current_idx]

        config = {"server_round": server_round, "cycling_position": self.current_idx}
        if has_model:
            model_bytes = pickle.dumps(self.current_model)
            fit_params = Parameters(tensors=[model_bytes], tensor_type="pickle")
            config["warm_start"] = True
            print(f"  [Servidor] Enviando modelo ({len(model_bytes)/1024:.1f} KB) "
                  f"para posicao {self.current_idx} (CID={selected.cid})")
        else:
            fit_params = Parameters(tensors=[], tensor_type="")
            config["warm_start"] = False
            print(f"  [Servidor] Posicao {self.current_idx} (CID={selected.cid}) treina do zero")

        sys.stdout.flush()
        return [(selected, FitIns(parameters=fit_params, config=config))]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            print(f"  [Servidor] AVISO: Nenhum resultado recebido!")
            return None, {}

        _, fit_res = results[0]
        cid = int(fit_res.metrics.get("client_id", self.current_idx))
        self.current_model = deserialize_model(fit_res.parameters.tensors[0])

        t = fit_res.metrics.get("training_time", 0)
        acc = fit_res.metrics.get("accuracy", 0)
        f1 = fit_res.metrics.get("f1", 0)
        sz = fit_res.metrics.get("model_size_kb", 0)

        print(f"\n  [Servidor] Modelo recebido (client_id={cid}, posicao={self.current_idx}):")
        print(f"    Acc={acc:.4f} F1={f1:.4f} Tempo={t:.1f}s Modelo={sz:.1f}KB")
        print(f"  [Servidor] Modelo salvo. Proxima posicao: "
              f"{(self.current_idx + 1) % self.num_clients}")
        sys.stdout.flush()

        self._aggregate_resource_metrics(results)

        prev = self.current_idx
        self.current_idx = (self.current_idx + 1) % self.num_clients
        return None, {"trained_client": prev}
