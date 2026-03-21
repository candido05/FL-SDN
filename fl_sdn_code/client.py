"""
Cliente Flower com conexao gRPC explicita.

Uso:
    python client.py --client-id 0 --model xgboost
    python client.py --client-id 1 --model lightgbm

O numero de epocas locais e determinado pela categoria do cliente
(CLIENT_CATEGORIES no config.py), refletindo a heterogeneidade de
hardware (cat1=low, cat2=medium, cat3=high).
"""

import argparse
import logging
import os
import pickle
import sys
import time
import warnings

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*DEPRECATED.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")
warnings.filterwarnings("ignore", message=".*feature names.*")

import numpy as np

import flwr as fl
from flwr.common import (
    Code, Status, FitIns, FitRes, EvaluateIns, EvaluateRes,
    Parameters,
)

logging.getLogger("flwr").setLevel(logging.ERROR)
logging.getLogger("grpc").setLevel(logging.ERROR)

from config import (
    CLIENT_CONNECT_ADDRESS, LOCAL_EPOCHS, LOG_EVERY,
    LOCAL_EPOCHS_BY_CAT, CLIENT_CATEGORIES,
)
from core.metrics import compute_all_metrics, print_metrics_table
from core.resources import ResourceMonitor
from core.serialization import serialize_model
from datasets import DatasetRegistry
from models.factory import ModelFactory


# ---------------------------------------------------------------------------
# Cliente Flower
# ---------------------------------------------------------------------------

class SimpleClient(fl.client.Client):
    """
    Cliente FL que envia/recebe modelos como bytes puros via Parameters.tensors.
    Usa fl.client.Client (nao NumPyClient) para evitar conversoes numpy
    que corrompem a serializacao pickle.
    """

    def __init__(self, client_id: int, model_type: str, local_epochs: int,
                 category: str, X_train, y_train, X_test, y_test):
        self.client_id = client_id
        self.model_type = model_type
        self.local_epochs = local_epochs
        self.category = category
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = None

    def fit(self, ins: FitIns) -> FitRes:
        config = ins.config
        server_round = int(config.get("server_round", 0))
        use_warm = bool(config.get("warm_start", False))

        # Epocas adaptativas: o servidor SDN pode enviar um valor ajustado
        adapted_epochs = int(config.get("adapted_epochs", 0))
        round_epochs = adapted_epochs if adapted_epochs > 0 else self.local_epochs

        eff_score = float(config.get("efficiency_score", 0))

        print(f"\n{'─'*60}")
        print(f"  [Cliente {self.client_id}] INICIO Round {server_round}")
        print(f"  [Cliente {self.client_id}] Modelo: {self.model_type} | "
              f"Categoria: {self.category} | "
              f"Epocas locais: {round_epochs} | Warm start: {use_warm}")
        if adapted_epochs > 0 and adapted_epochs != self.local_epochs:
            print(f"  [Cliente {self.client_id}] Epocas adaptadas pelo SDN: "
                  f"{self.local_epochs} → {round_epochs} (score={eff_score:.4f})")
        print(f"  [Cliente {self.client_id}] Amostras treino: {len(self.X_train)}")
        print(f"{'─'*60}")
        sys.stdout.flush()

        warm_model = None
        if use_warm and ins.parameters.tensors:
            try:
                warm_model = pickle.loads(ins.parameters.tensors[0])
                print(f"    [Cliente {self.client_id}] Modelo global recebido para warm start")
                sys.stdout.flush()
            except Exception as e:
                print(f"    [Cliente {self.client_id}] Falha no warm start: {e}")

        resource_monitor = ResourceMonitor()
        resource_monitor.start()

        t0 = time.time()
        self.model = ModelFactory.train(
            self.model_type, self.X_train, self.y_train,
            self.client_id, server_round, round_epochs, warm_model,
        )
        elapsed = time.time() - t0

        resource_stats = resource_monitor.stop()

        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = compute_all_metrics(self.y_test, y_pred, y_prob)

        model_bytes = serialize_model(self.model)
        model_size_kb = len(model_bytes) / 1024

        print(f"\n  [Cliente {self.client_id}] FIM Round {server_round} | "
              f"Tempo: {elapsed:.1f}s | Modelo: {model_size_kb:.1f} KB")
        print(f"    CPU: {resource_stats['cpu_percent']:.1f}% | "
              f"RAM: {resource_stats['ram_mb']:.1f} MB "
              f"(pico: {resource_stats['ram_peak_mb']:.1f} MB, "
              f"{resource_stats['ram_percent']:.1f}%)")
        print_metrics_table(
            f"[Cliente {self.client_id}] Metricas no teste:", metrics,
        )
        print(f"    TP={metrics['_tp']} FP={metrics['_fp']} "
              f"TN={metrics['_tn']} FN={metrics['_fn']}")
        sys.stdout.flush()

        # Monta dict de metricas publicas para o servidor
        fit_metrics = {
            "client_id": self.client_id,
            "category": self.category,
            "local_epochs": round_epochs,
            "training_time": float(elapsed),
            "model_size_kb": float(model_size_kb),
            "cpu_percent": float(resource_stats["cpu_percent"]),
            "ram_mb": float(resource_stats["ram_mb"]),
            "ram_peak_mb": float(resource_stats["ram_peak_mb"]),
        }
        for k, v in metrics.items():
            if not k.startswith("_"):
                fit_metrics[k] = float(v)

        return FitRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=Parameters(tensors=[model_bytes], tensor_type="pickle"),
            num_examples=len(self.X_train),
            metrics=fit_metrics,
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        if self.model is None:
            return EvaluateRes(
                status=Status(code=Code.OK, message="No model"),
                loss=1.0, num_examples=0, metrics={},
            )

        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = compute_all_metrics(self.y_test, y_pred, y_prob)
        return EvaluateRes(
            status=Status(code=Code.OK, message="OK"),
            loss=float(1 - metrics["accuracy"]),
            num_examples=len(self.X_test),
            metrics={"accuracy": metrics["accuracy"], "auc": metrics["auc"]},
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cliente FL com conexao gRPC explicita")
    parser.add_argument("--client-id", type=int, required=True,
                        help="ID do cliente (0-5)")
    parser.add_argument("--model", type=str, required=True,
                        choices=["xgboost", "lightgbm", "catboost"])
    args = parser.parse_args()

    category = CLIENT_CATEGORIES.get(args.client_id, "cat1")
    local_epochs = LOCAL_EPOCHS_BY_CAT.get(category, LOCAL_EPOCHS)

    print(f"\n{'='*60}")
    print(f"  CLIENTE {args.client_id} - {args.model.upper()}")
    print(f"  Servidor:      {CLIENT_CONNECT_ADDRESS}")
    print(f"  Categoria:     {category}")
    print(f"  Epocas locais: {local_epochs} | Log a cada: {LOG_EVERY}")
    print(f"{'='*60}")

    print(f"[Cliente {args.client_id}] Carregando dataset Higgs...")
    X_train, y_train, X_test, y_test = DatasetRegistry.load(
        "higgs", role="client", client_id=args.client_id,
    )

    client = SimpleClient(
        client_id=args.client_id,
        model_type=args.model,
        local_epochs=local_epochs,
        category=category,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    print(f"\n[Cliente {args.client_id}] Conectando ao servidor: {CLIENT_CONNECT_ADDRESS}")

    fl.client.start_client(
        server_address=CLIENT_CONNECT_ADDRESS,
        client=client,
    )

    print(f"\n[Cliente {args.client_id}] Treinamento federado concluido!")


if __name__ == "__main__":
    main()
