"""
Servidor Flower com conexao gRPC explicita.

Uso:
    python server.py --model xgboost --strategy bagging
    python server.py --model lightgbm --strategy cycling

Logging:
    EXP=com_sdn  python server.py --model xgboost --strategy bagging
    EXP=sem_sdn  python server.py --model xgboost --strategy bagging
    → gera <EXP>_resultados.csv com elapsed_sec x metricas por round
"""

import argparse
import csv
import logging
import os
import pickle
import sys
import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*DEPRECATED.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")
warnings.filterwarnings("ignore", message=".*feature names.*")

import flwr as fl
from flwr.common import Parameters, Scalar, FitRes, EvaluateRes, FitIns, EvaluateIns
from flwr.server.strategy import Strategy
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager

logging.getLogger("flwr").setLevel(logging.ERROR)
logging.getLogger("grpc").setLevel(logging.ERROR)

from config import (
    SERVER_ADDRESS, NUM_CLIENTS, NUM_ROUNDS,
    N_SAMPLES, TEST_SIZE, RANDOM_SEED, LOCAL_EPOCHS,
)


# ---------------------------------------------------------------------------
# Logging por round — gera CSV incremental para os graficos do TCC
# ---------------------------------------------------------------------------
#
# Como funciona:
#   _EXP_NAME  lê a variável de ambiente EXP (ex: "com_sdn", "sem_sdn").
#              Permite rodar os dois experimentos sem mudar o código.
#   _t_start   é setado logo antes de fl.server.start_server() para medir
#              o tempo total a partir do momento em que o treinamento começa.
#   _log_round() é chamado dentro do método evaluate() de cada estratégia,
#              que o Flower invoca ao final de cada round com as métricas
#              já calculadas sobre o conjunto de teste.
#   O CSV é reescrito completo a cada round (não apenas append) para que
#   seja legível mesmo se o experimento travar no meio.

_EXP_NAME = os.environ.get("EXP", "experimento")
_LOG_FILE  = f"{_EXP_NAME}_resultados.csv"
_t_start: float = 0.0
_log_rows: List[Dict] = []


def _log_round(server_round: int, metrics: Dict) -> None:
    """Registra métricas do round atual no CSV e imprime no terminal."""
    elapsed = round(time.time() - _t_start, 2)
    row = {
        "round":       server_round,
        "elapsed_sec": elapsed,
        "accuracy":    round(metrics.get("accuracy",  0.0), 4),
        "f1":          round(metrics.get("f1",        0.0), 4),
        "auc":         round(metrics.get("auc",       0.0), 4),
        "precision":   round(metrics.get("precision", 0.0), 4),
        "recall":      round(metrics.get("recall",    0.0), 4),
    }
    _log_rows.append(row)

    print(f"  [LOG] Round {server_round:2d} | {elapsed:7.1f}s | "
          f"acc={row['accuracy']:.4f} | f1={row['f1']:.4f} | auc={row['auc']:.4f}")
    sys.stdout.flush()

    with open(_LOG_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerows(_log_rows)


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_X_PATH   = os.path.join(_DATA_DIR, "higgs_X.npy")
_Y_PATH   = os.path.join(_DATA_DIR, "higgs_y.npy")


def load_higgs_test_data():
    """Carrega subset do Higgs e retorna apenas o conjunto de teste."""
    print("[Servidor] Carregando dataset Higgs...")
    if os.path.exists(_X_PATH) and os.path.exists(_Y_PATH):
        print(f"[Servidor] Usando cache local: {_DATA_DIR}")
        X = np.load(_X_PATH)
        y = np.load(_Y_PATH).astype(int)
    else:
        higgs = fetch_openml(name="higgs", version=2, as_frame=False, parser="auto")
        X, y  = higgs.data[:N_SAMPLES], higgs.target[:N_SAMPLES].astype(int)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y,
    )
    print(f"[Servidor] Dataset carregado: {X_test.shape[0]} amostras teste, "
          f"{X_test.shape[1]} features")
    print(f"[Servidor] Classe 0: {(y_test == 0).sum()} | Classe 1: {(y_test == 1).sum()}")
    return X_test, y_test


def deserialize_model(raw_bytes: bytes):
    return pickle.loads(raw_bytes)


def print_metrics(prefix: str, y_true, y_pred, y_prob):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob)

    print(f"  {prefix}")
    print(f"    Accuracy  = {acc:.4f}")
    print(f"    Precision = {prec:.4f}")
    print(f"    Recall    = {rec:.4f}")
    print(f"    F1-Score  = {f1:.4f}")
    print(f"    AUC-ROC   = {auc:.4f}")
    sys.stdout.flush()

    return acc, prec, rec, f1, auc


# ---------------------------------------------------------------------------
# Estrategia Bagging
# ---------------------------------------------------------------------------

class SimpleBagging(Strategy):
    """Todos os clientes treinam em paralelo; ensemble por media de probabilidades."""

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
            fit_params  = Parameters(tensors=[model_bytes], tensor_type="pickle")
            config      = {"server_round": server_round, "warm_start": True}
            print(f"  [Servidor] Enviando melhor modelo ({len(model_bytes)/1024:.1f} KB) "
                  f"para {self.num_clients} clientes")
        else:
            fit_params = Parameters(tensors=[], tensor_type="")
            config     = {"server_round": server_round, "warm_start": False}
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
            print(f"    Cliente {cid}: Acc={acc:.4f} F1={f1:.4f} "
                  f"Tempo={t:.1f}s Modelo={sz:.1f}KB")

            if acc > best_acc:
                best_acc = acc
                best_cid = cid

        if best_cid >= 0:
            self.best_model = self.client_models[best_cid]
            print(f"  [Servidor] Melhor modelo: Cliente {best_cid} (Acc={best_acc:.4f})")

        if failures:
            print(f"  [Servidor] AVISO: {len(failures)} falha(s)")

        sys.stdout.flush()
        return None, {"num_models": len(self.client_models)}

    def configure_evaluate(self, server_round, parameters, client_manager):
        return []

    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}

    def evaluate(self, server_round: int, parameters: Parameters):
        if not self.client_models:
            return None

        # Ensemble: média das probabilidades de todos os modelos do round
        preds  = [m.predict_proba(self.X_test)[:, 1] for m in self.client_models.values()]
        y_prob = np.mean(preds, axis=0)
        y_pred = (y_prob >= 0.5).astype(int)

        print(f"\n  [Servidor] METRICAS ENSEMBLE Round {server_round}/{NUM_ROUNDS} "
              f"({len(self.client_models)} modelos agregados):")
        acc, prec, rec, f1, auc = print_metrics(
            "Avaliacao no conjunto de teste:", self.y_test, y_pred, y_prob,
        )

        # ── Logging CSV ──────────────────────────────────────────────────
        # Chamado aqui porque evaluate() é invocado pelo Flower ao final
        # de cada round, após aggregate_fit(), com acesso às métricas finais.
        _log_round(server_round, {
            "accuracy":  acc,
            "precision": prec,
            "recall":    rec,
            "f1":        f1,
            "auc":       auc,
        })
        # ─────────────────────────────────────────────────────────────────

        return float(1 - acc), {
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc,
        }


# ---------------------------------------------------------------------------
# Estrategia Cycling
# ---------------------------------------------------------------------------

class SimpleCycling(Strategy):
    """Um cliente por rodada (round-robin); modelo passa sequencialmente."""

    def __init__(self, num_clients: int, X_test: np.ndarray, y_test: np.ndarray):
        self.num_clients   = num_clients
        self.X_test        = X_test
        self.y_test        = y_test
        self.current_idx   = 0
        self.current_model = None

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
        print(f"# CYCLING - Round {server_round}/{NUM_ROUNDS}")
        print(f"# Cliente selecionado: {self.current_idx}")
        print(f"# Warm start: {has_model}")
        print(f"# Epocas locais: {LOCAL_EPOCHS}")
        print(f"{'#'*60}")
        sys.stdout.flush()

        clients  = client_manager.sample(
            num_clients=self.num_clients,
            min_num_clients=self.num_clients,
        )
        clients  = sorted(clients, key=lambda c: c.cid)
        selected = clients[self.current_idx]

        config = {"server_round": server_round, "cycling_position": self.current_idx}
        if has_model:
            model_bytes = pickle.dumps(self.current_model)
            fit_params  = Parameters(tensors=[model_bytes], tensor_type="pickle")
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

        _, fit_res         = results[0]
        cid                = int(fit_res.metrics.get("client_id", self.current_idx))
        self.current_model = deserialize_model(fit_res.parameters.tensors[0])

        t   = fit_res.metrics.get("training_time", 0)
        acc = fit_res.metrics.get("accuracy",      0)
        f1  = fit_res.metrics.get("f1",            0)
        sz  = fit_res.metrics.get("model_size_kb", 0)

        print(f"\n  [Servidor] Modelo recebido (client_id={cid}, posicao={self.current_idx}):")
        print(f"    Acc={acc:.4f} F1={f1:.4f} Tempo={t:.1f}s Modelo={sz:.1f}KB")
        print(f"  [Servidor] Modelo salvo. Proxima posicao: "
              f"{(self.current_idx + 1) % self.num_clients}")
        sys.stdout.flush()

        prev             = self.current_idx
        self.current_idx = (self.current_idx + 1) % self.num_clients
        return None, {"trained_client": prev}

    def configure_evaluate(self, server_round, parameters, client_manager):
        return []

    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}

    def evaluate(self, server_round: int, parameters: Parameters):
        if self.current_model is None:
            return None

        y_prob = self.current_model.predict_proba(self.X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        trained_client = (self.current_idx - 1) % self.num_clients
        print(f"\n  [Servidor] METRICAS Round {server_round}/{NUM_ROUNDS} "
              f"(modelo do cliente {trained_client}):")
        acc, prec, rec, f1, auc = print_metrics(
            "Avaliacao no conjunto de teste:", self.y_test, y_pred, y_prob,
        )

        # ── Logging CSV ──────────────────────────────────────────────────
        _log_round(server_round, {
            "accuracy":  acc,
            "precision": prec,
            "recall":    rec,
            "f1":        f1,
            "auc":       auc,
        })
        # ─────────────────────────────────────────────────────────────────

        return float(1 - acc), {
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc,
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Servidor FL com conexao gRPC explicita")
    parser.add_argument("--model",    type=str, required=True,
                        choices=["xgboost", "lightgbm", "catboost"])
    parser.add_argument("--strategy", type=str, required=True,
                        choices=["bagging", "cycling"])
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  SERVIDOR FL - {args.model.upper()} + {args.strategy.upper()}")
    print(f"  Experimento: {_EXP_NAME}  →  {_LOG_FILE}")
    print(f"{'='*60}")

    X_test, y_test = load_higgs_test_data()

    strategy = (
        SimpleBagging(NUM_CLIENTS, X_test, y_test)
        if args.strategy == "bagging"
        else SimpleCycling(NUM_CLIENTS, X_test, y_test)
    )

    print(f"\n[Servidor] Configuracao:")
    print(f"    Modelo:      {args.model}")
    print(f"    Estrategia:  {args.strategy}")
    print(f"    Rounds:      {NUM_ROUNDS}")
    print(f"    Clientes:    {NUM_CLIENTS}")
    print(f"    Epocas loc:  {LOCAL_EPOCHS}")
    print(f"    Endereco:    {SERVER_ADDRESS}")
    print(f"    Log CSV:     {_LOG_FILE}")
    print(f"\n[Servidor] Aguardando {NUM_CLIENTS} cliente(s) conectarem...\n")
    sys.stdout.flush()

    # Timer inicia aqui: mede o tempo real de treinamento federado,
    # excluindo o carregamento do dataset (que não varia entre experimentos).
    global _t_start
    _t_start = time.time()

    fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

    total = round(time.time() - _t_start, 1)
    print(f"\n{'='*60}")
    print(f"  SERVIDOR - TREINAMENTO CONCLUIDO")
    print(f"  Tempo total: {total}s  |  Rounds: {NUM_ROUNDS}  |  Log: {_LOG_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()