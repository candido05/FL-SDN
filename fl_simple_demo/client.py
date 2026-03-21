"""
Cliente Flower com conexao gRPC explicita.

Uso:
    python client.py --client-id 0 --model xgboost
    python client.py --client-id 1 --model lightgbm

O numero de epocas locais e determinado pela categoria do cliente
(CLIENT_CATEGORIES no config.py), refletindo a heterogeneidade de
hardware descrita no artigo (cat1=low, cat2=medium, cat3=high).
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
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    log_loss, matthews_corrcoef, balanced_accuracy_score, cohen_kappa_score,
    brier_score_loss, average_precision_score, confusion_matrix,
)

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

import flwr as fl
from flwr.common import (
    Code, Status, FitIns, FitRes, EvaluateIns, EvaluateRes,
    Parameters,
)

logging.getLogger("flwr").setLevel(logging.ERROR)
logging.getLogger("grpc").setLevel(logging.ERROR)

from config import (
    CLIENT_CONNECT_ADDRESS, NUM_CLIENTS, LOCAL_EPOCHS, LOG_EVERY,
    N_SAMPLES, TEST_SIZE, RANDOM_SEED,
    XGBOOST_PARAMS, LIGHTGBM_PARAMS, CATBOOST_PARAMS,
    LOCAL_EPOCHS_BY_CAT, CLIENT_CATEGORIES,   # ← novo: épocas por categoria
)


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_X_PATH   = os.path.join(_DATA_DIR, "higgs_X.npy")
_Y_PATH   = os.path.join(_DATA_DIR, "higgs_y.npy")


def load_higgs_client_data(client_id: int):
    """Carrega o Higgs e retorna a particao IID deste cliente."""
    print(f"[Cliente {client_id}] Carregando dataset Higgs ({N_SAMPLES} amostras)...")
    if os.path.exists(_X_PATH) and os.path.exists(_Y_PATH):
        print(f"[Cliente {client_id}] Usando cache local: {_DATA_DIR}")
        X = np.load(_X_PATH)
        y = np.load(_Y_PATH).astype(int)
    else:
        higgs = fetch_openml(name="higgs", version=2, as_frame=False, parser="auto")
        X, y  = higgs.data[:N_SAMPLES], higgs.target[:N_SAMPLES].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y,
    )

    indices  = np.array_split(np.arange(len(y_train)), NUM_CLIENTS)
    my_idx   = indices[client_id]
    X_client = X_train[my_idx]
    y_client = y_train[my_idx]

    print(f"[Cliente {client_id}] Particao carregada:")
    print(f"    Treino: {len(my_idx)} amostras")
    print(f"    Teste:  {len(y_test)} amostras")
    print(f"    Classe 0: {(y_client == 0).sum()} ({(y_client == 0).mean()*100:.1f}%)")
    print(f"    Classe 1: {(y_client == 1).sum()} ({(y_client == 1).mean()*100:.1f}%)")

    return X_client, y_client, X_test, y_test


# ---------------------------------------------------------------------------
# Callbacks de progresso por epoca local
# ---------------------------------------------------------------------------

class XGBoostProgressCallback(xgb.callback.TrainingCallback):
    """Imprime progresso a cada LOG_EVERY boosting rounds."""

    def __init__(self, client_id: int, server_round: int, total_rounds: int):
        self.client_id    = client_id
        self.server_round = server_round
        self.total_rounds = total_rounds

    def after_iteration(self, model, epoch, evals_log):
        current = epoch + 1
        if current % LOG_EVERY == 0 or current == self.total_rounds:
            loss_info = ""
            if evals_log:
                for ds_name, metrics_dict in evals_log.items():
                    for metric_name, values in metrics_dict.items():
                        val = values[-1] if isinstance(values[-1], float) else values[-1][0]
                        loss_info = f" | {ds_name}_{metric_name}={val:.4f}"
            print(f"    [Cliente {self.client_id}] Round {self.server_round} | "
                  f"Epoca local {current}/{self.total_rounds}{loss_info}")
            sys.stdout.flush()
        return False


def lgb_progress_callback(client_id: int, server_round: int, total_rounds: int):
    """Factory de callback para LightGBM."""
    start_iter = [None]
    def callback(env):
        if start_iter[0] is None:
            start_iter[0] = env.iteration
        local_iter = env.iteration - start_iter[0] + 1
        if local_iter % LOG_EVERY == 0 or local_iter == total_rounds:
            loss_info = ""
            if env.evaluation_result_list:
                ds, metric, val, _ = env.evaluation_result_list[0]
                loss_info = f" | {ds}_{metric}={val:.4f}"
            print(f"    [Cliente {client_id}] Round {server_round} | "
                  f"Epoca local {local_iter}/{total_rounds}{loss_info}")
            sys.stdout.flush()
    callback.order = 10
    return callback


# ---------------------------------------------------------------------------
# Treinamento de modelos
# ---------------------------------------------------------------------------

def train_model(model_type: str, X_train, y_train, client_id: int,
                server_round: int, local_epochs: int, warm_start_model=None):
    """
    Treina modelo com progresso por epoca.

    O parametro `local_epochs` substitui o global LOCAL_EPOCHS para
    permitir que cada cliente use um numero diferente de epocas
    conforme sua categoria (cat1/cat2/cat3).
    """
    if model_type == "xgboost":
        params = {**XGBOOST_PARAMS, "n_estimators": local_epochs}
        cb     = XGBoostProgressCallback(client_id, server_round, local_epochs)
        model  = xgb.XGBClassifier(**params, callbacks=[cb])
        eval_set = [(X_train, y_train)]
        if warm_start_model is not None:
            model.fit(X_train, y_train, xgb_model=warm_start_model.get_booster(),
                      eval_set=eval_set, verbose=False)
        else:
            model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        # Limpar callbacks para que pickle nao tente serializa-los
        model.set_params(callbacks=None)
        if hasattr(model, '_callbacks'):
            model._callbacks = None

    elif model_type == "lightgbm":
        params = {**LIGHTGBM_PARAMS, "n_estimators": local_epochs}
        cb     = lgb_progress_callback(client_id, server_round, local_epochs)
        model  = lgb.LGBMClassifier(**params)
        if warm_start_model is not None:
            model.fit(X_train, y_train, init_model=warm_start_model,
                      eval_set=[(X_train, y_train)], callbacks=[cb])
        else:
            model.fit(X_train, y_train,
                      eval_set=[(X_train, y_train)], callbacks=[cb])

    elif model_type == "catboost":
        params = {**CATBOOST_PARAMS, "iterations": local_epochs, "verbose": 0}
        model  = CatBoostClassifier(**params)
        print(f"    [Cliente {client_id}] Round {server_round} | "
              f"Treinando CatBoost ({local_epochs} iteracoes)...")
        sys.stdout.flush()
        if warm_start_model is not None:
            model.fit(X_train, y_train, init_model=warm_start_model, verbose=False)
        else:
            model.fit(X_train, y_train, verbose=False)
        print(f"    [Cliente {client_id}] Round {server_round} | "
              f"CatBoost {local_epochs}/{local_epochs} iteracoes concluidas")
        sys.stdout.flush()
    else:
        raise ValueError(f"Modelo desconhecido: {model_type}")

    return model


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
        self.client_id    = client_id
        self.model_type   = model_type
        self.local_epochs = local_epochs   # épocas resolvidas pela categoria
        self.category     = category
        self.X_train      = X_train
        self.y_train      = y_train
        self.X_test       = X_test
        self.y_test       = y_test
        self.model        = None

    def fit(self, ins: FitIns) -> FitRes:
        config       = ins.config
        server_round = int(config.get("server_round",  0))
        use_warm     = bool(config.get("warm_start",   False))

        # Epocas adaptativas: o servidor SDN pode enviar um valor ajustado
        # Se nao receber, usa o valor da categoria (comportamento original)
        adapted_epochs = int(config.get("adapted_epochs", 0))
        if adapted_epochs > 0:
            round_epochs = adapted_epochs
        else:
            round_epochs = self.local_epochs

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

        t0         = time.time()
        self.model = train_model(
            self.model_type, self.X_train, self.y_train,
            self.client_id, server_round,
            round_epochs,   # ← usa epocas adaptativas se disponivel
            warm_model,
        )
        elapsed = time.time() - t0

        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        acc    = accuracy_score(self.y_test, y_pred)
        prec   = precision_score(self.y_test, y_pred, zero_division=0)
        rec    = recall_score(self.y_test, y_pred, zero_division=0)
        f1     = f1_score(self.y_test, y_pred, zero_division=0)
        auc    = roc_auc_score(self.y_test, y_prob)
        loglss = log_loss(self.y_test, y_prob)
        mcc    = matthews_corrcoef(self.y_test, y_pred)
        bal_ac = balanced_accuracy_score(self.y_test, y_pred)
        kappa  = cohen_kappa_score(self.y_test, y_pred)
        brier  = brier_score_loss(self.y_test, y_prob)
        pr_auc = average_precision_score(self.y_test, y_prob)

        # Specificity (TNR) via confusion matrix
        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        model_bytes    = pickle.dumps(self.model)
        model_size_kb  = len(model_bytes) / 1024

        print(f"\n  [Cliente {self.client_id}] FIM Round {server_round} | Tempo: {elapsed:.1f}s")
        print(f"  [Cliente {self.client_id}] Tamanho modelo: {model_size_kb:.1f} KB")
        print(f"  [Cliente {self.client_id}] Metricas no teste:")
        print(f"    Accuracy      = {acc:.4f}")
        print(f"    Bal. Accuracy = {bal_ac:.4f}")
        print(f"    Precision     = {prec:.4f}")
        print(f"    Recall (TPR)  = {rec:.4f}")
        print(f"    Specificity   = {spec:.4f}")
        print(f"    F1-Score      = {f1:.4f}")
        print(f"    AUC-ROC       = {auc:.4f}")
        print(f"    PR-AUC (AP)   = {pr_auc:.4f}")
        print(f"    Log Loss      = {loglss:.4f}")
        print(f"    Brier Score   = {brier:.4f}")
        print(f"    MCC           = {mcc:.4f}")
        print(f"    Cohen Kappa   = {kappa:.4f}")
        print(f"    TP={tp} FP={fp} TN={tn} FN={fn}")
        sys.stdout.flush()

        return FitRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=Parameters(tensors=[model_bytes], tensor_type="pickle"),
            num_examples=len(self.X_train),
            metrics={
                "client_id":          self.client_id,
                "category":           self.category,
                "local_epochs":       round_epochs,
                "accuracy":           float(acc),
                "balanced_accuracy":  float(bal_ac),
                "precision":          float(prec),
                "recall":             float(rec),
                "specificity":        float(spec),
                "f1":                 float(f1),
                "auc":                float(auc),
                "pr_auc":             float(pr_auc),
                "log_loss":           float(loglss),
                "brier_score":        float(brier),
                "mcc":                float(mcc),
                "cohen_kappa":        float(kappa),
                "training_time":      float(elapsed),
                "model_size_kb":      float(model_size_kb),
            },
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        if self.model is None:
            return EvaluateRes(
                status=Status(code=Code.OK, message="No model"),
                loss=1.0, num_examples=0, metrics={},
            )

        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        acc    = accuracy_score(self.y_test, y_pred)
        auc    = roc_auc_score(self.y_test, y_prob)

        return EvaluateRes(
            status=Status(code=Code.OK, message="OK"),
            loss=float(1 - acc),
            num_examples=len(self.X_test),
            metrics={"accuracy": acc, "auc": auc},
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cliente FL com conexao gRPC explicita")
    parser.add_argument("--client-id", type=int, required=True,
                        help="ID do cliente (0–5)")
    parser.add_argument("--model", type=str, required=True,
                        choices=["xgboost", "lightgbm", "catboost"])
    args = parser.parse_args()

    # Resolver categoria e épocas a partir do client_id
    # CLIENT_CATEGORIES e LOCAL_EPOCHS_BY_CAT estão no config.py
    # Se o client_id não estiver mapeado, usa cat1 como fallback seguro
    category     = CLIENT_CATEGORIES.get(args.client_id, "cat1")
    local_epochs = LOCAL_EPOCHS_BY_CAT.get(category, LOCAL_EPOCHS)

    print(f"\n{'='*60}")
    print(f"  CLIENTE {args.client_id} - {args.model.upper()}")
    print(f"  Servidor:      {CLIENT_CONNECT_ADDRESS}")
    print(f"  Categoria:     {category}")
    print(f"  Epocas locais: {local_epochs} | Log a cada: {LOG_EVERY}")
    print(f"{'='*60}")

    X_train, y_train, X_test, y_test = load_higgs_client_data(args.client_id)

    client = SimpleClient(
        client_id    = args.client_id,
        model_type   = args.model,
        local_epochs = local_epochs,
        category     = category,
        X_train      = X_train,
        y_train      = y_train,
        X_test       = X_test,
        y_test       = y_test,
    )

    print(f"\n[Cliente {args.client_id}] Conectando ao servidor: {CLIENT_CONNECT_ADDRESS}")

    fl.client.start_client(
        server_address=CLIENT_CONNECT_ADDRESS,
        client=client,
    )

    print(f"\n[Cliente {args.client_id}] Treinamento federado concluido!")


if __name__ == "__main__":
    main()