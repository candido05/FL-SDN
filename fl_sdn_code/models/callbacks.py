"""
Callbacks de progresso e captura de historico para treino local dos modelos.

Fornecem dois servicos:
  1. Feedback visual por epoca durante o treinamento
  2. Captura do historico de loss para logging em epocas_locais.csv

O historico e armazenado em callback.loss_history e consumido
por ModelFactory.train() apos o fit().
"""

import sys
from typing import List, Optional, Tuple

import xgboost as xgb

from config import LOG_EVERY


class XGBoostProgressCallback(xgb.callback.TrainingCallback):
    """
    Imprime progresso e captura historico de loss a cada boosting round.

    Apos o fit(), consulte:
        cb.loss_history  ->  List[ (local_epoch, train_loss, val_loss_or_None) ]
    """

    def __init__(self, client_id: int, server_round: int, total_rounds: int):
        self.client_id = client_id
        self.server_round = server_round
        self.total_rounds = total_rounds
        # Historico de loss: lista de (epoch, train_loss, val_loss|None)
        self.loss_history: List[Tuple[int, float, Optional[float]]] = []

    def after_iteration(self, model, epoch, evals_log):
        current = epoch + 1
        train_loss: Optional[float] = None
        val_loss: Optional[float] = None

        if evals_log:
            items = list(evals_log.items())
            # Primeiro dataset = treino/validacao principal
            for ds_idx, (ds_name, metrics_dict) in enumerate(items):
                for metric_name, values in metrics_dict.items():
                    raw = values[-1]
                    val = raw if isinstance(raw, float) else raw[0]
                    if ds_idx == 0:
                        train_loss = val
                    else:
                        val_loss = val

        if train_loss is not None:
            self.loss_history.append((current, train_loss, val_loss))

        if current % LOG_EVERY == 0 or current == self.total_rounds:
            loss_info = ""
            if evals_log:
                parts = []
                for ds_name, metrics_dict in evals_log.items():
                    for metric_name, values in metrics_dict.items():
                        raw = values[-1]
                        v = raw if isinstance(raw, float) else raw[0]
                        parts.append(f"{ds_name}_{metric_name}={v:.4f}")
                loss_info = " | " + " | ".join(parts) if parts else ""
            print(f"    [Cliente {self.client_id}] Round {self.server_round} | "
                  f"Epoca local {current}/{self.total_rounds}{loss_info}")
            sys.stdout.flush()
        return False


def lgb_progress_callback(client_id: int, server_round: int, total_rounds: int):
    """
    Factory de callback para LightGBM.

    Retorna (callback_fn, loss_history_ref) onde loss_history_ref e uma
    lista mutavel preenchida durante o fit():
        [(local_epoch, train_loss, val_loss_or_None), ...]
    """
    start_iter = [None]
    loss_history: List[Tuple[int, float, Optional[float]]] = []

    def callback(env):
        if start_iter[0] is None:
            start_iter[0] = env.iteration

        local_iter = env.iteration - start_iter[0] + 1

        train_loss: Optional[float] = None
        val_loss: Optional[float] = None

        if env.evaluation_result_list:
            for idx, (ds, metric, val, _) in enumerate(env.evaluation_result_list):
                if idx == 0:
                    train_loss = val
                else:
                    val_loss = val

        if train_loss is not None:
            loss_history.append((local_iter, train_loss, val_loss))

        if local_iter % LOG_EVERY == 0 or local_iter == total_rounds:
            loss_info = ""
            if env.evaluation_result_list:
                ds, metric, val, _ = env.evaluation_result_list[0]
                loss_info = f" | {ds}_{metric}={val:.4f}"
            print(f"    [Cliente {client_id}] Round {server_round} | "
                  f"Epoca local {local_iter}/{total_rounds}{loss_info}")
            sys.stdout.flush()

    callback.order = 10
    return callback, loss_history


class CatBoostEpochRecorder:
    """
    Callback minimalista para CatBoost que captura o loss a cada iteracao.

    CatBoost usa callbacks com interface diferente (nao herda de classe base).
    Passado via CatBoostClassifier(callbacks=[recorder]).

    Apos o fit():
        recorder.loss_history -> [(local_epoch, train_loss, val_loss|None), ...]
    """

    def __init__(self, client_id: int, server_round: int, total_rounds: int):
        self.client_id = client_id
        self.server_round = server_round
        self.total_rounds = total_rounds
        self.loss_history: List[Tuple[int, float, Optional[float]]] = []
        self._iteration = 0

    def after_iteration(self, info) -> bool:
        """
        Chamado pelo CatBoost apos cada iteracao.
        info.iteration: iteracao atual (0-indexed)
        info.metrics: dict com metricas de treino e validacao
        """
        self._iteration += 1
        epoch = self._iteration

        train_loss: Optional[float] = None
        val_loss: Optional[float] = None

        try:
            metrics = info.metrics
            if "learn" in metrics:
                learn_m = metrics["learn"]
                if learn_m:
                    first_metric = next(iter(learn_m.values()))
                    if first_metric:
                        train_loss = float(first_metric[-1])
            if "validation" in metrics:
                val_m = metrics["validation"]
                if val_m:
                    first_metric = next(iter(val_m.values()))
                    if first_metric:
                        val_loss = float(first_metric[-1])
        except Exception:
            pass  # Nao quebra o treino se o callback der erro

        if train_loss is not None:
            self.loss_history.append((epoch, train_loss, val_loss))

        if epoch % LOG_EVERY == 0 or epoch == self.total_rounds:
            loss_str = f"{train_loss:.4f}" if train_loss is not None else "?"
            print(f"    [Cliente {self.client_id}] Round {self.server_round} | "
                  f"Epoca local {epoch}/{self.total_rounds} | train_loss={loss_str}")
            sys.stdout.flush()

        return True  # True = continuar treinando
