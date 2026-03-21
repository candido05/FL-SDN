"""
Callbacks de progresso para treino local dos modelos.

Fornecem feedback visual por epoca durante o treinamento,
util para monitorar convergencia em rounds longos.
"""

import sys

import xgboost as xgb

from config import LOG_EVERY


class XGBoostProgressCallback(xgb.callback.TrainingCallback):
    """Imprime progresso a cada LOG_EVERY boosting rounds."""

    def __init__(self, client_id: int, server_round: int, total_rounds: int):
        self.client_id = client_id
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
