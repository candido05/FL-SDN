"""
Factory Pattern para criacao e treino de modelos.

Encapsula a logica de instanciacao e warm-start de XGBoost, LightGBM e CatBoost,
substituindo o bloco if/elif que existia em client.py.
"""

import sys

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

from config import XGBOOST_PARAMS, LIGHTGBM_PARAMS, CATBOOST_PARAMS
from models.callbacks import XGBoostProgressCallback, lgb_progress_callback


def _count_trees(model) -> int:
    """Conta arvores em um modelo XGBoost existente."""
    try:
        booster = model.get_booster()
        return int(booster.num_boosted_rounds())
    except Exception:
        return 0


class ModelFactory:
    """
    Factory que cria e treina modelos gradient boosting.

    Uso:
        model = ModelFactory.train("xgboost", X, y, client_id=0,
                                   server_round=1, local_epochs=100)
    """

    _BUILDERS = {}

    @classmethod
    def register(cls, name: str):
        """Decorator para registrar um builder de modelo."""
        def decorator(fn):
            cls._BUILDERS[name] = fn
            return fn
        return decorator

    @classmethod
    def train(cls, model_type: str, X_train, y_train, client_id: int,
              server_round: int, local_epochs: int, warm_start_model=None):
        """Cria e treina modelo do tipo especificado."""
        builder = cls._BUILDERS.get(model_type)
        if builder is None:
            raise ValueError(
                f"Modelo desconhecido: {model_type}. "
                f"Disponiveis: {list(cls._BUILDERS.keys())}"
            )
        return builder(X_train, y_train, client_id, server_round,
                        local_epochs, warm_start_model)


@ModelFactory.register("xgboost")
def _train_xgboost(X_train, y_train, client_id, server_round,
                    local_epochs, warm_start_model):
    # Warm start: reduz n_estimators gradualmente para evitar overfitting
    # O modelo ja tem arvores do round anterior — precisa de menos refinamento
    if warm_start_model is not None:
        n_existing = _count_trees(warm_start_model)
        # Diminui 20% por round, minimo 10% das epocas base
        reduction = max(0.1, 1.0 - (server_round - 1) * 0.2)
        n_new = max(int(local_epochs * reduction), 10)
        print(f"    [XGBoost] Warm start: {n_existing} arvores existentes + "
              f"{n_new} novas (reducao {reduction:.0%} do base {local_epochs})")
    else:
        n_new = local_epochs

    params = {**XGBOOST_PARAMS, "n_estimators": n_new}
    cb = XGBoostProgressCallback(client_id, server_round, n_new)
    model = xgb.XGBClassifier(**params, callbacks=[cb])
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
    return model


@ModelFactory.register("lightgbm")
def _train_lightgbm(X_train, y_train, client_id, server_round,
                     local_epochs, warm_start_model):
    if warm_start_model is not None:
        reduction = max(0.1, 1.0 - (server_round - 1) * 0.2)
        n_new = max(int(local_epochs * reduction), 10)
        print(f"    [LightGBM] Warm start: {n_new} novas iteracoes "
              f"(reducao {reduction:.0%} do base {local_epochs})")
    else:
        n_new = local_epochs

    params = {**LIGHTGBM_PARAMS, "n_estimators": n_new}
    cb = lgb_progress_callback(client_id, server_round, n_new)
    model = lgb.LGBMClassifier(**params)

    if warm_start_model is not None:
        model.fit(X_train, y_train, init_model=warm_start_model,
                  eval_set=[(X_train, y_train)], callbacks=[cb])
    else:
        model.fit(X_train, y_train,
                  eval_set=[(X_train, y_train)], callbacks=[cb])
    return model


@ModelFactory.register("catboost")
def _train_catboost(X_train, y_train, client_id, server_round,
                    local_epochs, warm_start_model):
    if warm_start_model is not None:
        reduction = max(0.1, 1.0 - (server_round - 1) * 0.2)
        n_new = max(int(local_epochs * reduction), 10)
        print(f"    [CatBoost] Warm start: {n_new} novas iteracoes "
              f"(reducao {reduction:.0%} do base {local_epochs})")
    else:
        n_new = local_epochs

    params = {**CATBOOST_PARAMS, "iterations": n_new, "verbose": 0}
    model = CatBoostClassifier(**params)

    print(f"    [Cliente {client_id}] Round {server_round} | "
          f"Treinando CatBoost ({n_new} iteracoes)...")
    sys.stdout.flush()

    if warm_start_model is not None:
        model.fit(X_train, y_train, init_model=warm_start_model, verbose=False)
    else:
        model.fit(X_train, y_train, verbose=False)

    print(f"    [Cliente {client_id}] Round {server_round} | "
          f"CatBoost {n_new}/{n_new} iteracoes concluidas")
    sys.stdout.flush()
    return model
