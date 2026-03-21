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
    params = {**XGBOOST_PARAMS, "n_estimators": local_epochs}
    cb = XGBoostProgressCallback(client_id, server_round, local_epochs)
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
    params = {**LIGHTGBM_PARAMS, "n_estimators": local_epochs}
    cb = lgb_progress_callback(client_id, server_round, local_epochs)
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
    params = {**CATBOOST_PARAMS, "iterations": local_epochs, "verbose": 0}
    model = CatBoostClassifier(**params)

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
    return model
