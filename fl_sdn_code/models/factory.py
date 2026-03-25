"""
Factory Pattern para criacao e treino de modelos.

Encapsula a logica de instanciacao e warm-start de XGBoost, LightGBM e CatBoost,
substituindo o bloco if/elif que existia em client.py.

Mitigacoes de overfitting implementadas:
  - Tree cap: limita total de arvores acumuladas via warm start (MAX_TOTAL_TREES)
  - Early stopping: para treinamento se validacao nao melhora (EARLY_STOPPING_ROUNDS)
  - Validation split: reserva parcela do treino para monitorar generalizacao
  - Extra params: permite injetar hiperparametros tunados (via grid_search.py)
"""

import sys

import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

from config import (
    XGBOOST_PARAMS, LIGHTGBM_PARAMS, CATBOOST_PARAMS,
    RANDOM_SEED, MAX_TOTAL_TREES, EARLY_STOPPING_ROUNDS, VALIDATION_SPLIT,
    WARM_START_TREE_DECAY, WARM_START_TREE_MIN_RATIO,
    WARM_START_LR_DECAY, WARM_START_LR_MIN_RATIO,
)
from models.callbacks import XGBoostProgressCallback, lgb_progress_callback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_trees(model) -> int:
    """Conta arvores em um modelo XGBoost existente."""
    try:
        booster = model.get_booster()
        return int(booster.num_boosted_rounds())
    except Exception:
        return 0


def _count_trees_lgb(model) -> int:
    """Conta arvores em um modelo LightGBM existente."""
    try:
        return model.booster_.num_trees()
    except Exception:
        try:
            return model.n_estimators_
        except Exception:
            return 0


def _count_trees_catboost(model) -> int:
    """Conta arvores em um modelo CatBoost existente."""
    try:
        return model.tree_count_
    except Exception:
        return 0


def _apply_tree_cap(n_new, n_existing, label=""):
    """Aplica o limite de arvores totais, retornando n_new ajustado."""
    total = n_existing + n_new
    if total > MAX_TOTAL_TREES:
        capped = max(MAX_TOTAL_TREES - n_existing, 0)
        if capped == 0:
            print(f"    [{label}] Tree cap atingido: {n_existing} arvores existentes "
                  f">= {MAX_TOTAL_TREES}. Retornando modelo existente.")
        else:
            print(f"    [{label}] Tree cap: {n_new} → {capped} novas arvores "
                  f"(total {n_existing}+{capped}={n_existing+capped}, "
                  f"limite={MAX_TOTAL_TREES})")
        return capped
    return n_new


def _compute_warm_start_schedule(
    base_lr: float,
    local_epochs: int,
    server_round: int,
    extra_params: dict = None,
) -> tuple:
    """
    Calcula o schedule de decaimento exponencial para warm start.

    Tanto n_estimators quanto learning_rate decaem suavemente conforme
    os rounds avancam — o modelo ja acumulou base e precisa de ajustes
    cada vez mais refinados (menor LR, menos arvores novas).

    Formulacao:
        tree_ratio = max(WARM_START_TREE_MIN_RATIO,
                         WARM_START_TREE_DECAY ^ (server_round - 1))
        lr         = max(base_lr * WARM_START_LR_MIN_RATIO,
                         base_lr * WARM_START_LR_DECAY ^ (server_round - 1))

    Args:
        base_lr:      Learning rate base do modelo (de XGBOOST_PARAMS etc.).
        local_epochs: Epocas locais configuradas (LOCAL_EPOCHS).
        server_round: Round atual do servidor (comeca em 1).
        extra_params: Hiperparametros tunados (grid search); se contem
                      'learning_rate', usa esse valor como base.

    Returns:
        (n_new, lr_adjusted)
        - n_new:       Numero de novas arvores a adicionar neste round.
        - lr_adjusted: Learning rate ajustado para este round.
    """
    # Se o grid search tuneu o LR, usa o tunado como base para o decay
    effective_base_lr = base_lr
    if extra_params and "learning_rate" in extra_params:
        effective_base_lr = float(extra_params["learning_rate"])

    # Decaimento exponencial do n_estimators (piso = WARM_START_TREE_MIN_RATIO)
    tree_ratio = max(
        WARM_START_TREE_MIN_RATIO,
        WARM_START_TREE_DECAY ** (server_round - 1),
    )
    n_new = max(int(local_epochs * tree_ratio), 10)

    # Decaimento exponencial do learning_rate (piso = MIN_RATIO * base)
    lr_adjusted = max(
        effective_base_lr * WARM_START_LR_MIN_RATIO,
        effective_base_lr * WARM_START_LR_DECAY ** (server_round - 1),
    )
    lr_adjusted = round(lr_adjusted, 6)

    return n_new, lr_adjusted


def _make_val_split(X_train, y_train):
    """
    Separa uma parcela do treino para validacao (early stopping).
    Retorna (X_fit, X_val, y_fit, y_val).
    Se o dataset for muito pequeno (<50 amostras), nao faz split.
    """
    if len(X_train) < 50 or VALIDATION_SPLIT <= 0:
        return X_train, None, y_train, None

    return train_test_split(
        X_train, y_train,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_SEED,
        stratify=y_train,
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

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
              server_round: int, local_epochs: int, warm_start_model=None,
              extra_params=None):
        """Cria e treina modelo do tipo especificado."""
        builder = cls._BUILDERS.get(model_type)
        if builder is None:
            raise ValueError(
                f"Modelo desconhecido: {model_type}. "
                f"Disponiveis: {list(cls._BUILDERS.keys())}"
            )
        return builder(X_train, y_train, client_id, server_round,
                        local_epochs, warm_start_model, extra_params)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

@ModelFactory.register("xgboost")
def _train_xgboost(X_train, y_train, client_id, server_round,
                    local_epochs, warm_start_model, extra_params):
    # Warm start: decaimento exponencial de n_estimators e learning_rate
    base_lr = XGBOOST_PARAMS["learning_rate"]
    if warm_start_model is not None:
        n_existing = _count_trees(warm_start_model)
        n_new, lr_adjusted = _compute_warm_start_schedule(
            base_lr, local_epochs, server_round, extra_params,
        )
        # Tree cap
        n_new = _apply_tree_cap(n_new, n_existing, "XGBoost")
        if n_new == 0:
            return warm_start_model
        print(f"    [XGBoost] Warm start: {n_existing} arvores existentes + "
              f"{n_new} novas | lr={lr_adjusted:.5f} "
              f"(round {server_round}, base={local_epochs})")
    else:
        n_new = local_epochs
        lr_adjusted = base_lr
        if extra_params and "learning_rate" in extra_params:
            lr_adjusted = float(extra_params["learning_rate"])

    params = {**XGBOOST_PARAMS, "n_estimators": n_new, "learning_rate": lr_adjusted}
    if extra_params:
        # extra_params pode sobrescrever tudo exceto learning_rate no warm start
        # (ja calculamos lr_adjusted usando extra_params como base)
        ep = {k: v for k, v in extra_params.items() if k != "learning_rate"}
        params.update(ep)

    # Validation split para early stopping
    X_fit, X_val, y_fit, y_val = _make_val_split(X_train, y_train)

    # early_stopping_rounds vai no construtor (XGBoost >= 2.0)
    if X_val is not None:
        params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS

    cb = XGBoostProgressCallback(client_id, server_round, n_new)
    model = xgb.XGBClassifier(**params, callbacks=[cb])

    fit_kwargs = {"verbose": False}
    if X_val is not None:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
    else:
        fit_kwargs["eval_set"] = [(X_fit, y_fit)]

    if warm_start_model is not None:
        model.fit(X_fit, y_fit, xgb_model=warm_start_model.get_booster(),
                  **fit_kwargs)
    else:
        model.fit(X_fit, y_fit, **fit_kwargs)

    # Log early stopping se ocorreu
    if X_val is not None and hasattr(model, 'best_iteration'):
        actual = model.best_iteration + 1
        if actual < n_new:
            print(f"    [XGBoost] Early stopping na iteracao {actual}/{n_new}")

    # Limpar callbacks para que pickle nao tente serializa-los
    model.set_params(callbacks=None)
    if hasattr(model, '_callbacks'):
        model._callbacks = None
    return model


@ModelFactory.register("lightgbm")
def _train_lightgbm(X_train, y_train, client_id, server_round,
                     local_epochs, warm_start_model, extra_params):
    base_lr = LIGHTGBM_PARAMS["learning_rate"]
    if warm_start_model is not None:
        n_existing = _count_trees_lgb(warm_start_model)
        n_new, lr_adjusted = _compute_warm_start_schedule(
            base_lr, local_epochs, server_round, extra_params,
        )
        # Tree cap
        n_new = _apply_tree_cap(n_new, n_existing, "LightGBM")
        if n_new == 0:
            return warm_start_model
        print(f"    [LightGBM] Warm start: {n_new} novas iteracoes | "
              f"lr={lr_adjusted:.5f} (round {server_round}, base={local_epochs})")
    else:
        n_new = local_epochs
        lr_adjusted = base_lr
        if extra_params and "learning_rate" in extra_params:
            lr_adjusted = float(extra_params["learning_rate"])

    params = {**LIGHTGBM_PARAMS, "n_estimators": n_new, "learning_rate": lr_adjusted}
    if extra_params:
        ep = {k: v for k, v in extra_params.items() if k != "learning_rate"}
        params.update(ep)

    # Validation split para early stopping
    X_fit, X_val, y_fit, y_val = _make_val_split(X_train, y_train)

    progress_cb = lgb_progress_callback(client_id, server_round, n_new)
    callbacks = [progress_cb]
    if X_val is not None:
        callbacks.append(lgb.early_stopping(EARLY_STOPPING_ROUNDS))

    model = lgb.LGBMClassifier(**params)

    if X_val is not None:
        eval_set = [(X_val, y_val)]
    else:
        eval_set = [(X_fit, y_fit)]

    if warm_start_model is not None:
        model.fit(X_fit, y_fit, init_model=warm_start_model,
                  eval_set=eval_set, callbacks=callbacks)
    else:
        model.fit(X_fit, y_fit,
                  eval_set=eval_set, callbacks=callbacks)

    # Log early stopping se ocorreu
    if X_val is not None and hasattr(model, 'best_iteration_'):
        actual = model.best_iteration_
        if actual > 0 and actual < n_new:
            print(f"    [LightGBM] Early stopping na iteracao {actual}/{n_new}")

    return model


@ModelFactory.register("catboost")
def _train_catboost(X_train, y_train, client_id, server_round,
                    local_epochs, warm_start_model, extra_params):
    base_lr = CATBOOST_PARAMS["learning_rate"]
    if warm_start_model is not None:
        n_existing = _count_trees_catboost(warm_start_model)
        n_new, lr_adjusted = _compute_warm_start_schedule(
            base_lr, local_epochs, server_round, extra_params,
        )
        # Tree cap
        n_new = _apply_tree_cap(n_new, n_existing, "CatBoost")
        if n_new == 0:
            return warm_start_model
        print(f"    [CatBoost] Warm start: {n_new} novas iteracoes | "
              f"lr={lr_adjusted:.5f} (round {server_round}, base={local_epochs})")
    else:
        n_new = local_epochs
        lr_adjusted = base_lr
        if extra_params and "learning_rate" in extra_params:
            lr_adjusted = float(extra_params["learning_rate"])

    params = {**CATBOOST_PARAMS, "iterations": n_new, "learning_rate": lr_adjusted, "verbose": 0}
    if extra_params:
        ep = {k: v for k, v in extra_params.items() if k != "learning_rate"}
        params.update(ep)

    # Validation split para early stopping
    X_fit, X_val, y_fit, y_val = _make_val_split(X_train, y_train)

    if X_val is not None:
        params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS

    model = CatBoostClassifier(**params)

    print(f"    [Cliente {client_id}] Round {server_round} | "
          f"Treinando CatBoost ({n_new} iteracoes)...")
    sys.stdout.flush()

    fit_kwargs = {"verbose": False}
    if X_val is not None:
        fit_kwargs["eval_set"] = (X_val, y_val)

    if warm_start_model is not None:
        model.fit(X_fit, y_fit, init_model=warm_start_model, **fit_kwargs)
    else:
        model.fit(X_fit, y_fit, **fit_kwargs)

    # Log early stopping se ocorreu
    if X_val is not None:
        actual = model.tree_count_
        if actual < n_new:
            print(f"    [CatBoost] Early stopping na iteracao {actual}/{n_new}")

    print(f"    [Cliente {client_id}] Round {server_round} | "
          f"CatBoost {model.tree_count_}/{n_new} iteracoes concluidas")
    sys.stdout.flush()
    return model
