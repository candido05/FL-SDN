"""
Grid Search para hiperparametros do treinamento federado.

Script pre-experimento que testa combinacoes de hiperparametros
usando cross-validation e salva os melhores em JSON.

Uso:
    python grid_search.py --dataset higgs --model xgboost
    python grid_search.py --dataset epsilon --model lightgbm --sample-size 30000
    python grid_search.py --dataset avazu --model catboost --output tuned_params.json

O JSON gerado pode ser carregado no config.py:
    import json
    TUNED_PARAMS = json.load(open("tuned_params.json"))
"""

import argparse
import json
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from sklearn.model_selection import StratifiedKFold

from config import (
    LOCAL_EPOCHS, RANDOM_SEED,
    XGBOOST_PARAMS, LIGHTGBM_PARAMS, CATBOOST_PARAMS,
)
from datasets import DatasetRegistry


# ---------------------------------------------------------------------------
# Grids de hiperparametros por modelo
# ---------------------------------------------------------------------------

XGBOOST_GRID = {
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "max_depth": [4, 6, 8],
    "reg_alpha": [0.0, 0.1, 0.5, 1.0],
    "reg_lambda": [1.0, 2.0, 5.0],
    "min_child_weight": [1, 3, 5, 10],
}

LIGHTGBM_GRID = {
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "max_depth": [4, 6, 8],
    "reg_alpha": [0.0, 0.1, 0.5, 1.0],
    "reg_lambda": [1.0, 2.0, 5.0],
    "min_child_weight": [1, 3, 5, 10],
}

CATBOOST_GRID = {
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "depth": [4, 6, 8],
    "l2_leaf_reg": [1.0, 3.0, 5.0, 10.0],
}


def _generate_combinations(grid):
    """Gera todas as combinacoes de um grid de hiperparametros."""
    keys = list(grid.keys())
    values = list(grid.values())
    combos = [{}]
    for key, vals in zip(keys, values):
        new_combos = []
        for combo in combos:
            for v in vals:
                new_combos.append({**combo, key: v})
        combos = new_combos
    return combos


def _train_and_score(model_type, params, X_train, y_train, X_val, y_val):
    """Treina modelo com params e retorna accuracy no validation."""
    if model_type == "xgboost":
        import xgboost as xgb
        base = {**XGBOOST_PARAMS}
        base.update(params)
        base["early_stopping_rounds"] = 10
        model = xgb.XGBClassifier(**base)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  verbose=False)
    elif model_type == "lightgbm":
        import lightgbm as lgb
        base = {**LIGHTGBM_PARAMS}
        base.update(params)
        model = lgb.LGBMClassifier(**base)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)])
    elif model_type == "catboost":
        from catboost import CatBoostClassifier
        base = {**CATBOOST_PARAMS}
        base.update(params)
        base["early_stopping_rounds"] = 10
        model = CatBoostClassifier(**base)
        model.fit(X_train, y_train,
                  eval_set=(X_val, y_val),
                  verbose=False)
    else:
        raise ValueError(f"Modelo desconhecido: {model_type}")

    y_pred = model.predict(X_val)
    return float(np.mean(y_pred == y_val))


def _cv_score(model_type, params, X, y, n_folds=3):
    """Cross-validation stratificada, retorna media da accuracy."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        acc = _train_and_score(
            model_type, params,
            X[train_idx], y[train_idx],
            X[val_idx], y[val_idx],
        )
        scores.append(acc)
    return np.mean(scores)


def run_grid_search(model_type, X, y):
    """Executa grid search."""
    if model_type == "xgboost":
        grid = XGBOOST_GRID
    elif model_type == "lightgbm":
        grid = LIGHTGBM_GRID
    elif model_type == "catboost":
        grid = CATBOOST_GRID
    else:
        raise ValueError(f"Modelo desconhecido: {model_type}")

    combos = _generate_combinations(grid)
    total = len(combos)
    print(f"\n{'='*60}")
    print(f"  Grid Search: {model_type} | {LOCAL_EPOCHS} estimators")
    print(f"  {total} combinacoes | {len(X)} amostras | 3-fold CV")
    print(f"{'='*60}")

    best_score = -1
    best_params = None

    for i, params in enumerate(combos):
        # Injeta n_estimators
        if model_type in ("xgboost", "lightgbm"):
            params["n_estimators"] = LOCAL_EPOCHS
        elif model_type == "catboost":
            params["iterations"] = LOCAL_EPOCHS

        try:
            score = _cv_score(model_type, params, X, y)
        except Exception as e:
            print(f"  [{i+1}/{total}] ERRO: {e}")
            continue

        if score > best_score:
            best_score = score
            best_params = {k: v for k, v in params.items()
                          if k not in ("n_estimators", "iterations")}
            print(f"  [{i+1}/{total}] Acc={score:.4f} *** NOVO MELHOR *** {best_params}")
        elif (i + 1) % 50 == 0:
            print(f"  [{i+1}/{total}] Progresso... (melhor ate agora: {best_score:.4f})")

        sys.stdout.flush()

    print(f"\n  Melhor: Acc={best_score:.4f}")
    print(f"  Params: {best_params}")
    return best_params, best_score


def main():
    parser = argparse.ArgumentParser(
        description="Grid search de hiperparametros para FL"
    )
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["higgs", "higgs_full", "epsilon", "avazu"])
    parser.add_argument("--model", type=str, required=True,
                        choices=["xgboost", "lightgbm", "catboost"])
    parser.add_argument("--sample-size", type=int, default=50000,
                        help="Amostras para grid search (default: 50000)")
    parser.add_argument("--output", type=str, default=None,
                        help="Arquivo JSON para salvar resultados")
    args = parser.parse_args()

    print(f"\nCarregando dataset {args.dataset}...")
    X_train, y_train, X_test, y_test = DatasetRegistry.load(
        args.dataset, role="client", client_id=0, num_clients=1,
    )

    # Subsample se necessario
    if args.sample_size < len(X_train):
        rng = np.random.RandomState(RANDOM_SEED)
        idx = rng.choice(len(X_train), size=args.sample_size, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]
        print(f"Subamostrado para {args.sample_size} amostras")

    print(f"Dataset: {X_train.shape[0]} treino, {X_test.shape[0]} teste, "
          f"{X_train.shape[1]} features")
    print(f"Classe 0: {(y_train == 0).sum()} ({(y_train == 0).mean()*100:.1f}%) | "
          f"Classe 1: {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.1f}%)")

    t0 = time.time()
    best_params, best_score = run_grid_search(args.model, X_train, y_train)
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"  RESULTADO FINAL — {args.model} | {args.dataset}")
    print(f"  Tempo total: {elapsed:.1f}s")
    print(f"  Acc: {best_score:.4f}")
    print(f"  Params: {best_params}")
    print(f"{'='*60}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(best_params, f, indent=2)
        print(f"\n  Salvo em: {args.output}")
        print(f"  Para usar no config.py:")
        print(f'    import json')
        print(f'    TUNED_PARAMS = json.load(open("{args.output}"))')
    else:
        print(f"\n  Para salvar em JSON, use: --output tuned_params.json")
        print(f"\n  Ou copie para config.py:")
        print(f"    TUNED_PARAMS = {json.dumps(best_params, indent=2)}")


if __name__ == "__main__":
    main()
