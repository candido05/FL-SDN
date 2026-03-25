"""
Grid Search rapido para hiperparametros do treinamento federado.

Testa combinacoes reduzidas de hiperparametros com 2-fold CV
e salva os melhores em JSON. Pode rodar para cada modelo individualmente
ou para todos os modelos de uma vez.

Uso:
    python tools/grid_search.py --dataset higgs --model xgboost
    python tools/grid_search.py --dataset higgs --all-models
    python tools/grid_search.py --dataset higgs --model xgboost --output tuned_xgboost.json
    python tools/grid_search.py --dataset higgs --model xgboost --sample-size 50000

O JSON gerado pode ser carregado no config.py:
    import json
    TUNED_PARAMS = json.load(open("tuned_params.json"))
"""

import argparse
import json
import os
import sys
import time
import warnings

# Adiciona diretorio pai ao path para encontrar config, datasets, etc.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore")

import numpy as np
from sklearn.model_selection import StratifiedKFold

from config import (
    LOCAL_EPOCHS, RANDOM_SEED,
    XGBOOST_PARAMS, LIGHTGBM_PARAMS, CATBOOST_PARAMS,
)
from datasets import DatasetRegistry


# ---------------------------------------------------------------------------
# Grids REDUZIDOS — ~20-24 combinacoes por modelo (rapido)
# ---------------------------------------------------------------------------

XGBOOST_GRID = {
    "learning_rate": [0.03, 0.05, 0.1],
    "max_depth": [4, 6],
    "reg_alpha": [0.0, 0.5],
    "reg_lambda": [1.0, 3.0],
}

LIGHTGBM_GRID = {
    "learning_rate": [0.03, 0.05, 0.1],
    "max_depth": [4, 6],
    "reg_alpha": [0.0, 0.5],
    "reg_lambda": [1.0, 3.0],
}

CATBOOST_GRID = {
    "learning_rate": [0.03, 0.05, 0.1],
    "depth": [4, 6],
    "l2_leaf_reg": [1.0, 5.0],
}

ALL_MODELS = ["xgboost", "lightgbm", "catboost"]


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


def _cv_score(model_type, params, X, y, n_folds=2):
    """Cross-validation stratificada (2-fold para rapidez)."""
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
    """Executa grid search reduzido."""
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
    print(f"  {total} combinacoes | {len(X):,} amostras | 2-fold CV")
    print(f"{'='*60}")

    best_score = -1
    best_params = None

    for i, params in enumerate(combos):
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
        elif (i + 1) % 10 == 0:
            print(f"  [{i+1}/{total}] Progresso... (melhor ate agora: {best_score:.4f})")

        sys.stdout.flush()

    print(f"\n  Melhor: Acc={best_score:.4f}")
    print(f"  Params: {best_params}")
    return best_params, best_score


def run_grid_search_all_models(X, y, output_dir=None):
    """Executa grid search para todos os modelos e retorna dict com resultados."""
    results = {}

    for model_type in ALL_MODELS:
        t0 = time.time()
        best_params, best_score = run_grid_search(model_type, X, y)
        elapsed = time.time() - t0

        results[model_type] = {
            "params": best_params,
            "score": best_score,
            "time": elapsed,
        }

        print(f"  [{model_type}] Concluido em {elapsed:.1f}s — Acc={best_score:.4f}")

        if output_dir:
            out_path = os.path.join(output_dir, f"tuned_{model_type}.json")
            with open(out_path, "w") as f:
                json.dump(best_params, f, indent=2)
            print(f"  [{model_type}] Salvo em: {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Grid search rapido de hiperparametros para FL"
    )
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["higgs", "higgs_full", "epsilon", "avazu"])
    parser.add_argument("--model", type=str, default=None,
                        choices=ALL_MODELS,
                        help="Modelo (omita para --all-models)")
    parser.add_argument("--all-models", action="store_true",
                        help="Roda grid search para todos os modelos")
    parser.add_argument("--sample-size", type=int, default=30000,
                        help="Amostras para grid search (default: 30000)")
    parser.add_argument("--output", type=str, default=None,
                        help="Arquivo JSON para salvar resultados (modo --model)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Diretorio para salvar JSONs (modo --all-models)")
    args = parser.parse_args()

    if not args.model and not args.all_models:
        parser.error("Especifique --model ou --all-models")

    print(f"\nCarregando dataset {args.dataset}...")
    X_train, y_train, X_test, y_test = DatasetRegistry.load(
        args.dataset, role="client", client_id=0, num_clients=1,
    )

    if args.sample_size < len(X_train):
        rng = np.random.RandomState(RANDOM_SEED)
        idx = rng.choice(len(X_train), size=args.sample_size, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]
        print(f"Subamostrado para {args.sample_size:,} amostras")

    print(f"Dataset: {X_train.shape[0]:,} treino, {X_test.shape[0]:,} teste, "
          f"{X_train.shape[1]} features")
    print(f"Classe 0: {(y_train == 0).sum():,} ({(y_train == 0).mean()*100:.1f}%) | "
          f"Classe 1: {(y_train == 1).sum():,} ({(y_train == 1).mean()*100:.1f}%)")

    t_total = time.time()

    if args.all_models:
        results = run_grid_search_all_models(
            X_train, y_train, output_dir=args.output_dir,
        )
        elapsed = time.time() - t_total

        print(f"\n{'='*60}")
        print(f"  RESULTADO FINAL — Todos os modelos | {args.dataset}")
        print(f"  Tempo total: {elapsed:.1f}s")
        for m, r in results.items():
            print(f"    {m}: Acc={r['score']:.4f} ({r['time']:.1f}s)")
            print(f"      Params: {r['params']}")
        print(f"{'='*60}")
    else:
        best_params, best_score = run_grid_search(args.model, X_train, y_train)
        elapsed = time.time() - t_total

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


if __name__ == "__main__":
    main()
