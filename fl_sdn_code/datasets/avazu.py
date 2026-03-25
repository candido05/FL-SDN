"""
Dataset Avazu — CTR (Click-Through Rate) prediction, classificacao binaria.

Fonte: Kaggle (Avazu Click-Through Rate Prediction)
Pre-processado por prepare_datasets.py → data/avazu/avazu_X.npy

Caracteristicas:
  - ~40M amostras no dataset original
  - Features categoricas convertidas via Feature Hashing (256 dimensoes)
  - Label: click (0 = nao clicou, 1 = clicou)
  - Dataset altamente desbalanceado (~17% positivos)
  - Benchmark classico para CTR e classificacao em larga escala
"""

import sys

import numpy as np
from sklearn.model_selection import train_test_split

from config import TEST_SIZE, RANDOM_SEED, NUM_CLIENTS
from datasets.registry import DatasetRegistry, stratified_partition
from datasets.paths import npy_paths, is_prepared


@DatasetRegistry.register("avazu")
def load(role: str, client_id: int = 0, **kwargs):
    """
    Carrega o dataset Avazu.

    Args:
        role: "server" retorna (X_test, y_test).
              "client" retorna (X_train_partition, y_train_partition, X_test, y_test).
        client_id: ID do cliente para particionar dados.
    """
    if not is_prepared("avazu"):
        print("ERRO: Dataset Avazu nao preparado.")
        print("  Execute: python prepare_datasets.py --dataset avazu")
        sys.exit(1)

    x_path, y_path = npy_paths("avazu")
    X = np.load(x_path)
    y = np.load(y_path).astype(int)

    # Subconjunto opcional para testes rapidos
    max_samples = kwargs.get("max_samples", len(X))
    if max_samples < len(X):
        rng = np.random.RandomState(RANDOM_SEED)
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X = X[idx]
        y = y[idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y,
    )

    if role == "server":
        print(f"[Servidor] Dataset Avazu: {X_test.shape[0]:,} amostras teste, "
              f"{X_test.shape[1]} features")
        print(f"[Servidor] Classe 0: {(y_test == 0).sum():,} | "
              f"Classe 1: {(y_test == 1).sum():,}")
        return X_test, y_test

    # role == "client"
    num_clients = kwargs.get("num_clients", NUM_CLIENTS)
    indices = stratified_partition(y_train, num_clients, RANDOM_SEED)
    my_idx = indices[client_id]
    X_client = X_train[my_idx]
    y_client = y_train[my_idx]

    print(f"[Cliente {client_id}] Dataset Avazu: {len(my_idx):,} treino, "
          f"{len(y_test):,} teste")
    print(f"[Cliente {client_id}] Classe 0: {(y_client == 0).sum():,} "
          f"({(y_client == 0).mean()*100:.1f}%) | "
          f"Classe 1: {(y_client == 1).sum():,} "
          f"({(y_client == 1).mean()*100:.1f}%)")
    return X_client, y_client, X_test, y_test
