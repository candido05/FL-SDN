"""
Dataset Higgs Full — classificacao binaria (11M amostras, UCI).

Fonte: UCI ML Repository
  URL: https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz

Pre-processado por tools/prepare_datasets.py → data/higgs_full/higgs_full_X.npy

Caracteristicas apos preprocessing:
  - 11M amostras (dataset completo)
  - 28 features originais + interacoes de alto nivel = ~45 features
  - Labels: 0 (fundo) ou 1 (sinal Higgs)
  - Benchmark padrao para classificacao em larga escala
"""

import sys

import numpy as np
from sklearn.model_selection import train_test_split

from config import TEST_SIZE, RANDOM_SEED, NUM_CLIENTS
from datasets.registry import DatasetRegistry, stratified_partition
from datasets.paths import npy_paths, is_prepared
from datasets.noise import apply_noise


@DatasetRegistry.register("higgs_full")
def load(role: str, client_id: int = 0, **kwargs):
    """
    Carrega o dataset Higgs Full (11M amostras pre-processadas).

    Args:
        role: "server" retorna (X_test, y_test).
              "client" retorna (X_train_partition, y_train_partition, X_test, y_test).
        client_id: ID do cliente para particionar dados.
    """
    if not is_prepared("higgs_full"):
        print("ERRO: Dataset Higgs Full nao preparado.")
        print("  Execute: python tools/prepare_datasets.py --dataset higgs_full")
        print("  Requer: data/higgs_full/HIGGS.csv.gz (baixar de UCI)")
        sys.exit(1)

    x_path, y_path = npy_paths("higgs_full")
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
        print(f"[Servidor] Dataset Higgs Full: {X_test.shape[0]:,} amostras teste, "
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

    print(f"[Cliente {client_id}] Dataset Higgs Full: {len(my_idx):,} treino, "
          f"{len(y_test):,} teste")
    print(f"[Cliente {client_id}] Classe 0: {(y_client == 0).sum():,} "
          f"({(y_client == 0).mean()*100:.1f}%) | "
          f"Classe 1: {(y_client == 1).sum():,} "
          f"({(y_client == 1).mean()*100:.1f}%)")
    X_client, y_client = apply_noise(X_client, y_client, client_id, "higgs_full")
    return X_client, y_client, X_test, y_test
