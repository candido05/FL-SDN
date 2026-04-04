"""
Dataset Higgs — classificacao binaria de eventos de particulas.

Fonte: OpenML (https://www.openml.org/d/23512)
- 50k amostras (configuravel via N_SAMPLES)
- 28 features numericas
- Labels: 0 (fundo) ou 1 (sinal Higgs)

Os arquivos .npy sao gerados por download_higgs.py e ficam em data/higgs/.
"""

import os

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from config import N_SAMPLES, TEST_SIZE, RANDOM_SEED, NUM_CLIENTS
from datasets.registry import DatasetRegistry, stratified_partition
from datasets.noise import apply_noise

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "higgs")
_X_PATH = os.path.join(_DATA_DIR, "higgs_X.npy")
_Y_PATH = os.path.join(_DATA_DIR, "higgs_y.npy")

# Compatibilidade: tambem procura na pasta data/ raiz (layout antigo)
_LEGACY_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
_LEGACY_X = os.path.join(_LEGACY_DIR, "higgs_X.npy")
_LEGACY_Y = os.path.join(_LEGACY_DIR, "higgs_y.npy")


def _load_raw():
    """Carrega dados brutos do cache local ou OpenML."""
    # Tenta novo layout: data/higgs/
    if os.path.exists(_X_PATH) and os.path.exists(_Y_PATH):
        X = np.load(_X_PATH)
        y = np.load(_Y_PATH).astype(int)
        return X, y

    # Fallback: layout antigo data/
    if os.path.exists(_LEGACY_X) and os.path.exists(_LEGACY_Y):
        X = np.load(_LEGACY_X)
        y = np.load(_LEGACY_Y).astype(int)
        return X, y

    # Download do OpenML
    higgs = fetch_openml(name="higgs", version=2, as_frame=False, parser="auto")
    X, y = higgs.data[:N_SAMPLES], higgs.target[:N_SAMPLES].astype(int)
    return X, y


@DatasetRegistry.register("higgs")
def load(role: str, client_id: int = 0, **kwargs):
    """
    Carrega o dataset Higgs.

    Args:
        role: "server" retorna (X_test, y_test).
              "client" retorna (X_train_partition, y_train_partition, X_test, y_test).
        client_id: ID do cliente para particionar dados.
    """
    X, y = _load_raw()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y,
    )

    if role == "server":
        print(f"[Servidor] Dataset Higgs: {X_test.shape[0]} amostras teste, "
              f"{X_test.shape[1]} features")
        print(f"[Servidor] Classe 0: {(y_test == 0).sum()} | "
              f"Classe 1: {(y_test == 1).sum()}")
        return X_test, y_test

    # role == "client"
    num_clients = kwargs.get("num_clients", NUM_CLIENTS)
    indices = stratified_partition(y_train, num_clients, RANDOM_SEED)
    my_idx = indices[client_id]
    X_client = X_train[my_idx]
    y_client = y_train[my_idx]

    print(f"[Cliente {client_id}] Dataset Higgs: {len(my_idx)} treino, "
          f"{len(y_test)} teste")
    print(f"[Cliente {client_id}] Classe 0: {(y_client == 0).sum()} "
          f"({(y_client == 0).mean()*100:.1f}%) | "
          f"Classe 1: {(y_client == 1).sum()} "
          f"({(y_client == 1).mean()*100:.1f}%)")
    X_client, y_client = apply_noise(X_client, y_client, client_id, "higgs")
    return X_client, y_client, X_test, y_test
