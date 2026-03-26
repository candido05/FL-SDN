"""
Dataset MNIST — classificacao binaria de digitos manuscritos.

Fonte: OpenML (mnist_784) / sklearn.datasets.fetch_openml
- 70k amostras (60k treino + 10k teste)
- 784 features originais (28x28 pixels)
- Convertido para classificacao binaria: digitos 0-4 (classe 0) vs 5-9 (classe 1)

Os arquivos .npy sao gerados por prepare_datasets.py e ficam em data/mnist/.

Preprocessing aplicado (tudo data-independent, compativel com FL):
  1. Normalizacao [0, 1] (divisao por 255)
  2. Remocao de pixels constantes (bordas sempre pretas)
  3. Remocao de features com variancia muito baixa (< 0.01)
  4. Binarizacao das labels: 0-4 → classe 0, 5-9 → classe 1
"""

import sys

import numpy as np
from sklearn.model_selection import train_test_split

from config import TEST_SIZE, RANDOM_SEED, NUM_CLIENTS
from datasets.registry import DatasetRegistry, stratified_partition
from datasets.paths import npy_paths, is_prepared


@DatasetRegistry.register("mnist")
def load(role: str, client_id: int = 0, **kwargs):
    """
    Carrega o dataset MNIST (binario: 0-4 vs 5-9).

    Args:
        role: "server" retorna (X_test, y_test).
              "client" retorna (X_train_partition, y_train_partition, X_test, y_test).
        client_id: ID do cliente para particionar dados.
    """
    if not is_prepared("mnist"):
        print("ERRO: Dataset MNIST nao preparado.")
        print("  Execute: python tools/prepare_datasets.py --dataset mnist")
        sys.exit(1)

    x_path, y_path = npy_paths("mnist")
    X = np.load(x_path)
    y = np.load(y_path).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y,
    )

    if role == "server":
        print(f"[Servidor] Dataset MNIST: {X_test.shape[0]:,} amostras teste, "
              f"{X_test.shape[1]} features")
        print(f"[Servidor] Classe 0 (digitos 0-4): {(y_test == 0).sum():,} | "
              f"Classe 1 (digitos 5-9): {(y_test == 1).sum():,}")
        return X_test, y_test

    # role == "client"
    num_clients = kwargs.get("num_clients", NUM_CLIENTS)
    indices = stratified_partition(y_train, num_clients, RANDOM_SEED)
    my_idx = indices[client_id]
    X_client = X_train[my_idx]
    y_client = y_train[my_idx]

    print(f"[Cliente {client_id}] Dataset MNIST: {len(my_idx):,} treino, "
          f"{len(y_test):,} teste")
    print(f"[Cliente {client_id}] Classe 0 (0-4): {(y_client == 0).sum():,} "
          f"({(y_client == 0).mean()*100:.1f}%) | "
          f"Classe 1 (5-9): {(y_client == 1).sum():,} "
          f"({(y_client == 1).mean()*100:.1f}%)")
    return X_client, y_client, X_test, y_test
