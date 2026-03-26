"""
Dataset Credit Card Fraud Detection — classificacao binaria de transacoes fraudulentas.

Fonte: ULB (Universidade Livre de Bruxelas) via OpenML/Kaggle
- 284.807 transacoes
- 30 features: V1-V28 (resultado de PCA anonimizado) + Amount normalizado + Hour
- Label: 0 (legitima) ou 1 (fraude)
- Altamente desbalanceado: ~0.17% fraudes

Os arquivos .npy sao gerados por prepare_datasets.py e ficam em data/creditcard/.

Preprocessing aplicado (tudo data-independent, compativel com FL):
  1. Remocao da coluna Time (sequencial, sem sentido em FL particionado)
  2. Criacao de feature Hour (hora ciclica derivada de Time, data-independent)
  3. Normalizacao de Amount via log1p (data-independent, sem estatisticas globais)
  4. Features V1-V28 mantidas como estao (ja PCA-normalizadas)
"""

import sys

import numpy as np
from sklearn.model_selection import train_test_split

from config import TEST_SIZE, RANDOM_SEED, NUM_CLIENTS
from datasets.registry import DatasetRegistry, stratified_partition
from datasets.paths import npy_paths, is_prepared


@DatasetRegistry.register("creditcard")
def load(role: str, client_id: int = 0, **kwargs):
    """
    Carrega o dataset Credit Card Fraud Detection.

    Args:
        role: "server" retorna (X_test, y_test).
              "client" retorna (X_train_partition, y_train_partition, X_test, y_test).
        client_id: ID do cliente para particionar dados.
    """
    if not is_prepared("creditcard"):
        print("ERRO: Dataset Credit Card nao preparado.")
        print("  Execute: python tools/prepare_datasets.py --dataset creditcard")
        sys.exit(1)

    x_path, y_path = npy_paths("creditcard")
    X = np.load(x_path)
    y = np.load(y_path).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y,
    )

    if role == "server":
        n_fraud = (y_test == 1).sum()
        n_total = len(y_test)
        print(f"[Servidor] Dataset CreditCard: {n_total:,} amostras teste, "
              f"{X_test.shape[1]} features")
        print(f"[Servidor] Legitimas: {(y_test == 0).sum():,} | "
              f"Fraudes: {n_fraud:,} ({n_fraud/n_total*100:.3f}%)")
        return X_test, y_test

    # role == "client"
    num_clients = kwargs.get("num_clients", NUM_CLIENTS)
    indices = stratified_partition(y_train, num_clients, RANDOM_SEED)
    my_idx = indices[client_id]
    X_client = X_train[my_idx]
    y_client = y_train[my_idx]

    n_fraud = (y_client == 1).sum()
    n_total = len(y_client)
    print(f"[Cliente {client_id}] Dataset CreditCard: {n_total:,} treino, "
          f"{len(y_test):,} teste")
    print(f"[Cliente {client_id}] Legitimas: {(y_client == 0).sum():,} "
          f"({(y_client == 0).mean()*100:.2f}%) | "
          f"Fraudes: {n_fraud:,} "
          f"({n_fraud/n_total*100:.3f}%)")
    return X_client, y_client, X_test, y_test
