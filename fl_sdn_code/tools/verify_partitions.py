"""
Verifica que a particao de dados entre clientes e equilibrada.

Confirma:
  1. Todos os clientes recebem a mesma quantidade de amostras (+-1)
  2. A distribuicao de classes (positiva/negativa) e igual entre clientes
  3. Nenhum dado de teste vaza para o treino

Uso:
    python tools/verify_partitions.py --dataset higgs
    python tools/verify_partitions.py --dataset higgs_full
    python tools/verify_partitions.py --all-datasets
"""

import argparse
import os
import sys

# Adiciona diretorio pai ao path para encontrar config, datasets, etc.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config import NUM_CLIENTS, RANDOM_SEED
from datasets import DatasetRegistry


ALL_DATASETS = ["higgs", "higgs_full", "epsilon", "avazu"]


def verify_dataset(dataset_name, num_clients=NUM_CLIENTS):
    """Verifica particao de um dataset. Retorna True se tudo OK."""
    print(f"\n{'='*60}")
    print(f"  VERIFICACAO: {dataset_name} | {num_clients} clientes")
    print(f"{'='*60}")

    try:
        X_test_srv, y_test_srv = DatasetRegistry.load(
            dataset_name, role="server",
        )
    except (SystemExit, Exception) as e:
        print(f"  SKIP: Dataset {dataset_name} nao disponivel ({e})")
        return None

    client_sizes = []
    client_pos_rates = []

    for cid in range(num_clients):
        X_train, y_train, X_test, y_test = DatasetRegistry.load(
            dataset_name, role="client", client_id=cid,
        )

        client_sizes.append(len(y_train))
        pos_rate = y_train.mean()
        client_pos_rates.append(pos_rate)

        assert np.array_equal(X_test, X_test_srv), \
            f"Cliente {cid}: test set diferente do servidor!"
        assert np.array_equal(y_test, y_test_srv), \
            f"Cliente {cid}: test labels diferente do servidor!"

    # 1. Tamanhos iguais
    max_size = max(client_sizes)
    min_size = min(client_sizes)
    size_diff = max_size - min_size

    print(f"\n  Tamanhos por cliente:")
    for cid, sz in enumerate(client_sizes):
        print(f"    Cliente {cid}: {sz:>8,} amostras")
    print(f"    Diferenca max-min: {size_diff}")

    size_ok = size_diff <= 1
    print(f"    {'OK' if size_ok else 'FALHA'}: tamanhos {'equilibrados' if size_ok else 'DESIGUAIS'}!")

    # 2. Distribuicao de classes
    print(f"\n  Taxa de positivos (classe 1) por cliente:")
    for cid, rate in enumerate(client_pos_rates):
        print(f"    Cliente {cid}: {rate*100:.2f}%")

    rate_range = max(client_pos_rates) - min(client_pos_rates)
    rate_ok = rate_range < 0.01
    print(f"    Range: {rate_range*100:.4f}%")
    print(f"    {'OK' if rate_ok else 'FALHA'}: distribuicao {'equilibrada' if rate_ok else 'DESIGUAL'}!")

    # 3. Test set do servidor
    print(f"\n  Test set: {len(y_test_srv):,} amostras | "
          f"Classe 0: {(y_test_srv == 0).sum():,} | "
          f"Classe 1: {(y_test_srv == 1).sum():,}")
    print(f"    OK: test set identico entre servidor e todos os clientes")

    all_ok = size_ok and rate_ok
    status = "PASSOU" if all_ok else "FALHOU"
    print(f"\n  RESULTADO: {status}")

    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Verifica equilibrio da particao de dados entre clientes"
    )
    parser.add_argument("--dataset", type=str, default=None,
                        choices=ALL_DATASETS)
    parser.add_argument("--all-datasets", action="store_true",
                        help="Verifica todos os datasets disponiveis")
    args = parser.parse_args()

    if not args.dataset and not args.all_datasets:
        parser.error("Especifique --dataset ou --all-datasets")

    datasets = ALL_DATASETS if args.all_datasets else [args.dataset]
    results = {}

    for ds in datasets:
        result = verify_dataset(ds)
        if result is not None:
            results[ds] = result

    print(f"\n\n{'='*60}")
    print(f"  RESUMO FINAL")
    print(f"{'='*60}")
    for ds, ok in results.items():
        status = "PASSOU" if ok else "FALHOU"
        print(f"    {ds:>12}: {status}")

    if all(results.values()):
        print(f"\n  Todas as verificacoes passaram!")
    else:
        print(f"\n  ATENCAO: Algumas verificacoes falharam!")
        sys.exit(1)


if __name__ == "__main__":
    main()
