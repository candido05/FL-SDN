"""
Registry Pattern para datasets.

Permite adicionar novos datasets sem modificar codigo existente.
"""

import numpy as np


def stratified_partition(y, num_parts, seed):
    """
    Particiona indices de forma estratificada, garantindo que cada parte
    tenha a mesma distribuicao de classes.

    Args:
        y: Array de labels.
        num_parts: Numero de particoes (= num_clients).
        seed: Seed para reproducibilidade.

    Returns:
        Lista de arrays de indices, um por particao.
    """
    rng = np.random.RandomState(seed)
    classes = np.unique(y)
    parts = [[] for _ in range(num_parts)]
    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        rng.shuffle(cls_indices)
        splits = np.array_split(cls_indices, num_parts)
        for i, split in enumerate(splits):
            parts[i].append(split)
    return [np.concatenate(p) for p in parts]


class DatasetRegistry:
    """Registry central de datasets disponveis."""

    _DATASETS = {}

    @classmethod
    def register(cls, name: str):
        """Decorator para registrar um loader de dataset."""
        def decorator(fn):
            cls._DATASETS[name] = fn
            return fn
        return decorator

    @classmethod
    def load(cls, name: str, role: str, client_id: int = 0, **kwargs):
        """
        Carrega dataset pelo nome.

        Args:
            name: Nome registrado do dataset (ex: "higgs").
            role: "server" ou "client".
            client_id: ID do cliente (usado apenas se role="client").
            **kwargs: Argumentos extras passados ao loader.

        Returns:
            Se role="server": (X_test, y_test)
            Se role="client": (X_train, y_train, X_test, y_test)
        """
        loader = cls._DATASETS.get(name)
        if loader is None:
            raise ValueError(
                f"Dataset desconhecido: '{name}'. "
                f"Disponiveis: {cls.available()}"
            )
        return loader(role=role, client_id=client_id, **kwargs)

    @classmethod
    def available(cls) -> list:
        """Lista nomes de datasets registrados."""
        return list(cls._DATASETS.keys())
