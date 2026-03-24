"""
Estrategias de Federated Learning.

- SimpleBagging:  Todos treinam em paralelo, ensemble por media
- SimpleCycling:  Um por vez, round-robin sequencial
- SDNBagging:     Bagging com selecao por metricas de rede
- SDNCycling:     Cycling com selecao adaptativa por rede
"""

from strategies.bagging import SimpleBagging
from strategies.cycling import SimpleCycling
from strategies.sdn_bagging import SDNBagging
from strategies.sdn_cycling import SDNCycling

STRATEGY_MAP = {
    "bagging":     SimpleBagging,
    "cycling":     SimpleCycling,
    "sdn-bagging": SDNBagging,
    "sdn-cycling": SDNCycling,
}


def create_strategy(name: str, num_clients: int, X_test, y_test, logger=None):
    """
    Factory para criar estrategia por nome.

    Args:
        name: Nome da estrategia (bagging, cycling, sdn-bagging, sdn-cycling).
        num_clients: Numero de clientes federados.
        X_test, y_test: Dados de teste para avaliacao server-side.
        logger: CSVLogger para registro de metricas (injetado pelo server.py).
    """
    cls = STRATEGY_MAP.get(name)
    if cls is None:
        raise ValueError(f"Estrategia desconhecida: {name}. "
                         f"Disponiveis: {list(STRATEGY_MAP.keys())}")
    strategy = cls(num_clients, X_test, y_test)
    if logger is not None:
        strategy.set_logger(logger)
    return strategy
