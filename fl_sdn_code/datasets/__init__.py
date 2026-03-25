"""
Registry de datasets para Federated Learning.

Para adicionar um novo dataset:
1. Crie um arquivo em datasets/ (ex: datasets/mnist.py)
2. Implemente uma funcao load(role, client_id, num_clients, **kwargs)
   que retorne os mesmos formatos de higgs.py
3. Registre com @DatasetRegistry.register("nome")

Uso:
    from datasets import DatasetRegistry
    X_test, y_test = DatasetRegistry.load("higgs", role="server")
"""

from datasets.registry import DatasetRegistry

# Importa modulos de dataset para que se auto-registrem
import datasets.higgs
import datasets.higgs_full
import datasets.epsilon
import datasets.avazu
