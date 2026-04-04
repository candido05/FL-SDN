"""
Injecao de ruido nos dados de clientes FL.

Simula clientes com dados de baixa qualidade (sensores defeituosos, coleta
corrompida, etc.) para avaliar a robustez do mecanismo de selecao adaptativa.

Configurado em config.py:
    NOISE_ENABLED         — liga/desliga o mecanismo globalmente
    NOISE_DATASET         — dataset-alvo (string ou None para todos)
    NOISE_CLIENTS         — lista de client_ids que recebem ruido
    NOISE_LABEL_FLIP_RATE — fracao de labels invertidos (0.0 a 1.0)
    NOISE_FEATURE_STD     — desvio padrao do ruido gaussiano nas features

O ruido e aplicado APENAS nos dados de treino do cliente. Os dados de teste
(usados pelo servidor para avaliacao) permanecem limpos, o que garante que
a queda de desempenho reflita a degradacao real da contribuicao do cliente.
"""

import numpy as np

import config


def apply_noise(X, y, client_id, dataset_name):
    """
    Aplica ruido aos dados de treino de um cliente, se configurado.

    Args:
        X:            Array de features (shape: [n_samples, n_features]).
        y:            Array de labels (shape: [n_samples]).
        client_id:    ID do cliente.
        dataset_name: Nome do dataset (ex: "mnist", "higgs").

    Returns:
        (X_noisy, y_noisy) — copias modificadas se ruido ativado,
        ou (X, y) originais caso contrario.
    """
    if not getattr(config, "NOISE_ENABLED", False):
        return X, y

    target_dataset = getattr(config, "NOISE_DATASET", None)
    if target_dataset is not None and dataset_name != target_dataset:
        return X, y

    noise_clients = getattr(config, "NOISE_CLIENTS", [])
    if client_id not in noise_clients:
        return X, y

    flip_rate = getattr(config, "NOISE_LABEL_FLIP_RATE", 0.0)
    feature_std = getattr(config, "NOISE_FEATURE_STD", 0.0)

    rng = np.random.RandomState(config.RANDOM_SEED + client_id)
    X_out = X.copy().astype(np.float32)
    y_out = y.copy()

    n = len(y_out)
    n_flipped = 0
    if flip_rate > 0.0:
        n_flip = int(n * flip_rate)
        flip_idx = rng.choice(n, size=n_flip, replace=False)
        y_out[flip_idx] = 1 - y_out[flip_idx]
        n_flipped = n_flip

    if feature_std > 0.0:
        X_out += rng.normal(0.0, feature_std, X_out.shape).astype(np.float32)

    print(
        f"[RUIDO] Cliente {client_id} | dataset={dataset_name} | "
        f"labels invertidos={n_flipped}/{n} ({flip_rate*100:.0f}%) | "
        f"ruido_features=N(0,{feature_std})"
    )

    return X_out, y_out
