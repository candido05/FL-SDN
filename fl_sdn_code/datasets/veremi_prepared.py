"""Leitura segura do VeReMi preparado pelo M0 (seção 6 da especificação).

REGRA: nenhum outro código do projeto abre os .npy de data/veremi/prepared_vN/
diretamente. Todos usam load_prepared(), que confere versão, hashes, formas e
dtypes antes de entregar os dados. Assim, um arquivo corrompido, de outra
versão ou com colunas em outra ordem dá erro na hora, e não um resultado
estranho semanas depois.

Este módulo NÃO importa nada de tools/prepare_veremi.py: o único contrato entre
os dois é o diretório gravado (os .npy + manifest.json).

Uso:
    from datasets.veremi_prepared import load_prepared
    data = load_prepared("data/veremi/prepared_v1")
    data.X, data.y, data.tipo, data.familia, data.sender
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np

SUPPORTED_SCHEMA_VERSIONS = {"1.0"}

# Nomes das colunas de X, NA ORDEM. Tem que bater com FEATURE_NAMES de
# tools/prepare_veremi.py. A lista é repetida aqui de propósito: este módulo
# não importa nada do script de preparação (o contrato é o diretório gravado).
# Mudou a lista ou a ordem -> nova schema_version e novo diretório prepared_vN.
EXPECTED_FEATURE_NAMES = [
    "posx", "posy", "spdx", "spdy", "aclx", "acly", "hedx", "hedy",
    "posx_n", "posy_n", "spdx_n", "spdy_n", "aclx_n", "acly_n", "hedx_n", "hedy_n",
    "dt_prev", "rate_5s", "pos_resid", "spd_resid",
]
N_FEATURES = len(EXPECTED_FEATURE_NAMES)

# Forma (em função de N) e dtype de cada array (tabela do passo 10).
EXPECTED_ARRAYS = {
    "X": (np.float32, 2),
    "y": (np.int8, 1),
    "tipo": (np.int8, 1),
    "familia": (np.int8, 1),
    "sender": (np.int64, 1),
}

HASH_BLOCK = 16 * 1024 * 1024


class PreparedDataError(Exception):
    """Os dados preparados não passaram numa conferência. Não use."""


@dataclass(frozen=True)
class PreparedVeremi:
    X: np.ndarray            # (N, 20) float32
    y: np.ndarray            # (N,) int8: 0 = benigno, 1 = ataque
    tipo: np.ndarray         # (N,) int8: class original (0, 1, 3-19)
    familia: np.ndarray      # (N,) int8: 0-5 (ver manifest["family_names"])
    sender: np.ndarray       # (N,) int64: SÓ para dividir por veículo; nunca é feature
    feature_names: list[str]
    manifest: dict


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for bloco in iter(lambda: f.read(HASH_BLOCK), b""):
            h.update(bloco)
    return h.hexdigest()


def load_prepared(path: str, verify_hashes: bool = True, mmap: bool = True) -> PreparedVeremi:
    """Carrega o diretório preparado, conferindo tudo antes de devolver.

    verify_hashes: recalcula o SHA-256 de cada .npy e compara com o manifesto.
    mmap: abre os arrays sem copiá-los para a RAM (somente leitura,
          compartilhados entre processos).
    """
    # 1) Manifesto e versão.
    manifest_path = os.path.join(path, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise PreparedDataError(f"manifest.json não encontrado em {path}")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    version = manifest.get("schema_version")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise PreparedDataError(
            f"schema_version {version!r} não suportada (suportadas: "
            f"{sorted(SUPPORTED_SCHEMA_VERSIONS)})"
        )

    files = manifest.get("files", {})
    for name in EXPECTED_ARRAYS:
        fname = f"{name}.npy"
        if fname not in files:
            raise PreparedDataError(f"{fname} não está listado no manifesto")
        if not os.path.isfile(os.path.join(path, fname)):
            raise PreparedDataError(f"{fname} não encontrado em {path}")

    # 2) Hashes.
    if verify_hashes:
        for name in EXPECTED_ARRAYS:
            fname = f"{name}.npy"
            obtained = _sha256_file(os.path.join(path, fname))
            expected = files[fname]["sha256"]
            if obtained != expected:
                raise PreparedDataError(
                    f"SHA-256 de {fname} não confere: esperado {expected}, obtido {obtained}"
                )

    # 3) Carregar.
    mode = "r" if mmap else None
    arrays = {
        name: np.load(os.path.join(path, f"{name}.npy"), mmap_mode=mode, allow_pickle=False)
        for name in EXPECTED_ARRAYS
    }

    # 4) Formas, dtypes e mesmo N.
    n = arrays["X"].shape[0]
    for name, (dtype, ndim) in EXPECTED_ARRAYS.items():
        arr = arrays[name]
        if arr.dtype != np.dtype(dtype):
            raise PreparedDataError(f"{name}: dtype {arr.dtype}, esperado {np.dtype(dtype)}")
        if arr.ndim != ndim:
            raise PreparedDataError(f"{name}: {arr.ndim} dimensões, esperado {ndim}")
        if arr.shape[0] != n:
            raise PreparedDataError(
                f"{name} tem {arr.shape[0]} linhas, mas X tem {n}: arrays desalinhados"
            )
        if list(arr.shape) != files[f"{name}.npy"].get("shape"):
            raise PreparedDataError(
                f"{name}: forma {list(arr.shape)} diferente da registrada no manifesto "
                f"{files[f'{name}.npy'].get('shape')}"
            )

    # 5a) Nomes E ordem das features: a coluna j de X tem que ser a feature
    #     que o código de treino espera na posição j.
    feature_names = manifest.get("feature_names")
    if feature_names != EXPECTED_FEATURE_NAMES:
        raise PreparedDataError(
            f"feature_names do manifesto não confere (nomes ou ordem). "
            f"Esperado: {EXPECTED_FEATURE_NAMES}. Manifesto: {feature_names}"
        )

    # 5b) Número de colunas de X.
    if arrays["X"].shape[1] != N_FEATURES:
        raise PreparedDataError(
            f"X tem {arrays['X'].shape[1]} colunas, esperado {N_FEATURES}"
        )

    # 6) Devolver.
    return PreparedVeremi(
        X=arrays["X"],
        y=arrays["y"],
        tipo=arrays["tipo"],
        familia=arrays["familia"],
        sender=arrays["sender"],
        feature_names=list(feature_names),
        manifest=manifest,
    )
