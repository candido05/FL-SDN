"""Preparação offline do VeReMi (M0).

Especificação: ESPEC_M0_FASE_OFFLINE.md (o que fazer).
Guia:          GUIA_M0_FASE_OFFLINE.md  (por que fazer).

ETAPA 1 (esta versão): passos 1, 2 e 3 da especificação.
    1. conferir o SHA-256 do CSV
    2. ler as 21 colunas necessárias, com dtypes fixos
    3. conferir a regra "cada veículo tem uma única classe"
Os passos 4 a 10 serão acrescentados nas próximas etapas.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constantes (seção 3 da especificação)
# ---------------------------------------------------------------------------

# 16 features de conteúdo, NESTA ORDEM (define a ordem das colunas de X).
CONTENT_FEATURES = [
    "posx", "posy", "spdx", "spdy", "aclx", "acly", "hedx", "hedy",
    "posx_n", "posy_n", "spdx_n", "spdy_n", "aclx_n", "acly_n", "hedx_n", "hedy_n",
]

# Colunas de controle: usadas nos cálculos, NUNCA como feature.
CONTROL_COLUMNS = ["sender", "senderPseudo", "sendTime", "messageID"]

LABEL_COLUMN = "class"

# dtype de cada coluna lida (passo 2). sendTime em float64 de propósito:
# em float32 a resolução perto de 86.400 s cai para ~8 ms.
DTYPES = {
    **{c: np.float32 for c in CONTENT_FEATURES},
    "sendTime": np.float64,
    "sender": np.int64,
    "senderPseudo": np.int64,
    "messageID": np.int64,
    LABEL_COLUMN: np.int16,
}

EXPECTED_ROWS = 3_194_808   # linhas do CSV real (seção 9)
CHUNK_ROWS = 500_000        # linhas por pedaço na leitura
HASH_BLOCK = 16 * 1024 * 1024


class PreparationError(Exception):
    """Uma regra da preparação falhou. O resultado não é confiável."""


# ---------------------------------------------------------------------------
# Passo 1 — hash do CSV
# ---------------------------------------------------------------------------

def sha256_file(path: str) -> str:
    """SHA-256 de um arquivo, lido em blocos de 16 MB."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for bloco in iter(lambda: f.read(HASH_BLOCK), b""):
            h.update(bloco)
    return h.hexdigest()


def check_sha256(path: str, expected: str) -> str:
    obtained = sha256_file(path)
    if obtained != expected.lower():
        raise PreparationError(
            f"SHA-256 do CSV não confere: esperado {expected}, obtido {obtained}"
        )
    return obtained


# ---------------------------------------------------------------------------
# Passo 2 — leitura
# ---------------------------------------------------------------------------

def read_csv(path: str, expected_rows: int | None = EXPECTED_ROWS) -> pd.DataFrame:
    """Lê só as 21 colunas necessárias, em pedaços, com dtypes fixos.

    expected_rows=None desliga a conferência de tamanho (usado nos testes
    com CSVs sintéticos pequenos).
    """
    chunks = pd.read_csv(
        path,
        usecols=list(DTYPES),
        dtype=DTYPES,
        chunksize=CHUNK_ROWS,
    )
    df = pd.concat(chunks, ignore_index=True)

    if expected_rows is not None and len(df) != expected_rows:
        raise PreparationError(
            f"Número de linhas lidas não confere: esperado {expected_rows}, obtido {len(df)}"
        )
    return df


# ---------------------------------------------------------------------------
# Passo 3 — cada veículo tem uma única classe
# ---------------------------------------------------------------------------

def check_one_class_per_sender(df: pd.DataFrame) -> None:
    n_classes = df.groupby("sender")[LABEL_COLUMN].nunique()
    bad = n_classes[n_classes > 1]
    if len(bad) > 0:
        first = bad.index[:10]
        classes = (
            df[df["sender"].isin(first)]
            .groupby("sender")[LABEL_COLUMN]
            .unique()
        )
        details = "; ".join(f"sender {s}: classes {sorted(c.tolist())}" for s, c in classes.items())
        raise PreparationError(
            f"Regra '1 classe por veículo' violada em {len(bad)} veículo(s). "
            f"Primeiros: {details}"
        )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preparação offline do VeReMi (M0).")
    p.add_argument("--input", required=True, help="caminho do CSV bruto")
    p.add_argument("--output", required=True, help="diretório de saída (não pode existir)")
    p.add_argument("--expected-sha256", required=True, help="SHA-256 esperado do CSV")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    t0 = time.perf_counter()
    try:
        print("[1] Conferindo SHA-256 do CSV...")
        check_sha256(args.input, args.expected_sha256)
        print("    ok")

        print("[2] Lendo o CSV...")
        df = read_csv(args.input)
        mem_mb = df.memory_usage(deep=True).sum() / 1e6
        print(f"    ok: {len(df):,} linhas, {df.shape[1]} colunas, {mem_mb:.0f} MB em memória")

        print("[3] Conferindo '1 classe por veículo'...")
        check_one_class_per_sender(df)
        print(f"    ok: {df['sender'].nunique():,} veículos, nenhuma violação")

    except PreparationError as e:
        print(f"ERRO: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Etapa 1 concluída em {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
