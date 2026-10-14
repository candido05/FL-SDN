"""Preparação offline do VeReMi (M0).

Especificação: ESPEC_M0_FASE_OFFLINE.md (o que fazer).
Guia:          GUIA_M0_FASE_OFFLINE.md  (por que fazer).

ETAPA 3 (esta versão): passos 1 a 8 da especificação.
    1. conferir o SHA-256 do CSV
    2. ler as 21 colunas necessárias, com dtypes fixos
    3. conferir a regra "cada veículo tem uma única classe"
    4. remover os veículos do ataque tipo 2
    5. remover as linhas do pseudônimo 1 (só as linhas)
    6. conferir a regra "cada pseudônimo pertence a um único veículo"
    7. criar os rótulos y, tipo e familia
    8. calcular as 4 features temporais (causais, por pseudônimo)
Os passos 9 e 10 serão acrescentados na próxima etapa.
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

EXCLUDED_ATTACK_TYPE = 2    # Constant Position Offset (passo 4)
BROKEN_PSEUDONYM = 1        # defeito do gerador (passo 5)

# Passo 7: nome de cada família e os tipos (class) que ela agrupa.
# Definido UMA vez aqui; FAMILY_MAP é derivado dele e vai para o manifesto.
FAMILIES = {
    0: ("benigno", [0]),
    1: ("POS", [1, 3, 4]),
    2: ("SPD", [5, 6, 7, 8]),
    3: ("BEH", [9, 10, 11, 12]),
    4: ("DOS", [13, 14, 15]),
    5: ("SYB", [16, 17, 18, 19]),
}
FAMILY_MAP = {tipo: fam for fam, (_, tipos) in FAMILIES.items() for tipo in tipos}
FAMILY_NAMES = {fam: nome for fam, (nome, _) in FAMILIES.items()}

# Passo 8: as 4 features temporais, nesta ordem (colunas 16-19 de X).
TEMPORAL_FEATURES = ["dt_prev", "rate_5s", "pos_resid", "spd_resid"]
RATE_WINDOW_S = 5.0

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
# Passo 4 — remover os veículos do ataque tipo 2
# ---------------------------------------------------------------------------

def remove_attack_type_2(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove TODAS as linhas dos veículos com class == 2.

    Como o passo 3 garantiu 1 classe por veículo, filtrar pela classe
    equivale a remover os veículos inteiros.
    Devolve o DataFrame filtrado e as contagens para o manifesto.
    """
    mask = df[LABEL_COLUMN] == EXCLUDED_ATTACK_TYPE
    stats = {
        "rows_removed": int(mask.sum()),
        "vehicles_removed": int(df.loc[mask, "sender"].nunique()),
    }
    return df[~mask].reset_index(drop=True), stats


# ---------------------------------------------------------------------------
# Passo 5 — remover as linhas do pseudônimo 1
# ---------------------------------------------------------------------------

def remove_pseudonym_1(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove SÓ as linhas com senderPseudo == 1.

    Os veículos que usaram esse pseudônimo continuam no dataset com as
    mensagens enviadas sob os outros pseudônimos.
    """
    mask = df["senderPseudo"] == BROKEN_PSEUDONYM
    stats = {
        "rows_removed": int(mask.sum()),
        "vehicles_affected": int(df.loc[mask, "sender"].nunique()),
    }
    return df[~mask].reset_index(drop=True), stats


# ---------------------------------------------------------------------------
# Passo 6 — cada pseudônimo pertence a um único veículo
# ---------------------------------------------------------------------------

def check_one_sender_per_pseudonym(df: pd.DataFrame) -> None:
    n_senders = df.groupby("senderPseudo")["sender"].nunique()
    bad = n_senders[n_senders > 1]
    if len(bad) > 0:
        details = "; ".join(f"pseudônimo {p}: {n} veículos" for p, n in bad.head(10).items())
        raise PreparationError(
            f"Regra '1 veículo por pseudônimo' violada em {len(bad)} pseudônimo(s). "
            f"Primeiros: {details}"
        )


# ---------------------------------------------------------------------------
# Passo 7 — rótulos
# ---------------------------------------------------------------------------

def make_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Devolve um DataFrame com y, tipo e familia, alinhado linha a linha com df."""
    classes = df[LABEL_COLUMN]
    unknown = sorted(set(classes.unique()) - set(FAMILY_MAP))
    if (classes == EXCLUDED_ATTACK_TYPE).any():
        raise PreparationError("Ainda há linhas de tipo 2: o passo 4 não rodou antes do 7")
    if unknown:
        raise PreparationError(f"Valores de class sem família: {unknown}")

    return pd.DataFrame({
        "y": (classes != 0).astype(np.int8),
        "tipo": classes.astype(np.int8),
        "familia": classes.map(FAMILY_MAP).astype(np.int8),
    }, index=df.index)


# ---------------------------------------------------------------------------
# Passo 8 — features temporais
# ---------------------------------------------------------------------------

def compute_temporal_features(df: pd.DataFrame, window_s: float = RATE_WINDOW_S) -> pd.DataFrame:
    """Calcula dt_prev, rate_5s, pos_resid e spd_resid.

    Regras (seção 8.2 da especificação):
      - agrupa por senderPseudo (nunca por sender);
      - ordena cada grupo por (sendTime, messageID), ordenação estável;
      - usa só mensagens ANTERIORES à atual (causal).
    Devolve as 4 colunas na ORDEM ORIGINAL das linhas de df.
    """
    n = len(df)

    # 1) Ordena por (pseudônimo, tempo, messageID), estável.
    #    row_id guarda a posição original de cada linha, para desfazer a ordenação no fim.
    order = (
        df[["senderPseudo", "sendTime", "messageID"]]
        .assign(row_id=np.arange(n))
        .sort_values(["senderPseudo", "sendTime", "messageID"], kind="mergesort")
    )
    row_id = order["row_id"].to_numpy()

    pseudo = df["senderPseudo"].to_numpy()[row_id]
    t = df["sendTime"].to_numpy(dtype=np.float64)[row_id]
    # Contas em float64 para não perder precisão; o resultado vira float32 em X.
    col = lambda c: df[c].to_numpy(dtype=np.float64)[row_id]
    posx, posy = col("posx"), col("posy")
    spdx, spdy = col("spdx"), col("spdy")
    aclx, acly = col("aclx"), col("acly")

    # 2) first[i] = True se a linha i é a primeira do seu pseudônimo.
    #  detecção vetorizada de mudanças (change-point detection)
    first = np.ones(n, dtype=bool)
    first[1:] = pseudo[1:] != pseudo[:-1]

    # 3) "Valor da mensagem anterior": desloca o array 1 posição para a direita.
    #    Na primeira linha de cada pseudônimo, o anterior não existe -> NaN.
    def prev(a):
        out = np.empty_like(a)
        out[0] = np.nan
        out[1:] = a[:-1] # shift right
        out[first] = np.nan
        return out

    dt_prev = t - prev(t)

    # pos prevista = pos_anterior + vel_anterior * dt
    dx = posx - (prev(posx) + prev(spdx) * dt_prev)
    dy = posy - (prev(posy) + prev(spdy) * dt_prev)
    pos_resid = np.sqrt(dx * dx + dy * dy)

    # vel prevista = vel_anterior + acl_anterior * dt
    dx = spdx - (prev(spdx) + prev(aclx) * dt_prev)
    dy = spdy - (prev(spdy) + prev(acly) * dt_prev)
    spd_resid = np.sqrt(dx * dx + dy * dy)

    # 4) rate_5s: mensagens do mesmo pseudônimo com t_atual - 5 <= t < t_atual.
    #    Dentro de cada grupo (já ordenado por tempo):
    #      ini = primeira mensagem com t >= t_atual - 5
    #      fim = primeira mensagem com t >= t_atual   (exclui a atual e empates)
    #    contagem = fim - ini
    #    técnica: contagem em intervalos usando busca binária
    rate = np.zeros(n, dtype=np.float64)
    starts = np.flatnonzero(first)
    ends = np.append(starts[1:], n)
    for a, b in zip(starts, ends):
        tg = t[a:b]
        ini = np.searchsorted(tg, tg - window_s, side="left")
        fim = np.searchsorted(tg, tg, side="left")
        rate[a:b] = fim - ini

    # 5) Volta para a ordem original das linhas.
    result = np.empty((n, 4), dtype=np.float64)
    result[row_id] = np.column_stack([dt_prev, rate, pos_resid, spd_resid])
    return pd.DataFrame(result, columns=TEMPORAL_FEATURES, index=df.index)


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

        print("[4] Removendo os veículos do ataque tipo 2...")
        df, stats_t2 = remove_attack_type_2(df)
        print(f"    removidas {stats_t2['rows_removed']:,} linhas, "
              f"{stats_t2['vehicles_removed']:,} veículos "
              f"(esperado: 43,567 linhas, 390 veículos)")

        print("[5] Removendo as linhas do pseudônimo 1...")
        df, stats_p1 = remove_pseudonym_1(df)
        print(f"    removidas {stats_p1['rows_removed']:,} linhas, "
              f"de {stats_p1['vehicles_affected']:,} veículos "
              f"(esperado: 40,853 linhas, 377 veículos)")

        print("[6] Conferindo '1 veículo por pseudônimo'...")
        check_one_sender_per_pseudonym(df)
        print(f"    ok: {df['senderPseudo'].nunique():,} pseudônimos, nenhuma violação "
              f"(esperado: 118,518)")

        print(f"Restam {len(df):,} linhas (esperado: 3,110,388) "
              f"e {df['sender'].nunique():,} veículos (esperado: 24,273)")

        print("[7] Criando os rótulos...")
        labels = make_labels(df)
        print(f"    fração benigna: {(labels['y'] == 0).mean():.3f} (esperado: 0.611)")
        print("    família     linhas (esperado)        veículos (esperado)")
        exp_rows = [1_900_539, 130_085, 171_125, 173_509, 387_299, 347_831]
        exp_veh = [17_264, 1_169, 1_557, 1_558, 1_168, 1_557]
        for fam, nome in FAMILY_NAMES.items():
            m = labels["familia"] == fam
            print(f"    {fam} {nome:8s} {int(m.sum()):>10,} ({exp_rows[fam]:>9,})"
                  f"   {df.loc[m, 'sender'].nunique():>8,} ({exp_veh[fam]:>6,})")

        print("[8] Calculando as features temporais...")
        t8 = time.perf_counter()
        temporal = compute_temporal_features(df)
        print(f"    ok em {time.perf_counter() - t8:.1f} s")
        for c in TEMPORAL_FEATURES:
            print(f"    {c:10s} NaN: {temporal[c].isna().mean():.4f}")
        print("    (esperado: 0.0381 em dt_prev, pos_resid, spd_resid; 0 em rate_5s)")
        n_ties = int((temporal["dt_prev"] == 0).sum())
        print(f"    mensagens com dt_prev == 0 (empate de tempo no mesmo pseudônimo): {n_ties:,}")

    except PreparationError as e:
        print(f"ERRO: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Etapa 3 concluída em {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
