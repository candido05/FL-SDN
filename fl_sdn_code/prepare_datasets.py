"""
Pre-processamento dos datasets para Federated Learning.

Converte os arquivos brutos baixados pelo Joao Victor em .npy
prontos para uso pelo FL-SDN.

Uso:
    python prepare_datasets.py --dataset higgs_full
    python prepare_datasets.py --dataset epsilon
    python prepare_datasets.py --dataset avazu
    python prepare_datasets.py --all          # prepara todos os disponiveis

Requisitos:
    pip install pandas scikit-learn

Antes de executar, coloque os arquivos brutos nas pastas:
    data/higgs_full/HIGGS.csv.gz
    data/epsilon/epsilon_normalized.bz2 + epsilon_normalized.t.bz2
    data/avazu/train.csv (ou train.gz)
"""

import argparse
import os
import sys
import time

import numpy as np

from datasets.paths import data_dir, npy_paths, is_prepared, DATASET_INFO


# ======================================================================
# HIGGS Full (11M amostras, 28 features)
# ======================================================================

def prepare_higgs_full():
    """
    Pre-processa HIGGS.csv.gz do UCI ML Repository.

    Formato: CSV sem header, coluna 0 = label (0/1), colunas 1-28 = features.
    Arquivo: ~2.6 GB comprimido, ~8 GB descomprimido.

    Gera:
        data/higgs_full/higgs_full_X.npy  (~1.2 GB em float32)
        data/higgs_full/higgs_full_y.npy  (~11 MB em int8)
    """
    import pandas as pd

    d = data_dir("higgs_full")
    raw_file = os.path.join(d, "HIGGS.csv.gz")

    if not os.path.exists(raw_file):
        # Tenta sem compressao
        raw_file = os.path.join(d, "HIGGS.csv")
        if not os.path.exists(raw_file):
            print(f"ERRO: Arquivo nao encontrado.")
            print(f"  Esperado: {os.path.join(d, 'HIGGS.csv.gz')}")
            print(f"  Ou:       {os.path.join(d, 'HIGGS.csv')}")
            print(f"\n  Baixe de: {DATASET_INFO['higgs_full']['url']}")
            return False

    print(f"[HIGGS Full] Lendo {raw_file}...")
    print(f"[HIGGS Full] Isso pode levar varios minutos (11M linhas)...")
    t0 = time.time()

    # Leitura em chunks para nao estourar memoria
    chunks = []
    chunk_size = 500_000
    for i, chunk in enumerate(pd.read_csv(raw_file, header=None, chunksize=chunk_size)):
        chunks.append(chunk.values)
        loaded = (i + 1) * chunk_size
        print(f"  Lido: {loaded:,} linhas...", end="\r")

    data = np.vstack(chunks)
    print(f"\n[HIGGS Full] {data.shape[0]:,} amostras carregadas em {time.time()-t0:.0f}s")

    # Coluna 0 = label, colunas 1-28 = features
    y = data[:, 0].astype(np.int8)
    X = data[:, 1:].astype(np.float32)

    x_path, y_path = npy_paths("higgs_full")
    os.makedirs(os.path.dirname(x_path), exist_ok=True)
    np.save(x_path, X)
    np.save(y_path, y)

    print(f"[HIGGS Full] Salvo: {x_path} — {X.shape} ({X.nbytes/1024/1024:.0f} MB)")
    print(f"[HIGGS Full] Salvo: {y_path} — {y.shape} ({y.nbytes/1024:.0f} KB)")
    print(f"[HIGGS Full] Distribuicao: classe 0 = {(y==0).sum():,} | classe 1 = {(y==1).sum():,}")
    return True


# ======================================================================
# Epsilon (400k treino + 100k teste, 2000 features)
# ======================================================================

def prepare_epsilon():
    """
    Pre-processa epsilon_normalized do LIBSVM.

    Formato: LIBSVM sparse (label feature:value feature:value ...)
    Arquivos:
        epsilon_normalized.bz2     (~treino, 400k amostras)
        epsilon_normalized.t.bz2   (~teste, 100k amostras)

    NOTA: este dataset tem 2000 features — modelos gradient boosting
    podem ser mais lentos. Considere usar um subconjunto se necessario.

    Gera:
        data/epsilon/epsilon_X.npy  (~3.8 GB em float32 se full)
        data/epsilon/epsilon_y.npy
    """
    from sklearn.datasets import load_svmlight_file

    d = data_dir("epsilon")

    # Procura arquivos com e sem extensao .bz2
    train_candidates = [
        os.path.join(d, "epsilon_normalized.bz2"),
        os.path.join(d, "epsilon_normalized"),
    ]
    test_candidates = [
        os.path.join(d, "epsilon_normalized.t.bz2"),
        os.path.join(d, "epsilon_normalized.t"),
    ]

    train_file = None
    for f in train_candidates:
        if os.path.exists(f):
            train_file = f
            break

    test_file = None
    for f in test_candidates:
        if os.path.exists(f):
            test_file = f
            break

    if train_file is None:
        print(f"ERRO: Arquivo de treino nao encontrado.")
        print(f"  Esperado: {train_candidates[0]}")
        print(f"\n  Baixe de: {DATASET_INFO['epsilon']['url_train']}")
        return False

    print(f"[Epsilon] Lendo treino: {train_file}...")
    print(f"[Epsilon] Isso pode levar varios minutos (2000 features)...")
    t0 = time.time()

    X_train, y_train = load_svmlight_file(train_file, n_features=2000)
    X_train = X_train.toarray().astype(np.float32)
    y_train = ((y_train + 1) / 2).astype(np.int8)  # converte -1/+1 para 0/1
    print(f"  Treino: {X_train.shape} em {time.time()-t0:.0f}s")

    if test_file:
        print(f"[Epsilon] Lendo teste: {test_file}...")
        X_test, y_test = load_svmlight_file(test_file, n_features=2000)
        X_test = X_test.toarray().astype(np.float32)
        y_test = ((y_test + 1) / 2).astype(np.int8)
        print(f"  Teste: {X_test.shape}")

        # Concatena treino + teste para particionar depois pelo FL
        X = np.vstack([X_train, X_test])
        y = np.concatenate([y_train, y_test])
    else:
        print(f"[Epsilon] AVISO: arquivo de teste nao encontrado, usando apenas treino.")
        X = X_train
        y = y_train

    x_path, y_path = npy_paths("epsilon")
    os.makedirs(os.path.dirname(x_path), exist_ok=True)
    np.save(x_path, X)
    np.save(y_path, y)

    print(f"[Epsilon] Salvo: {x_path} — {X.shape} ({X.nbytes/1024/1024:.0f} MB)")
    print(f"[Epsilon] Salvo: {y_path} — {y.shape} ({y.nbytes/1024:.0f} KB)")
    print(f"[Epsilon] Distribuicao: classe 0 = {(y==0).sum():,} | classe 1 = {(y==1).sum():,}")
    return True


# ======================================================================
# Avazu (40M amostras, features categoricas)
# ======================================================================

def prepare_avazu():
    """
    Pre-processa Avazu CTR dataset do Kaggle.

    Formato: CSV com header (id, click, hour, C1, banner_pos, ..., C21).
    Arquivo: train.csv (~6 GB) ou train.gz (~1.2 GB comprimido).

    Processamento:
    1. Remove colunas id e hour (ou extrai hora do dia de hour)
    2. Feature hashing das colunas categoricas para dimensao fixa
    3. Label = coluna "click" (0 ou 1)

    Gera:
        data/avazu/avazu_X.npy
        data/avazu/avazu_y.npy
    """
    import pandas as pd
    from sklearn.feature_extraction import FeatureHasher

    d = data_dir("avazu")

    raw_candidates = [
        os.path.join(d, "train.gz"),
        os.path.join(d, "train.csv"),
        os.path.join(d, "train.csv.gz"),
    ]

    raw_file = None
    for f in raw_candidates:
        if os.path.exists(f):
            raw_file = f
            break

    if raw_file is None:
        print(f"ERRO: Arquivo nao encontrado.")
        print(f"  Esperado: {raw_candidates[0]} ou {raw_candidates[1]}")
        print(f"\n  Baixe de: {DATASET_INFO['avazu']['url']}")
        return False

    print(f"[Avazu] Lendo {raw_file}...")
    print(f"[Avazu] Isso pode levar varios minutos (40M linhas)...")
    t0 = time.time()

    # Leitura em chunks
    n_hash_features = 256  # dimensao do feature hashing
    hasher = FeatureHasher(n_features=n_hash_features, input_type="string")

    # Colunas categoricas do Avazu
    cat_cols = ["C1", "banner_pos", "site_id", "site_domain", "site_category",
                "app_id", "app_domain", "app_category", "device_id", "device_ip",
                "device_model", "device_type", "device_conn_type",
                "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21"]

    X_chunks = []
    y_chunks = []
    chunk_size = 1_000_000
    total_rows = 0

    for chunk in pd.read_csv(raw_file, chunksize=chunk_size):
        y_chunk = chunk["click"].values.astype(np.int8)

        # Extrai hora do dia da coluna hour (formato YYMMDDHH)
        if "hour" in chunk.columns:
            chunk["hour_of_day"] = chunk["hour"].astype(str).str[-2:]

        # Feature hashing das colunas categoricas
        available_cats = [c for c in cat_cols if c in chunk.columns]
        if "hour_of_day" in chunk.columns:
            available_cats.append("hour_of_day")

        # Converte para formato "coluna=valor" para o hasher (vetorizado)
        cat_data = chunk[available_cats].astype(str)
        records = []
        for col in available_cats:
            cat_data[col] = col + "=" + cat_data[col]
        records = cat_data.values.tolist()

        X_chunk = hasher.transform(records).toarray().astype(np.float32)

        X_chunks.append(X_chunk)
        y_chunks.append(y_chunk)
        total_rows += len(chunk)
        print(f"  Processado: {total_rows:,} linhas...", end="\r")

    X = np.vstack(X_chunks)
    y = np.concatenate(y_chunks)
    print(f"\n[Avazu] {X.shape[0]:,} amostras processadas em {time.time()-t0:.0f}s")

    x_path, y_path = npy_paths("avazu")
    os.makedirs(os.path.dirname(x_path), exist_ok=True)
    np.save(x_path, X)
    np.save(y_path, y)

    print(f"[Avazu] Salvo: {x_path} — {X.shape} ({X.nbytes/1024/1024:.0f} MB)")
    print(f"[Avazu] Salvo: {y_path} — {y.shape} ({y.nbytes/1024:.0f} KB)")
    print(f"[Avazu] Distribuicao: click=0 = {(y==0).sum():,} | click=1 = {(y==1).sum():,}")
    return True


# ======================================================================
# Main
# ======================================================================

PREPARERS = {
    "higgs_full": prepare_higgs_full,
    "epsilon": prepare_epsilon,
    "avazu": prepare_avazu,
}


def main():
    parser = argparse.ArgumentParser(
        description="Pre-processa datasets brutos para .npy",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, choices=list(PREPARERS.keys()),
                        help="Dataset a preparar")
    parser.add_argument("--all", action="store_true",
                        help="Prepara todos os datasets com arquivos brutos disponiveis")
    parser.add_argument("--force", action="store_true",
                        help="Reprocessa mesmo se .npy ja existem")
    args = parser.parse_args()

    if not args.dataset and not args.all:
        parser.print_help()
        print("\n\nDatasets disponiveis:")
        for name, info in DATASET_INFO.items():
            status = "PRONTO" if is_prepared(name) else "nao preparado"
            print(f"  {name:15s} — {info['description']} [{status}]")
        return

    targets = list(PREPARERS.keys()) if args.all else [args.dataset]

    ok = 0
    fail = 0
    skip = 0

    for name in targets:
        print(f"\n{'='*60}")
        print(f"  Preparando: {name}")
        print(f"{'='*60}")

        if is_prepared(name) and not args.force:
            print(f"[{name}] Ja preparado (use --force para reprocessar)")
            skip += 1
            continue

        preparer = PREPARERS.get(name)
        if preparer is None:
            print(f"[{name}] Sem preparador (dataset pre-pronto)")
            skip += 1
            continue

        try:
            success = preparer()
            if success:
                ok += 1
            else:
                fail += 1
        except Exception as e:
            print(f"[{name}] ERRO: {e}")
            fail += 1

    print(f"\n{'='*60}")
    print(f"  Resultado: {ok} preparados | {skip} ja prontos | {fail} falhas")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
