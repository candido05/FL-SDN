"""
Pre-processamento dos datasets para Federated Learning.

Converte os arquivos brutos em .npy prontos para uso pelo FL-SDN,
aplicando feature engineering e tratamento de dados especificos
para cada dataset.

Pipeline por dataset:
  HIGGS Full:   limpeza de outliers + interacoes fisicas + remocao de correlacoes
  Epsilon:      selecao de features por variancia + remocao de correlacoes
  Avazu:        feature engineering temporal + Feature Hashing (data-independent)

Uso:
    python tools/prepare_datasets.py --dataset higgs_full
    python tools/prepare_datasets.py --dataset epsilon
    python tools/prepare_datasets.py --dataset avazu
    python tools/prepare_datasets.py --all

Requisitos:
    pip install pandas scikit-learn
"""

import argparse
import os
import sys
import time

# Adiciona diretorio pai ao path para encontrar config, datasets, etc.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from datasets.paths import data_dir, npy_paths, is_prepared, DATASET_INFO


# ======================================================================
# Utilidades de preprocessing compartilhadas
# ======================================================================

def remove_constant_features(X, threshold=0.0, label=""):
    """Remove features com variancia zero ou abaixo do threshold."""
    variances = np.var(X, axis=0)
    mask = variances > threshold
    n_removed = X.shape[1] - mask.sum()
    if n_removed > 0:
        print(f"  [{label}] Removidas {n_removed} features constantes "
              f"(variancia <= {threshold})")
    return X[:, mask], mask


def remove_highly_correlated(X, threshold=0.95, label=""):
    """
    Remove uma de cada par de features com correlacao >= threshold.
    Usa amostragem para datasets grandes (evita OOM).
    """
    n_samples = min(50000, len(X))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X), size=n_samples, replace=False)
    X_sample = X[idx]

    corr = np.corrcoef(X_sample, rowvar=False)
    n_features = corr.shape[0]
    to_remove = set()

    for i in range(n_features):
        if i in to_remove:
            continue
        for j in range(i + 1, n_features):
            if j in to_remove:
                continue
            if abs(corr[i, j]) >= threshold:
                to_remove.add(j)

    if to_remove:
        mask = np.ones(n_features, dtype=bool)
        mask[list(to_remove)] = False
        print(f"  [{label}] Removidas {len(to_remove)} features com "
              f"correlacao >= {threshold} ({n_features} → {mask.sum()})")
        return X[:, mask], mask
    return X, np.ones(n_features, dtype=bool)


def clip_outliers(X, n_sigma=5.0, label=""):
    """Clippa valores extremos a n_sigma desvios padrao da media."""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1.0

    lower = mean - n_sigma * std
    upper = mean + n_sigma * std

    n_clipped = np.sum((X < lower) | (X > upper))
    total = X.size
    pct = n_clipped / total * 100

    X_clipped = np.clip(X, lower, upper)
    if n_clipped > 0:
        print(f"  [{label}] Clipping de outliers: {n_clipped:,} valores "
              f"({pct:.3f}%) clippados a {n_sigma}σ")
    return X_clipped


def log_preprocessing_summary(X_original_shape, X_final_shape, label=""):
    """Imprime resumo do preprocessing."""
    n_orig, f_orig = X_original_shape
    n_final, f_final = X_final_shape
    print(f"  [{label}] Preprocessing: {f_orig} → {f_final} features "
          f"({n_final:,} amostras)")


# ======================================================================
# HIGGS Full (11M amostras, 28 features)
# ======================================================================

def prepare_higgs_full():
    """
    Pre-processa HIGGS.csv.gz do UCI ML Repository.

    Features do HIGGS:
      - Colunas 1-21: features de baixo nivel (momentos cineticos das particulas)
      - Colunas 22-28: features de alto nivel (funcoes das de baixo nivel)

    Preprocessing:
      1. Conversao para float32
      2. Clipping de outliers a 5σ
      3. Interacoes entre features de alto nivel (produto)
      4. Remocao de features com correlacao >= 0.95
    """
    import pandas as pd

    d = data_dir("higgs_full")
    raw_file = os.path.join(d, "HIGGS.csv.gz")

    if not os.path.exists(raw_file):
        raw_file = os.path.join(d, "HIGGS.csv")
        if not os.path.exists(raw_file):
            print(f"ERRO: Arquivo nao encontrado.")
            print(f"  Esperado: {os.path.join(d, 'HIGGS.csv.gz')}")
            print(f"\n  Baixe de: {DATASET_INFO['higgs_full']['url']}")
            return False

    print(f"[HIGGS Full] Lendo {raw_file}...")
    print(f"[HIGGS Full] Isso pode levar varios minutos (11M linhas)...")
    t0 = time.time()

    chunks = []
    chunk_size = 500_000
    for i, chunk in enumerate(pd.read_csv(raw_file, header=None, chunksize=chunk_size)):
        chunks.append(chunk.values)
        loaded = (i + 1) * chunk_size
        print(f"  Lido: {loaded:,} linhas...", end="\r")

    data = np.vstack(chunks)
    print(f"\n[HIGGS Full] {data.shape[0]:,} amostras carregadas em {time.time()-t0:.0f}s")

    y = data[:, 0].astype(np.int8)
    X = data[:, 1:].astype(np.float32)
    original_shape = X.shape

    print(f"\n[HIGGS Full] Iniciando preprocessing...")

    # 1. Clipping de outliers
    X = clip_outliers(X, n_sigma=5.0, label="HIGGS Full")

    # 2. Interacoes entre features de alto nivel (indices 21-27)
    high_level = X[:, 21:28]
    n_high = high_level.shape[1]

    interaction_features = []
    for i in range(n_high):
        for j in range(i + 1, n_high):
            product = high_level[:, i] * high_level[:, j]
            interaction_features.append(product.reshape(-1, 1))

    if interaction_features:
        X_interactions = np.hstack(interaction_features).astype(np.float32)
        X = np.hstack([X, X_interactions])
        print(f"  [HIGGS Full] Adicionadas {len(interaction_features)} "
              f"features de interacao (total: {X.shape[1]})")

    # 3. Remocao de correlacoes altas
    X, _ = remove_highly_correlated(X, threshold=0.95, label="HIGGS Full")

    log_preprocessing_summary(original_shape, X.shape, "HIGGS Full")

    x_path, y_path = npy_paths("higgs_full")
    os.makedirs(os.path.dirname(x_path), exist_ok=True)
    np.save(x_path, X)
    np.save(y_path, y)

    print(f"[HIGGS Full] Salvo: {x_path} — {X.shape} ({X.nbytes/1024/1024:.0f} MB)")
    print(f"[HIGGS Full] Salvo: {y_path} — {y.shape} ({y.nbytes/1024:.0f} KB)")
    print(f"[HIGGS Full] Distribuicao: classe 0 = {(y==0).sum():,} | "
          f"classe 1 = {(y==1).sum():,}")
    elapsed = time.time() - t0
    print(f"[HIGGS Full] Tempo total: {elapsed:.0f}s")
    return True


# ======================================================================
# Epsilon (400k treino + 100k teste, 2000 features)
# ======================================================================

def prepare_epsilon():
    """
    Pre-processa epsilon_normalized do LIBSVM.

    Preprocessing:
      1. Conversao de labels -1/+1 → 0/1
      2. Remocao de features constantes
      3. Selecao por variancia (top 500)
      4. Remocao de correlacoes >= 0.95
    """
    from sklearn.datasets import load_svmlight_file

    d = data_dir("epsilon")

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
    y_train = ((y_train + 1) / 2).astype(np.int8)
    print(f"  Treino: {X_train.shape} em {time.time()-t0:.0f}s")

    if test_file:
        print(f"[Epsilon] Lendo teste: {test_file}...")
        X_test, y_test = load_svmlight_file(test_file, n_features=2000)
        X_test = X_test.toarray().astype(np.float32)
        y_test = ((y_test + 1) / 2).astype(np.int8)
        print(f"  Teste: {X_test.shape}")

        X = np.vstack([X_train, X_test])
        y = np.concatenate([y_train, y_test])
    else:
        print(f"[Epsilon] AVISO: arquivo de teste nao encontrado, usando apenas treino.")
        X = X_train
        y = y_train

    original_shape = X.shape
    print(f"\n[Epsilon] Iniciando preprocessing...")

    # 1. Remover features constantes
    X, const_mask = remove_constant_features(X, threshold=1e-7, label="Epsilon")

    # 2. Selecao por variancia: top 500
    MAX_FEATURES = 500
    if X.shape[1] > MAX_FEATURES:
        variances = np.var(X, axis=0)
        top_indices = np.argsort(variances)[::-1][:MAX_FEATURES]
        top_indices = np.sort(top_indices)
        X = X[:, top_indices]
        print(f"  [Epsilon] Selecao por variancia: mantidas top {MAX_FEATURES} "
              f"features (de {const_mask.sum()})")

    # 3. Remocao de correlacoes altas
    X, _ = remove_highly_correlated(X, threshold=0.95, label="Epsilon")

    log_preprocessing_summary(original_shape, X.shape, "Epsilon")

    x_path, y_path = npy_paths("epsilon")
    os.makedirs(os.path.dirname(x_path), exist_ok=True)
    np.save(x_path, X)
    np.save(y_path, y)

    print(f"[Epsilon] Salvo: {x_path} — {X.shape} ({X.nbytes/1024/1024:.0f} MB)")
    print(f"[Epsilon] Salvo: {y_path} — {y.shape} ({y.nbytes/1024:.0f} KB)")
    print(f"[Epsilon] Distribuicao: classe 0 = {(y==0).sum():,} | "
          f"classe 1 = {(y==1).sum():,}")
    elapsed = time.time() - t0
    print(f"[Epsilon] Tempo total: {elapsed:.0f}s")
    return True


# ======================================================================
# Avazu (40M amostras, features categoricas)
# ======================================================================

def prepare_avazu():
    """
    Pre-processa Avazu CTR dataset do Kaggle.

    Preprocessing (tudo data-independent, compativel com FL):
      1. Feature engineering temporal (derivado da propria linha):
         hora do dia, dia da semana, periodo, codificacao ciclica sin/cos
      2. Feature Hashing de TODAS as colunas categoricas (1024 dimensoes)
         — data-independent: o mapeamento hash nao depende de dados de
         outros nos, essencial em Federated Learning.

    NOTA: Frequency Encoding NAO e usado pois exige passada global
    sobre todos os dados, violando a premissa de FL.
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

    n_hash_features = 1024
    hasher = FeatureHasher(n_features=n_hash_features, input_type="string")

    cat_cols = [
        "C1", "banner_pos",
        "site_id", "site_domain", "site_category",
        "app_id", "app_domain", "app_category",
        "device_id", "device_ip", "device_model",
        "device_type", "device_conn_type",
        "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21",
    ]

    X_chunks = []
    y_chunks = []
    total_rows = 0
    chunk_size = 1_000_000

    for chunk in pd.read_csv(raw_file, chunksize=chunk_size):
        y_chunk = chunk["click"].values.astype(np.int8)

        # Features temporais (data-independent)
        if "hour" in chunk.columns:
            hour_str = chunk["hour"].astype(str)
            hour_of_day = hour_str.str[-2:].astype(int).values
            day_hash = hour_str.str[:-2].astype(int).values
            day_of_week = (day_hash % 7).astype(np.float32)
            period = (hour_of_day // 6).astype(np.float32)
            hour_sin = np.sin(2 * np.pi * hour_of_day / 24).astype(np.float32)
            hour_cos = np.cos(2 * np.pi * hour_of_day / 24).astype(np.float32)

            temporal_features = np.column_stack([
                hour_of_day.astype(np.float32),
                day_of_week, period, hour_sin, hour_cos,
            ])
        else:
            temporal_features = np.zeros((len(chunk), 5), dtype=np.float32)

        # Feature Hashing de TODAS as categoricas
        available_cats = [c for c in cat_cols if c in chunk.columns]
        cat_data = chunk[available_cats].astype(str).copy()
        for col in available_cats:
            cat_data[col] = col + "=" + cat_data[col]
        records = cat_data.values.tolist()
        X_hashed = hasher.transform(records).toarray().astype(np.float32)

        X_chunk = np.hstack([temporal_features, X_hashed])

        X_chunks.append(X_chunk)
        y_chunks.append(y_chunk)
        total_rows += len(chunk)
        print(f"  Processado: {total_rows:,} linhas...", end="\r")

    X = np.vstack(X_chunks)
    y = np.concatenate(y_chunks)

    n_temporal = 5
    print(f"\n[Avazu] {X.shape[0]:,} amostras processadas em {time.time()-t0:.0f}s")
    print(f"  Features: {n_temporal} temporais + "
          f"{n_hash_features} hashed = {X.shape[1]} total")
    print(f"  Todas as features sao data-independent (compativel com FL)")
    print(f"  Desbalanceamento: click=0 = {(y==0).sum():,} ({(y==0).mean()*100:.1f}%) | "
          f"click=1 = {(y==1).sum():,} ({(y==1).mean()*100:.1f}%)")

    x_path, y_path = npy_paths("avazu")
    os.makedirs(os.path.dirname(x_path), exist_ok=True)
    np.save(x_path, X)
    np.save(y_path, y)

    print(f"[Avazu] Salvo: {x_path} — {X.shape} ({X.nbytes/1024/1024:.0f} MB)")
    print(f"[Avazu] Salvo: {y_path} — {y.shape} ({y.nbytes/1024:.0f} KB)")
    elapsed = time.time() - t0
    print(f"[Avazu] Tempo total: {elapsed:.0f}s")
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
            import traceback
            traceback.print_exc()
            fail += 1

    print(f"\n{'='*60}")
    print(f"  Resultado: {ok} preparados | {skip} ja prontos | {fail} falhas")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
