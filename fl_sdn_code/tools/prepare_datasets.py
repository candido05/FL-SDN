"""
Pre-processamento dos datasets para Federated Learning.

Converte os arquivos brutos em .npy prontos para uso pelo FL-SDN,
aplicando feature engineering e tratamento de dados especificos
para cada dataset.

Pipeline por dataset:
  HIGGS Full:   limpeza de outliers + interacoes fisicas + remocao de correlacoes
  Epsilon:      selecao de features por variancia + remocao de correlacoes
  MNIST:        normalizacao + remocao de pixels constantes/baixa variancia
  Avazu:        feature engineering temporal + Feature Hashing (data-independent)

Uso:
    python tools/prepare_datasets.py --dataset higgs_full
    python tools/prepare_datasets.py --dataset epsilon
    python tools/prepare_datasets.py --dataset mnist
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
    print(f"  [{label}] Preprocessing: {f_orig} -> {f_final} features "
          f"({n_final:,} amostras)")


# Constantes configuráveis (e patcháveis em testes unitários)
AVAZU_MAX_ROWS = 2_000_000   # máx. amostras Avazu; aumente se tiver mais RAM/disco
HIGGS_CHUNK_SIZE = 500_000   # linhas por chunk na leitura do HIGGS


# ======================================================================
# HIGGS Full (11M amostras, 28 features)
# ======================================================================

def prepare_higgs_full(_chunk_size=None):
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
    chunk_size = _chunk_size if _chunk_size is not None else HIGGS_CHUNK_SIZE
    try:
        for i, chunk in enumerate(pd.read_csv(raw_file, header=None, chunksize=chunk_size)):
            chunks.append(chunk.values)
            loaded = (i + 1) * chunk_size
            print(f"  Lido: ~{min(loaded, 11_000_000):,} linhas...", end="\r")
    except EOFError:
        print(f"\n[HIGGS Full] AVISO: arquivo truncado/incompleto.")
        print(f"  Usando {len(chunks)} chunks completos lidos ate o erro.")
        print(f"  Para o dataset completo, rebaixe HIGGS.csv.gz do UCI (2.62 GB).")

    if not chunks:
        print("[HIGGS Full] ERRO: nenhum chunk completo foi lido. "
              "Arquivo pode estar corrompido ou muito severamente truncado.")
        return False

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
# Credit Card Fraud Detection (284k amostras, 30 features)
# ======================================================================

def prepare_creditcard():
    """
    Pre-processa Credit Card Fraud Detection dataset (ULB).

    Fonte: OpenML (creditcard, dataset_id=1597) ou Kaggle.

    Preprocessing (tudo data-independent, compativel com FL):
      1. Remocao da coluna Time (indice sequencial, sem sentido em FL)
      2. Criacao de feature Hour: ciclo de 24h derivado de Time (sin/cos)
         — data-independent, cada transacao usa apenas seu proprio Time
      3. Normalizacao de Amount via log1p (data-independent)
      4. Features V1-V28 mantidas como estao (ja PCA-normalizadas pela ULB)
      5. Conversao para float32
    """
    d = data_dir("creditcard")
    os.makedirs(d, exist_ok=True)
    x_path, y_path = npy_paths("creditcard")

    # Tenta carregar CSV local primeiro
    csv_candidates = [
        os.path.join(d, "creditcard.csv"),
        os.path.join(d, "creditcard.csv.gz"),
    ]
    csv_file = None
    for f in csv_candidates:
        if os.path.exists(f):
            csv_file = f
            break

    t0 = time.time()

    if csv_file:
        import pandas as pd
        print(f"[CreditCard] Lendo {csv_file}...")
        df = pd.read_csv(csv_file)
        X_raw = df.drop(columns=["Class"]).values
        y = df["Class"].values
        col_names = [c for c in df.columns if c != "Class"]
    else:
        # Fallback: OpenML
        from sklearn.datasets import fetch_openml
        print("[CreditCard] Baixando dataset via OpenML (creditcard)...")
        print("[CreditCard] Isso pode levar alguns minutos na primeira vez...")
        data = fetch_openml(data_id=1597, as_frame=True, parser="auto")
        df = data.frame
        y = df["Class"].astype(int).values
        X_raw = df.drop(columns=["Class"]).values.astype(np.float64)
        col_names = [c for c in df.columns if c != "Class"]

    print(f"[CreditCard] {X_raw.shape[0]:,} transacoes carregadas em "
          f"{time.time()-t0:.0f}s")
    print(f"  Shape original: {X_raw.shape}")

    # Identificar colunas por nome
    time_idx = col_names.index("Time") if "Time" in col_names else None
    amount_idx = col_names.index("Amount") if "Amount" in col_names else None

    features = []
    feat_names = []

    # 1. Features V1-V28 (ja PCA-normalizadas, manter como estao)
    for i, name in enumerate(col_names):
        if name.startswith("V"):
            features.append(X_raw[:, i].reshape(-1, 1))
            feat_names.append(name)

    # 2. Hour features derivadas de Time (ciclo 24h, data-independent)
    if time_idx is not None:
        time_col = X_raw[:, time_idx]
        # Time esta em segundos desde a primeira transacao (~48h de dados)
        hour_of_day = (time_col % 86400) / 3600  # 0-24
        hour_sin = np.sin(2 * np.pi * hour_of_day / 24).astype(np.float32)
        hour_cos = np.cos(2 * np.pi * hour_of_day / 24).astype(np.float32)
        features.append(hour_sin.reshape(-1, 1))
        features.append(hour_cos.reshape(-1, 1))
        feat_names.extend(["hour_sin", "hour_cos"])
        print(f"  [CreditCard] Time -> hour_sin/hour_cos (ciclo 24h)")

    # 3. Amount normalizado via log1p (data-independent)
    if amount_idx is not None:
        amount = X_raw[:, amount_idx]
        amount_log = np.log1p(amount).astype(np.float32)
        features.append(amount_log.reshape(-1, 1))
        feat_names.append("amount_log1p")
        print(f"  [CreditCard] Amount -> log1p(Amount)")

    X = np.hstack(features).astype(np.float32)
    y = y.astype(np.int8)

    n_v = sum(1 for n in feat_names if n.startswith("V"))
    n_eng = len(feat_names) - n_v
    print(f"  [CreditCard] Features: {n_v} PCA (V1-V28) + {n_eng} engenheiradas "
          f"= {X.shape[1]} total")

    log_preprocessing_summary((X_raw.shape[0], len(col_names)), X.shape, "CreditCard")

    np.save(x_path, X)
    np.save(y_path, y)

    n_fraud = int(y.sum())
    n_total = len(y)
    print(f"[CreditCard] Salvo: {x_path} -- {X.shape} "
          f"({X.nbytes/1024/1024:.1f} MB)")
    print(f"[CreditCard] Salvo: {y_path} -- {y.shape} ({y.nbytes/1024:.0f} KB)")
    print(f"[CreditCard] Distribuicao: legitimas = {n_total - n_fraud:,} "
          f"({(n_total - n_fraud)/n_total*100:.2f}%) | "
          f"fraudes = {n_fraud:,} ({n_fraud/n_total*100:.3f}%)")
    elapsed = time.time() - t0
    print(f"[CreditCard] Tempo total: {elapsed:.0f}s")
    return True


# ======================================================================
# MNIST (70k amostras, 784 features -> ~443 apos preprocessing)
# ======================================================================

MNIST_VARIANCE_THRESHOLD = 0.01  # remove pixels quase constantes


def prepare_mnist():
    """
    Pre-processa MNIST via sklearn.datasets.fetch_openml.

    Classificacao binaria: digitos 0-4 (classe 0) vs 5-9 (classe 1).

    Preprocessing (tudo data-independent, compativel com FL):
      1. Normalizacao [0, 1] (divisao por 255)
      2. Conversao para float32
      3. Remocao de pixels constantes (bordas sempre pretas)
      4. Remocao de features com variancia < MNIST_VARIANCE_THRESHOLD
      5. Binarizacao das labels: 0-4 → 0, 5-9 → 1
    """
    from sklearn.datasets import fetch_openml

    d = data_dir("mnist")
    os.makedirs(d, exist_ok=True)

    x_path, y_path = npy_paths("mnist")

    print("[MNIST] Baixando dataset via OpenML (mnist_784)...")
    print("[MNIST] Isso pode levar alguns minutos na primeira vez...")
    t0 = time.time()

    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X_raw = mnist.data.astype(np.float32)
    y_raw = mnist.target.astype(int)

    print(f"[MNIST] {X_raw.shape[0]:,} amostras carregadas em {time.time()-t0:.0f}s")
    print(f"  Shape original: {X_raw.shape}")
    original_shape = X_raw.shape

    # 1. Normalizacao [0, 1]
    X = X_raw / 255.0
    print(f"  [MNIST] Normalizado para [0, 1] (divisao por 255)")

    # 2. Remocao de features constantes (pixels de borda sempre pretos)
    X, const_mask = remove_constant_features(X, threshold=0.0, label="MNIST")

    # 3. Remocao de features com variancia muito baixa
    # Pixels quase constantes nao ajudam gradient boosting (splits inuteis)
    variances = np.var(X, axis=0)
    var_mask = variances >= MNIST_VARIANCE_THRESHOLD
    n_removed_var = X.shape[1] - var_mask.sum()
    if n_removed_var > 0:
        X = X[:, var_mask]
        print(f"  [MNIST] Removidas {n_removed_var} features com variancia "
              f"< {MNIST_VARIANCE_THRESHOLD} ({X.shape[1]} restantes)")

    # 4. Binarizacao das labels: 0-4 → classe 0, 5-9 → classe 1
    y = (y_raw >= 5).astype(np.int8)

    log_preprocessing_summary(original_shape, X.shape, "MNIST")

    np.save(x_path, X)
    np.save(y_path, y)

    print(f"[MNIST] Salvo: {x_path} — {X.shape} ({X.nbytes/1024/1024:.1f} MB)")
    print(f"[MNIST] Salvo: {y_path} — {y.shape} ({y.nbytes/1024:.0f} KB)")
    print(f"[MNIST] Distribuicao: classe 0 (digitos 0-4) = {(y==0).sum():,} "
          f"({(y==0).mean()*100:.1f}%) | "
          f"classe 1 (digitos 5-9) = {(y==1).sum():,} "
          f"({(y==1).mean()*100:.1f}%)")
    elapsed = time.time() - t0
    print(f"[MNIST] Tempo total: {elapsed:.0f}s")
    return True


# ======================================================================
# Avazu (40M amostras, features categoricas)
# ======================================================================

def prepare_avazu(_max_rows=None):
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

    NOTA DE MEMORIA: o dataset completo (40M amostras x 1029 features x float32)
    ocuparia ~164 GB em RAM. Por isso, usa escrita incremental via memmap e
    limita a AVAZU_MAX_ROWS amostras. Aumente a constante se tiver mais RAM/disco.

    Args:
        _max_rows: sobreescreve AVAZU_MAX_ROWS (uso interno/testes).
    """
    import pandas as pd
    from sklearn.feature_extraction import FeatureHasher

    MAX_ROWS = _max_rows if _max_rows is not None else AVAZU_MAX_ROWS

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

    n_hash_features = 1024
    n_temporal = 5
    n_features = n_temporal + n_hash_features  # 1029

    print(f"[Avazu] Lendo {raw_file} (limitado a {MAX_ROWS:,} amostras)...")
    print(f"[Avazu] Escrita incremental via memmap — sem vstack em RAM.")
    t0 = time.time()

    hasher = FeatureHasher(n_features=n_hash_features, input_type="string")

    cat_cols = [
        "C1", "banner_pos",
        "site_id", "site_domain", "site_category",
        "app_id", "app_domain", "app_category",
        "device_id", "device_ip", "device_model",
        "device_type", "device_conn_type",
        "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21",
    ]

    x_path, y_path = npy_paths("avazu")
    os.makedirs(os.path.dirname(x_path), exist_ok=True)

    # Pre-aloca arquivos .npy via memmap — escrita direta sem acumular na RAM
    X_mm = np.lib.format.open_memmap(
        x_path, mode="w+", dtype=np.float32, shape=(MAX_ROWS, n_features),
    )
    y_mm = np.lib.format.open_memmap(
        y_path, mode="w+", dtype=np.int8, shape=(MAX_ROWS,),
    )

    row_offset = 0
    chunk_size = 500_000
    y_sum = 0

    for chunk in pd.read_csv(raw_file, chunksize=chunk_size):
        if row_offset >= MAX_ROWS:
            break

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

        n_take = min(len(X_chunk), MAX_ROWS - row_offset)
        X_mm[row_offset:row_offset + n_take] = X_chunk[:n_take]
        y_mm[row_offset:row_offset + n_take] = y_chunk[:n_take]
        y_sum += int(y_chunk[:n_take].sum())
        row_offset += n_take
        print(f"  Processado: {row_offset:,}/{MAX_ROWS:,} linhas...", end="\r")

    del X_mm, y_mm  # flush para disco

    total_rows = row_offset

    # Se o CSV tinha menos linhas que MAX_ROWS, o memmap foi pre-alocado
    # maior que o necessario (com zeros no final). Trunca para o tamanho real.
    if total_rows < MAX_ROWS:
        X_partial = np.load(x_path, mmap_mode="r")[:total_rows].copy()
        y_partial = np.load(y_path, mmap_mode="r")[:total_rows].copy()
        np.save(x_path, X_partial)
        np.save(y_path, y_partial)

    print(f"\n[Avazu] {total_rows:,} amostras salvas em {time.time()-t0:.0f}s")
    print(f"  Features: {n_temporal} temporais + "
          f"{n_hash_features} hashed = {n_features} total")
    print(f"  Todas as features sao data-independent (compativel com FL)")
    size_mb = total_rows * n_features * 4 / 1024 / 1024
    print(f"  Desbalanceamento: click=0 = {total_rows - y_sum:,} | "
          f"click=1 = {y_sum:,} ({y_sum/total_rows*100:.1f}%)")
    print(f"[Avazu] Salvo: {x_path} — ({total_rows}, {n_features}) ({size_mb:.0f} MB)")
    elapsed = time.time() - t0
    print(f"[Avazu] Tempo total: {elapsed:.0f}s")
    return True


# ======================================================================
# Main
# ======================================================================

PREPARERS = {
    "higgs_full": prepare_higgs_full,
    "epsilon": prepare_epsilon,
    "creditcard": prepare_creditcard,
    "mnist": prepare_mnist,
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
