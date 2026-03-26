"""
Gerenciamento centralizado de paths dos datasets.

Todos os datasets ficam em fl_sdn_code/data/<nome_dataset>/.
Joao Victor deve colocar os arquivos nas pastas corretas antes de executar.

Estrutura esperada:
    data/
    ├── higgs/
    │   ├── higgs_X.npy          (N_SAMPLES amostras, 28 features)
    │   └── higgs_y.npy          (N_SAMPLES labels)
    ├── higgs_full/
    │   ├── HIGGS.csv.gz         (11M amostras, baixar de UCI)
    │   ├── higgs_full_X.npy     (gerado por prepare_datasets.py)
    │   └── higgs_full_y.npy     (gerado por prepare_datasets.py)
    ├── mnist/
    │   ├── mnist_X.npy          (gerado por prepare_datasets.py, ~443 features)
    │   └── mnist_y.npy          (gerado por prepare_datasets.py, binario 0-4 vs 5-9)
    ├── creditcard/
    │   ├── creditcard.csv       (opcional, baixar do Kaggle; senao usa OpenML)
    │   ├── creditcard_X.npy     (gerado por prepare_datasets.py, 31 features)
    │   └── creditcard_y.npy     (gerado por prepare_datasets.py)
    ├── epsilon/
    │   ├── epsilon_normalized.bz2       (treino, baixar de LIBSVM)
    │   ├── epsilon_normalized.t.bz2     (teste, baixar de LIBSVM)
    │   ├── epsilon_X.npy        (gerado por prepare_datasets.py)
    │   └── epsilon_y.npy        (gerado por prepare_datasets.py)
    └── avazu/
        ├── train.csv            (ou train.gz, baixar do Kaggle)
        ├── avazu_X.npy          (gerado por prepare_datasets.py)
        └── avazu_y.npy          (gerado por prepare_datasets.py)
"""

import os

# Raiz da pasta data/
_BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def data_dir(dataset_name: str) -> str:
    """Retorna o path da pasta de um dataset."""
    return os.path.join(_BASE_DIR, dataset_name)


def npy_paths(dataset_name: str) -> tuple:
    """
    Retorna (X_path, y_path) dos .npy preprocessados de um dataset.

    Returns:
        (str, str): Caminhos para os arquivos X e y.
    """
    d = data_dir(dataset_name)
    x_path = os.path.join(d, f"{dataset_name}_X.npy")
    y_path = os.path.join(d, f"{dataset_name}_y.npy")
    return x_path, y_path


def is_prepared(dataset_name: str) -> bool:
    """Verifica se os .npy ja foram gerados para um dataset."""
    x_path, y_path = npy_paths(dataset_name)
    return os.path.exists(x_path) and os.path.exists(y_path)


# Metadados dos datasets
DATASET_INFO = {
    "higgs": {
        "description": "Higgs boson (reduzido, 50k amostras) — ja disponivel",
        "features": 28,
        "task": "binary",
        "source": "OpenML (higgs v2)",
    },
    "higgs_full": {
        "description": "Higgs boson (completo, 11M) — 28 features + interacoes fisicas",
        "features": "28 + interacoes (~45 apos preprocessing)",
        "task": "binary",
        "source": "UCI ML Repository — HIGGS.csv.gz",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz",
        "preprocessing": [
            "Clipping de outliers a 5σ",
            "Interacoes entre features de alto nivel (produtos)",
            "Remocao de features com correlacao >= 0.95",
        ],
    },
    "epsilon": {
        "description": "Epsilon (500k) — 2000 → ~500 features apos selecao",
        "features": "2000 → ~500 (selecao por variancia)",
        "task": "binary",
        "source": "LIBSVM — epsilon_normalized",
        "url_train": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.bz2",
        "url_test": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.t.bz2",
        "preprocessing": [
            "Remocao de features constantes",
            "Selecao por variancia (top 500)",
            "Remocao de features com correlacao >= 0.95",
        ],
    },
    "creditcard": {
        "description": "Credit Card Fraud (285k) — 28 PCA + 3 engenheiradas, desbalanceado 0.17%",
        "features": "30 -> 31 (V1-V28 + hour_sin/cos + amount_log1p)",
        "task": "binary",
        "source": "ULB via OpenML (data_id=1597) ou Kaggle",
        "url": "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
        "preprocessing": [
            "Remocao da coluna Time (sequencial, sem sentido em FL)",
            "Criacao de hour_sin/hour_cos (ciclo 24h, data-independent)",
            "Amount normalizado via log1p (data-independent)",
            "V1-V28 mantidas como estao (ja PCA-normalizadas pela ULB)",
        ],
    },
    "mnist": {
        "description": "MNIST (70k) — digitos 0-4 vs 5-9, ~580 features apos selecao",
        "features": "784 → ~580 (remocao de pixels constantes e baixa variancia)",
        "task": "binary",
        "source": "OpenML (mnist_784 v1)",
        "preprocessing": [
            "Normalizacao [0, 1] (divisao por 255)",
            "Remocao de pixels constantes (bordas sempre pretas)",
            "Remocao de features com variancia < 0.01",
            "Labels binarizadas: 0-4 → classe 0, 5-9 → classe 1",
        ],
    },
    "avazu": {
        "description": "Avazu CTR (40M) — temporal + Feature Hashing 1024d (data-independent)",
        "features": "5 temporal + 1024 hashed = 1029",
        "task": "binary",
        "source": "Kaggle — Avazu Click-Through Rate Prediction",
        "url": "https://www.kaggle.com/c/avazu-ctr-prediction/data",
        "preprocessing": [
            "Feature engineering temporal (hora, periodo, sin/cos ciclico)",
            "Feature Hashing de TODAS as categoricas (1024d, data-independent para FL)",
            "Sem frequency encoding (incompativel com FL — exige dados globais)",
        ],
    },
}
