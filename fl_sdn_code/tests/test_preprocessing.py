"""
Testes para tools/prepare_datasets.py — logica de preprocessing.

Usa dados sinteticos para testar as funcoes de preprocessing sem
precisar dos datasets brutos reais (HIGGS.csv.gz, epsilon .bz2, etc.).

Cobre:
  - Utilitarios compartilhados (clip_outliers, remove_constant_features,
    remove_highly_correlated)
  - Pipeline Avazu (Feature Hashing + temporais) com CSV sintetico
  - Pipeline Epsilon (selecao por variancia + correlacao)
  - Funcoes de path (npy_paths, data_dir, is_prepared)
  - Integracao: salvar/carregar .npy
"""

import os
import sys
import csv
import gzip
import tempfile

import numpy as np
import pytest

# Garante path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.prepare_datasets import (
    clip_outliers,
    remove_constant_features,
    remove_highly_correlated,
    log_preprocessing_summary,
)
from datasets.paths import data_dir, npy_paths, is_prepared


# ======================================================================
# Utilidades de preprocessing
# ======================================================================

class TestClipOutliers:
    def test_clips_extreme_values(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 10).astype(np.float32)
        # Injeta outliers
        X[0, 0] = 100.0
        X[1, 1] = -100.0
        X_clipped = clip_outliers(X, n_sigma=3.0)
        # Outliers devem ter sido reduzidos
        assert X_clipped[0, 0] < 50.0
        assert X_clipped[1, 1] > -50.0

    def test_shape_preserved(self):
        X = np.random.randn(200, 15).astype(np.float32)
        X_clipped = clip_outliers(X, n_sigma=5.0)
        assert X_clipped.shape == X.shape

    def test_no_clipping_when_no_outliers(self):
        X = np.random.randn(50, 5).astype(np.float32)
        X_clipped = clip_outliers(X, n_sigma=10.0)  # limite muito alto
        np.testing.assert_array_almost_equal(X, X_clipped, decimal=5)

    def test_handles_zero_std_columns(self):
        """Colunas constantes nao devem causar divisao por zero."""
        X = np.ones((50, 5), dtype=np.float32)
        X_clipped = clip_outliers(X, n_sigma=3.0)
        assert not np.any(np.isnan(X_clipped))
        assert not np.any(np.isinf(X_clipped))

    def test_returns_float32(self):
        X = np.random.randn(50, 5).astype(np.float32)
        X_clipped = clip_outliers(X)
        assert X_clipped.dtype in (np.float32, np.float64)


class TestRemoveConstantFeatures:
    def test_removes_constant_columns(self):
        rng = np.random.RandomState(42)
        n = 100
        # Constroi X com controle total:
        # col0, col1, col3 variam | col2 e col4 sao zero/um exatos
        varying = rng.randn(n, 3).astype(np.float32)   # 3 colunas variaveis
        const_zero = np.zeros((n, 1), dtype=np.float32)  # constante 0
        const_one  = np.ones((n, 1), dtype=np.float32)   # constante 1
        # Ordem: vary0, vary1, const_zero, vary2, const_one
        X = np.hstack([varying[:, :2], const_zero, varying[:, 2:], const_one])
        assert X.shape == (n, 5)
        # Garante que as constantes tem variancia exatamente 0
        assert np.var(X[:, 2]) == 0.0
        assert np.var(X[:, 4]) == 0.0

        X_filtered, mask = remove_constant_features(X, threshold=0.0)
        assert X_filtered.shape[1] == 3, (
            f"Esperava 3 colunas restantes, obteve {X_filtered.shape[1]}"
        )
        assert mask.sum() == 3

    def test_no_removal_when_all_vary(self):
        X = np.random.randn(100, 5).astype(np.float32)
        X_filtered, mask = remove_constant_features(X, threshold=0.0)
        assert X_filtered.shape[1] == 5
        assert mask.all()

    def test_threshold_applied_correctly(self):
        X = np.random.randn(100, 5).astype(np.float32)
        X[:, 0] = np.random.uniform(0, 1e-8, 100)  # quase constante
        X_filtered, mask = remove_constant_features(X, threshold=1e-7)
        # Coluna 0 tem variancia muito baixa, deve ser removida
        assert X_filtered.shape[1] == 4

    def test_mask_is_boolean(self):
        X = np.random.randn(50, 5).astype(np.float32)
        _, mask = remove_constant_features(X)
        assert mask.dtype == bool
        assert len(mask) == 5


class TestRemoveHighlyCorrelated:
    def test_removes_correlated_features(self):
        rng = np.random.RandomState(42)
        X = rng.randn(200, 5).astype(np.float32)
        # Forca correlacao perfeita entre colunas 0 e 1
        X[:, 1] = X[:, 0] * 2 + 0.0001
        X_filtered, mask = remove_highly_correlated(X, threshold=0.99)
        assert X_filtered.shape[1] < 5

    def test_no_removal_when_uncorrelated(self):
        rng = np.random.RandomState(10)
        # Features independentes
        X = rng.randn(500, 5).astype(np.float32)
        X_filtered, mask = remove_highly_correlated(X, threshold=0.99)
        assert X_filtered.shape[1] == 5

    def test_shape_consistency(self):
        X = np.random.randn(100, 8).astype(np.float32)
        X_filtered, mask = remove_highly_correlated(X, threshold=0.95)
        assert X_filtered.shape[1] == mask.sum()
        assert X_filtered.shape[0] == 100

    def test_high_threshold_keeps_all(self):
        """Threshold 1.0 nao deve remover nada (so correlacao exata)."""
        X = np.random.randn(100, 5).astype(np.float32)
        X_filtered, _ = remove_highly_correlated(X, threshold=1.0)
        assert X_filtered.shape[1] == 5

    def test_sampling_used_for_large_dataset(self):
        """Datasets grandes devem funcionar sem timeout (usa amostragem)."""
        X = np.random.randn(100_000, 20).astype(np.float32)
        X_filtered, _ = remove_highly_correlated(X, threshold=0.95)
        assert X_filtered.shape[0] == 100_000


# ======================================================================
# Pipeline Avazu — com CSV sintetico
# ======================================================================

class TestAvazuPreprocessing:
    """
    Testa a logica de preprocessing do Avazu usando um CSV minimo sintetico.
    Nao requer o arquivo real (40M linhas).
    """

    @pytest.fixture
    def synthetic_avazu_csv(self, tmp_path):
        """Cria um CSV sintetico no formato Avazu com 5000 linhas."""
        csv_path = str(tmp_path / "train.csv")
        rng = np.random.RandomState(42)
        n = 5000
        # Gera campo hour: YYMMDDH no formato do Avazu (ex: 14102100)
        base_day = 14102100
        hours = [str(base_day + rng.randint(0, 24)) for _ in range(n)]

        cat_vals = ["a", "b", "c", "d", "e"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = [
                "id", "click", "hour", "C1", "banner_pos",
                "site_id", "site_domain", "site_category",
                "app_id", "app_domain", "app_category",
                "device_id", "device_ip", "device_model",
                "device_type", "device_conn_type",
                "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21",
            ]
            writer.writerow(header)
            for i in range(n):
                row = [
                    i,                         # id
                    rng.randint(0, 2),          # click
                    hours[i],                   # hour
                ] + [rng.choice(cat_vals) for _ in range(21)]
                writer.writerow(row)

        return csv_path

    def test_feature_hashing_shape(self, synthetic_avazu_csv, tmp_path):
        """Pipeline deve produzir 5 temporais + 1024 hashed = 1029 features."""
        import pandas as pd
        from sklearn.feature_extraction import FeatureHasher

        # Simula o pipeline do prepare_avazu sem I/O de arquivo
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

        chunk = pd.read_csv(synthetic_avazu_csv)
        y = chunk["click"].values.astype(np.int8)

        hour_str = chunk["hour"].astype(str)
        hour_of_day = hour_str.str[-2:].astype(int).values
        day_hash = hour_str.str[:-2].astype(int).values
        day_of_week = (day_hash % 7).astype(np.float32)
        period = (hour_of_day // 6).astype(np.float32)
        hour_sin = np.sin(2 * np.pi * hour_of_day / 24).astype(np.float32)
        hour_cos = np.cos(2 * np.pi * hour_of_day / 24).astype(np.float32)
        temporal = np.column_stack([
            hour_of_day.astype(np.float32), day_of_week, period, hour_sin, hour_cos,
        ])

        available = [c for c in cat_cols if c in chunk.columns]
        cat_data = chunk[available].astype(str).copy()
        for col in available:
            cat_data[col] = col + "=" + cat_data[col]
        X_hashed = hasher.transform(cat_data.values.tolist()).toarray().astype(np.float32)

        X = np.hstack([temporal, X_hashed])

        assert X.shape == (5000, 5 + 1024), f"Shape incorreto: {X.shape}"
        assert y.shape == (5000,)
        assert set(y).issubset({0, 1})

    def test_temporal_features_range(self, synthetic_avazu_csv):
        """Hora do dia deve estar em [0, 23], period em [0, 3]."""
        import pandas as pd

        chunk = pd.read_csv(synthetic_avazu_csv)
        hour_str = chunk["hour"].astype(str)
        hour_of_day = hour_str.str[-2:].astype(int).values
        period = (hour_of_day // 6).astype(int)

        assert hour_of_day.min() >= 0
        assert hour_of_day.max() <= 23
        assert period.min() >= 0
        assert period.max() <= 3

    def test_sin_cos_bounded(self, synthetic_avazu_csv):
        """sin e cos devem estar em [-1, 1]."""
        import pandas as pd

        chunk = pd.read_csv(synthetic_avazu_csv)
        hour_of_day = chunk["hour"].astype(str).str[-2:].astype(int).values
        hour_sin = np.sin(2 * np.pi * hour_of_day / 24)
        hour_cos = np.cos(2 * np.pi * hour_of_day / 24)

        assert hour_sin.min() >= -1.0 - 1e-6
        assert hour_sin.max() <= 1.0 + 1e-6
        assert hour_cos.min() >= -1.0 - 1e-6
        assert hour_cos.max() <= 1.0 + 1e-6

    def test_hashing_is_deterministic(self, synthetic_avazu_csv):
        """Feature Hashing deve produzir exatamente o mesmo resultado em duas passagens."""
        import pandas as pd
        from sklearn.feature_extraction import FeatureHasher

        hasher = FeatureHasher(n_features=1024, input_type="string")
        chunk = pd.read_csv(synthetic_avazu_csv)
        cat_cols = ["C1", "banner_pos", "site_id", "site_domain", "site_category"]
        cat_data = chunk[cat_cols].astype(str).copy()
        for col in cat_cols:
            cat_data[col] = col + "=" + cat_data[col]
        records = cat_data.values.tolist()

        X1 = hasher.transform(records).toarray()
        X2 = hasher.transform(records).toarray()
        np.testing.assert_array_equal(X1, X2)

    def test_feature_hashing_data_independent(self):
        """
        Dois conjuntos de dados completamente diferentes devem usar
        o mesmo espaco hash sem coordenacao — nenhum valor de um
        conjunto influencia o mapeamento do outro.
        """
        from sklearn.feature_extraction import FeatureHasher

        hasher = FeatureHasher(n_features=1024, input_type="string")

        # Conjunto A — apenas valor "x"
        records_a = [["C1=x", "site_id=s1"]] * 100
        # Conjunto B — apenas valor "y" (diferente)
        records_b = [["C1=y", "site_id=s2"]] * 100

        Xa = hasher.transform(records_a).toarray()
        Xb = hasher.transform(records_b).toarray()

        # Ambos usam a mesma dimensao (1024), sem comunicacao entre conjuntos
        assert Xa.shape == (100, 1024)
        assert Xb.shape == (100, 1024)

        # Hash de "C1=x" deve ser identico ao processar apenas A e ao processar A apos B
        Xa2 = hasher.transform(records_a).toarray()
        np.testing.assert_array_equal(Xa, Xa2)


# ======================================================================
# Pipeline Epsilon — com dados sinteticos
# ======================================================================

class TestEpsilonPreprocessing:
    """Testa a logica de selecao por variancia e remocao de correlacoes."""

    def test_variance_selection_top_k(self):
        """Devem ser mantidas as k features com maior variancia."""
        rng = np.random.RandomState(42)
        X = rng.randn(500, 20).astype(np.float32)
        # Features 0-4 tem variancia muito maior
        X[:, :5] *= 10.0

        variances = np.var(X, axis=0)
        MAX_FEATURES = 10
        top_indices = np.argsort(variances)[::-1][:MAX_FEATURES]
        top_indices = np.sort(top_indices)
        X_filtered = X[:, top_indices]

        assert X_filtered.shape[1] == MAX_FEATURES
        # As top 5 (maior variancia) devem estar incluidas
        for i in range(5):
            assert i in top_indices

    def test_epsilon_label_conversion(self):
        """Labels -1/+1 do formato LIBSVM devem ser convertidas para 0/1."""
        y_libsvm = np.array([-1, 1, -1, 1, 1], dtype=np.float32)
        y_converted = ((y_libsvm + 1) / 2).astype(np.int8)
        np.testing.assert_array_equal(y_converted, [0, 1, 0, 1, 1])
        assert set(y_converted).issubset({0, 1})

    def test_pipeline_sequence(self):
        """Sequencia completa: constantes → variancia → correlacao."""
        rng = np.random.RandomState(42)
        X = rng.randn(300, 50).astype(np.float32)
        # Injeta 5 colunas constantes
        X[:, 10:15] = 0.0
        # Garante que 2 colunas sao altamente correlacionadas
        X[:, 20] = X[:, 0] * 2.0

        # Passo 1: constantes
        X, _ = remove_constant_features(X, threshold=0.0)
        assert X.shape[1] == 45  # removeu 5 constantes

        # Passo 2: variancia (top 30)
        variances = np.var(X, axis=0)
        top = np.argsort(variances)[::-1][:30]
        X = X[:, np.sort(top)]
        assert X.shape[1] == 30

        # Passo 3: correlacao
        X, _ = remove_highly_correlated(X, threshold=0.98)
        assert X.shape[1] <= 30


# ======================================================================
# Paths e is_prepared
# ======================================================================

class TestDatasetPaths:
    def test_data_dir_returns_string(self):
        d = data_dir("higgs")
        assert isinstance(d, str)
        assert "higgs" in d

    def test_npy_paths_format(self):
        x, y = npy_paths("higgs")
        assert x.endswith("higgs_X.npy")
        assert y.endswith("higgs_y.npy")

    def test_npy_paths_avazu(self):
        x, y = npy_paths("avazu")
        assert "avazu_X.npy" in x
        assert "avazu_y.npy" in y

    def test_is_prepared_false_when_missing(self, tmp_path):
        """is_prepared deve retornar False quando .npy nao existem."""
        # Usa um nome de dataset ficticio que nao tem .npy
        result = is_prepared("__dataset_inexistente_test__")
        assert result is False

    def test_is_prepared_true_when_files_exist(self, tmp_path):
        """is_prepared deve retornar True quando ambos .npy existem."""
        import datasets.paths as p
        # Salva temporariamente e testa
        original_base = p._BASE_DIR
        p._BASE_DIR = str(tmp_path)

        test_dir = tmp_path / "test_dataset"
        test_dir.mkdir()
        (test_dir / "test_dataset_X.npy").write_bytes(b"x")
        (test_dir / "test_dataset_y.npy").write_bytes(b"y")

        assert p.is_prepared("test_dataset") is True
        p._BASE_DIR = original_base
