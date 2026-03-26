"""
Testes para os datasets MNIST e Credit Card Fraud Detection.

Cobre:
  1. Registro no DatasetRegistry (loader funciona)
  2. Particionamento estratificado entre clientes
  3. Registro de metricas (12 campos + CSV logger)
  4. Overfitting dos modelos (XGBoost, LightGBM, CatBoost) com dados reais
  5. Consistencia de shapes e tipos
"""

import csv
import os

import numpy as np
import pytest

from datasets.registry import DatasetRegistry, stratified_partition
from datasets.paths import is_prepared, npy_paths, DATASET_INFO
from core.metrics import compute_all_metrics, CSV_METRIC_FIELDS
from core.csv_logger import CSVLogger, RESOURCE_FIELDS, NETWORK_FIELDS
from models.factory import ModelFactory


# ======================================================================
# Helpers
# ======================================================================

def _load_dataset_real(name, max_samples=5000):
    """Carrega dataset real e subsample para testes rapidos."""
    x_path, y_path = npy_paths(name)
    X = np.load(x_path)
    y = np.load(y_path).astype(int)
    if len(X) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X, y = X[idx], y[idx]
    return X, y


# ======================================================================
# 1. Registro e carregamento dos datasets
# ======================================================================

class TestDatasetRegistration:
    """Verifica que MNIST e CreditCard estao registrados e carregam."""

    def test_mnist_is_registered(self):
        assert "mnist" in DatasetRegistry.available()

    def test_creditcard_is_registered(self):
        assert "creditcard" in DatasetRegistry.available()

    @pytest.mark.skipif(not is_prepared("mnist"), reason="MNIST nao preparado")
    def test_mnist_server_load(self):
        X_test, y_test = DatasetRegistry.load("mnist", role="server")
        assert X_test.ndim == 2
        assert y_test.ndim == 1
        assert len(X_test) == len(y_test)
        assert set(np.unique(y_test)) == {0, 1}

    @pytest.mark.skipif(not is_prepared("creditcard"), reason="CreditCard nao preparado")
    def test_creditcard_server_load(self):
        X_test, y_test = DatasetRegistry.load("creditcard", role="server")
        assert X_test.ndim == 2
        assert y_test.ndim == 1
        assert len(X_test) == len(y_test)
        assert set(np.unique(y_test)) == {0, 1}

    @pytest.mark.skipif(not is_prepared("mnist"), reason="MNIST nao preparado")
    def test_mnist_client_load(self):
        result = DatasetRegistry.load("mnist", role="client", client_id=0)
        assert len(result) == 4
        X_train, y_train, X_test, y_test = result
        assert X_train.shape[1] == X_test.shape[1]
        assert len(X_train) > 0
        assert len(X_test) > 0

    @pytest.mark.skipif(not is_prepared("creditcard"), reason="CreditCard nao preparado")
    def test_creditcard_client_load(self):
        result = DatasetRegistry.load("creditcard", role="client", client_id=0)
        assert len(result) == 4
        X_train, y_train, X_test, y_test = result
        assert X_train.shape[1] == X_test.shape[1]

    def test_mnist_in_dataset_info(self):
        assert "mnist" in DATASET_INFO
        assert DATASET_INFO["mnist"]["task"] == "binary"

    def test_creditcard_in_dataset_info(self):
        assert "creditcard" in DATASET_INFO
        assert DATASET_INFO["creditcard"]["task"] == "binary"


# ======================================================================
# 2. Particionamento estratificado
# ======================================================================

class TestPartitioning:
    """Verifica que a particao estratificada preserva distribuicao."""

    @pytest.mark.skipif(not is_prepared("mnist"), reason="MNIST nao preparado")
    def test_mnist_partition_covers_all_samples(self):
        """Todas as amostras de treino devem estar em alguma particao."""
        from sklearn.model_selection import train_test_split
        from config import TEST_SIZE, RANDOM_SEED, NUM_CLIENTS

        x_path, y_path = npy_paths("mnist")
        y_full = np.load(y_path).astype(int)
        X_full = np.load(x_path)

        _, _, y_train, _ = train_test_split(
            X_full, y_full, test_size=TEST_SIZE,
            random_state=RANDOM_SEED, stratify=y_full,
        )

        parts = stratified_partition(y_train, NUM_CLIENTS, RANDOM_SEED)

        # Uniao de todas as particoes = todos os indices
        all_idx = np.concatenate(parts)
        assert len(all_idx) == len(y_train)
        assert len(np.unique(all_idx)) == len(y_train)

    @pytest.mark.skipif(not is_prepared("mnist"), reason="MNIST nao preparado")
    def test_mnist_partition_balanced_classes(self):
        """Cada particao deve ter proporcao similar de classes."""
        from sklearn.model_selection import train_test_split
        from config import TEST_SIZE, RANDOM_SEED, NUM_CLIENTS

        x_path, y_path = npy_paths("mnist")
        y_full = np.load(y_path).astype(int)
        X_full = np.load(x_path)

        _, _, y_train, _ = train_test_split(
            X_full, y_full, test_size=TEST_SIZE,
            random_state=RANDOM_SEED, stratify=y_full,
        )

        global_ratio = y_train.mean()
        parts = stratified_partition(y_train, NUM_CLIENTS, RANDOM_SEED)

        for i, idx in enumerate(parts):
            part_ratio = y_train[idx].mean()
            # Proporcao de cada particao deve estar a menos de 2% da global
            assert abs(part_ratio - global_ratio) < 0.02, \
                f"Cliente {i}: ratio={part_ratio:.4f}, global={global_ratio:.4f}"

    @pytest.mark.skipif(not is_prepared("creditcard"), reason="CreditCard nao preparado")
    def test_creditcard_partition_preserves_fraud_ratio(self):
        """Particao estratificada deve manter ~0.17% de fraudes em cada cliente."""
        from sklearn.model_selection import train_test_split
        from config import TEST_SIZE, RANDOM_SEED, NUM_CLIENTS

        x_path, y_path = npy_paths("creditcard")
        y_full = np.load(y_path).astype(int)
        X_full = np.load(x_path)

        _, _, y_train, _ = train_test_split(
            X_full, y_full, test_size=TEST_SIZE,
            random_state=RANDOM_SEED, stratify=y_full,
        )

        global_fraud_rate = y_train.mean()
        parts = stratified_partition(y_train, NUM_CLIENTS, RANDOM_SEED)

        for i, idx in enumerate(parts):
            fraud_rate = y_train[idx].mean()
            # Tolerancia de 0.1% para o desbalanceamento extremo
            assert abs(fraud_rate - global_fraud_rate) < 0.001, \
                f"Cliente {i}: fraud_rate={fraud_rate:.5f}, global={global_fraud_rate:.5f}"

    @pytest.mark.skipif(not is_prepared("creditcard"), reason="CreditCard nao preparado")
    def test_creditcard_each_client_has_frauds(self):
        """Cada cliente deve ter pelo menos 1 fraude (particao estratificada)."""
        from sklearn.model_selection import train_test_split
        from config import TEST_SIZE, RANDOM_SEED, NUM_CLIENTS

        x_path, y_path = npy_paths("creditcard")
        y_full = np.load(y_path).astype(int)
        X_full = np.load(x_path)

        _, _, y_train, _ = train_test_split(
            X_full, y_full, test_size=TEST_SIZE,
            random_state=RANDOM_SEED, stratify=y_full,
        )

        parts = stratified_partition(y_train, NUM_CLIENTS, RANDOM_SEED)

        for i, idx in enumerate(parts):
            n_fraud = y_train[idx].sum()
            assert n_fraud > 0, f"Cliente {i} sem fraudes!"

    @pytest.mark.skipif(not is_prepared("mnist"), reason="MNIST nao preparado")
    def test_mnist_partitions_are_disjoint(self):
        """Nenhuma amostra deve aparecer em duas particoes."""
        from sklearn.model_selection import train_test_split
        from config import TEST_SIZE, RANDOM_SEED, NUM_CLIENTS

        x_path, y_path = npy_paths("mnist")
        y_full = np.load(y_path).astype(int)
        X_full = np.load(x_path)

        _, _, y_train, _ = train_test_split(
            X_full, y_full, test_size=TEST_SIZE,
            random_state=RANDOM_SEED, stratify=y_full,
        )

        parts = stratified_partition(y_train, NUM_CLIENTS, RANDOM_SEED)
        seen = set()
        for idx in parts:
            overlap = seen.intersection(idx)
            assert len(overlap) == 0, f"Indices duplicados: {overlap}"
            seen.update(idx)

    @pytest.mark.skipif(not is_prepared("creditcard"), reason="CreditCard nao preparado")
    def test_creditcard_partitions_roughly_equal_size(self):
        """Particoes devem ter tamanhos aproximadamente iguais."""
        from sklearn.model_selection import train_test_split
        from config import TEST_SIZE, RANDOM_SEED, NUM_CLIENTS

        x_path, y_path = npy_paths("creditcard")
        y_full = np.load(y_path).astype(int)
        X_full = np.load(x_path)

        _, _, y_train, _ = train_test_split(
            X_full, y_full, test_size=TEST_SIZE,
            random_state=RANDOM_SEED, stratify=y_full,
        )

        parts = stratified_partition(y_train, NUM_CLIENTS, RANDOM_SEED)
        sizes = [len(p) for p in parts]
        expected = len(y_train) / NUM_CLIENTS

        for i, sz in enumerate(sizes):
            # Tolerancia de 1% do total
            assert abs(sz - expected) < len(y_train) * 0.01, \
                f"Cliente {i}: {sz} amostras, esperado ~{expected:.0f}"


# ======================================================================
# 3. Registro de metricas (12 campos + CSV logger)
# ======================================================================

class TestMetricsRegistration:
    """Verifica que compute_all_metrics funciona com dados reais dos datasets."""

    @pytest.mark.skipif(not is_prepared("mnist"), reason="MNIST nao preparado")
    def test_mnist_metrics_all_12_fields(self):
        """Treinar modelo real no MNIST e verificar que todas as 12 metricas sao geradas."""
        X, y = _load_dataset_real("mnist", max_samples=3000)
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

        model = ModelFactory.train("xgboost", X_tr, y_tr,
                                   client_id=0, server_round=1, local_epochs=10)
        y_prob = model.predict_proba(X_te)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = compute_all_metrics(y_te, y_pred, y_prob)

        for field in CSV_METRIC_FIELDS:
            assert field in metrics, f"Campo '{field}' ausente nas metricas MNIST"

    @pytest.mark.skipif(not is_prepared("creditcard"), reason="CreditCard nao preparado")
    def test_creditcard_metrics_all_12_fields(self):
        """Treinar modelo real no CreditCard e verificar todas as 12 metricas."""
        X, y = _load_dataset_real("creditcard", max_samples=5000)
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y,
        )

        model = ModelFactory.train("xgboost", X_tr, y_tr,
                                   client_id=0, server_round=1, local_epochs=10)
        y_prob = model.predict_proba(X_te)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = compute_all_metrics(y_te, y_pred, y_prob)

        for field in CSV_METRIC_FIELDS:
            assert field in metrics, f"Campo '{field}' ausente nas metricas CreditCard"

    @pytest.mark.skipif(not is_prepared("mnist"), reason="MNIST nao preparado")
    def test_mnist_csv_logger_writes_correctly(self, tmp_run_dir):
        """Simula logging de um round completo com dados MNIST reais."""
        X, y = _load_dataset_real("mnist", max_samples=2000)
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

        model = ModelFactory.train("xgboost", X_tr, y_tr,
                                   client_id=0, server_round=1, local_epochs=10)
        y_prob = model.predict_proba(X_te)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = compute_all_metrics(y_te, y_pred, y_prob)

        logger = CSVLogger(tmp_run_dir, exp_name="mnist_test")
        logger.start_timer()
        resource = {"training_time_avg": 2.5, "model_size_kb_avg": 45.0}
        logger.log_round(1, metrics, resource_metrics=resource)

        with open(logger.log_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert float(rows[0]["accuracy"]) > 0
        assert float(rows[0]["training_time_avg"]) == 2.5

    @pytest.mark.skipif(not is_prepared("creditcard"), reason="CreditCard nao preparado")
    def test_creditcard_csv_logger_writes_correctly(self, tmp_run_dir):
        """Simula logging de um round completo com dados CreditCard reais."""
        X, y = _load_dataset_real("creditcard", max_samples=5000)
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y,
        )

        model = ModelFactory.train("xgboost", X_tr, y_tr,
                                   client_id=0, server_round=1, local_epochs=10)
        y_prob = model.predict_proba(X_te)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = compute_all_metrics(y_te, y_pred, y_prob)

        logger = CSVLogger(tmp_run_dir, exp_name="cc_test")
        logger.start_timer()
        logger.log_round(1, metrics)

        with open(logger.log_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        # CreditCard: com 99.8% legitimas, accuracy deve ser alta mesmo sem treino bom
        assert float(rows[0]["accuracy"]) > 0

    @pytest.mark.skipif(not is_prepared("creditcard"), reason="CreditCard nao preparado")
    def test_creditcard_metrics_ranges_valid(self):
        """Todas as metricas devem estar em ranges validos para CreditCard."""
        X, y = _load_dataset_real("creditcard", max_samples=5000)
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y,
        )

        model = ModelFactory.train("xgboost", X_tr, y_tr,
                                   client_id=0, server_round=1, local_epochs=20)
        y_prob = model.predict_proba(X_te)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = compute_all_metrics(y_te, y_pred, y_prob)

        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["auc"] <= 1.0
        assert 0.0 <= metrics["pr_auc"] <= 1.0
        assert metrics["log_loss"] >= 0.0
        assert -1.0 <= metrics["mcc"] <= 1.0
        assert -1.0 <= metrics["cohen_kappa"] <= 1.0


# ======================================================================
# 4. Overfitting — modelos conseguem aprender com os datasets
# ======================================================================

class TestOverfitting:
    """
    Verifica que os 3 modelos (XGBoost, LightGBM, CatBoost) conseguem
    overfitar nos dados reais. Se um modelo nao consegue nem overfit no
    treino, algo esta errado com o preprocessing ou a compatibilidade.
    """

    @pytest.mark.skipif(not is_prepared("mnist"), reason="MNIST nao preparado")
    @pytest.mark.parametrize("model_type", ["xgboost", "lightgbm", "catboost"])
    def test_mnist_overfit_train(self, model_type):
        """Modelo deve atingir >95% accuracy no treino com epocas suficientes."""
        X, y = _load_dataset_real("mnist", max_samples=2000)
        model = ModelFactory.train(model_type, X, y,
                                   client_id=0, server_round=1, local_epochs=50)
        y_pred = model.predict(X)
        train_acc = (y_pred == y).mean()
        assert train_acc > 0.95, \
            f"{model_type} no MNIST: train_acc={train_acc:.4f} (esperado >0.95)"

    @pytest.mark.skipif(not is_prepared("creditcard"), reason="CreditCard nao preparado")
    @pytest.mark.parametrize("model_type", ["xgboost", "lightgbm", "catboost"])
    def test_creditcard_overfit_train(self, model_type):
        """Modelo deve atingir >99% accuracy no treino (dataset quase todo classe 0)."""
        X, y = _load_dataset_real("creditcard", max_samples=5000)
        model = ModelFactory.train(model_type, X, y,
                                   client_id=0, server_round=1, local_epochs=50)
        y_pred = model.predict(X)
        train_acc = (y_pred == y).mean()
        assert train_acc > 0.99, \
            f"{model_type} no CreditCard: train_acc={train_acc:.4f} (esperado >0.99)"

    @pytest.mark.skipif(not is_prepared("mnist"), reason="MNIST nao preparado")
    @pytest.mark.parametrize("model_type", ["xgboost", "lightgbm", "catboost"])
    def test_mnist_generalizes_above_baseline(self, model_type):
        """Modelo deve generalizar acima de 80% no teste (baseline ~51% = classe majoritaria)."""
        X, y = _load_dataset_real("mnist", max_samples=5000)
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

        model = ModelFactory.train(model_type, X_tr, y_tr,
                                   client_id=0, server_round=1, local_epochs=30)
        y_pred = model.predict(X_te)
        test_acc = (y_pred == y_te).mean()
        assert test_acc > 0.80, \
            f"{model_type} no MNIST teste: acc={test_acc:.4f} (esperado >0.80)"

    @pytest.mark.skipif(not is_prepared("creditcard"), reason="CreditCard nao preparado")
    @pytest.mark.parametrize("model_type", ["xgboost", "lightgbm", "catboost"])
    def test_creditcard_auc_above_baseline(self, model_type):
        """AUC deve ser >0.70 no teste (baseline=0.50, random)."""
        X, y = _load_dataset_real("creditcard", max_samples=10000)
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y,
        )

        model = ModelFactory.train(model_type, X_tr, y_tr,
                                   client_id=0, server_round=1, local_epochs=30)
        y_prob = model.predict_proba(X_te)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = compute_all_metrics(y_te, y_pred, y_prob)

        assert metrics["auc"] > 0.70, \
            f"{model_type} no CreditCard: AUC={metrics['auc']:.4f} (esperado >0.70)"

    @pytest.mark.skipif(not is_prepared("mnist"), reason="MNIST nao preparado")
    def test_mnist_more_epochs_improves_train(self):
        """Mais epocas locais devem melhorar (ou manter) accuracy de treino."""
        X, y = _load_dataset_real("mnist", max_samples=2000)

        model_few = ModelFactory.train("xgboost", X, y,
                                       client_id=0, server_round=1, local_epochs=5)
        model_many = ModelFactory.train("xgboost", X, y,
                                        client_id=0, server_round=1, local_epochs=50)

        acc_few = (model_few.predict(X) == y).mean()
        acc_many = (model_many.predict(X) == y).mean()

        assert acc_many >= acc_few, \
            f"Mais epocas piorou: {acc_few:.4f} -> {acc_many:.4f}"

    @pytest.mark.skipif(not is_prepared("creditcard"), reason="CreditCard nao preparado")
    def test_creditcard_warm_start_works(self):
        """Warm start deve funcionar sem erro nos dados reais de CreditCard."""
        X, y = _load_dataset_real("creditcard", max_samples=3000)

        model1 = ModelFactory.train("xgboost", X, y,
                                    client_id=0, server_round=1, local_epochs=10)
        model2 = ModelFactory.train("xgboost", X, y,
                                    client_id=0, server_round=2, local_epochs=10,
                                    warm_start_model=model1)
        assert model2 is not None
        y_pred = model2.predict(X)
        assert len(y_pred) == len(y)


# ======================================================================
# 5. Consistencia de shapes, tipos e features
# ======================================================================

class TestDataConsistency:
    """Verifica integridade dos .npy gerados."""

    @pytest.mark.skipif(not is_prepared("mnist"), reason="MNIST nao preparado")
    def test_mnist_npy_shapes_match(self):
        x_path, y_path = npy_paths("mnist")
        X = np.load(x_path)
        y = np.load(y_path)
        assert X.shape[0] == y.shape[0]
        assert X.shape[0] == 70_000
        # Deve ter entre 400-500 features apos remocao de baixa variancia
        assert 400 <= X.shape[1] <= 500

    @pytest.mark.skipif(not is_prepared("creditcard"), reason="CreditCard nao preparado")
    def test_creditcard_npy_shapes_match(self):
        x_path, y_path = npy_paths("creditcard")
        X = np.load(x_path)
        y = np.load(y_path)
        assert X.shape[0] == y.shape[0]
        assert X.shape[0] == 284_807
        # 28 PCA + hour_sin + hour_cos + amount_log1p (ou menos se Time ausente)
        assert 29 <= X.shape[1] <= 31

    @pytest.mark.skipif(not is_prepared("mnist"), reason="MNIST nao preparado")
    def test_mnist_dtype_float32(self):
        x_path, _ = npy_paths("mnist")
        X = np.load(x_path)
        assert X.dtype == np.float32

    @pytest.mark.skipif(not is_prepared("creditcard"), reason="CreditCard nao preparado")
    def test_creditcard_dtype_float32(self):
        x_path, _ = npy_paths("creditcard")
        X = np.load(x_path)
        assert X.dtype == np.float32

    @pytest.mark.skipif(not is_prepared("mnist"), reason="MNIST nao preparado")
    def test_mnist_no_nan_or_inf(self):
        x_path, _ = npy_paths("mnist")
        X = np.load(x_path)
        assert not np.any(np.isnan(X)), "MNIST contem NaN"
        assert not np.any(np.isinf(X)), "MNIST contem Inf"

    @pytest.mark.skipif(not is_prepared("creditcard"), reason="CreditCard nao preparado")
    def test_creditcard_no_nan_or_inf(self):
        x_path, _ = npy_paths("creditcard")
        X = np.load(x_path)
        assert not np.any(np.isnan(X)), "CreditCard contem NaN"
        assert not np.any(np.isinf(X)), "CreditCard contem Inf"

    @pytest.mark.skipif(not is_prepared("mnist"), reason="MNIST nao preparado")
    def test_mnist_pixel_values_normalized(self):
        """Pixels devem estar em [0, 1] apos normalizacao."""
        x_path, _ = npy_paths("mnist")
        X = np.load(x_path)
        assert X.min() >= 0.0, f"MNIST min={X.min()}"
        assert X.max() <= 1.0, f"MNIST max={X.max()}"

    @pytest.mark.skipif(not is_prepared("mnist"), reason="MNIST nao preparado")
    def test_mnist_labels_binary(self):
        _, y_path = npy_paths("mnist")
        y = np.load(y_path)
        assert set(np.unique(y)) == {0, 1}

    @pytest.mark.skipif(not is_prepared("creditcard"), reason="CreditCard nao preparado")
    def test_creditcard_labels_binary(self):
        _, y_path = npy_paths("creditcard")
        y = np.load(y_path)
        assert set(np.unique(y)) == {0, 1}

    @pytest.mark.skipif(not is_prepared("creditcard"), reason="CreditCard nao preparado")
    def test_creditcard_fraud_rate(self):
        """Taxa de fraude deve estar entre 0.1% e 0.3%."""
        _, y_path = npy_paths("creditcard")
        y = np.load(y_path).astype(int)
        fraud_rate = y.mean()
        assert 0.001 <= fraud_rate <= 0.003, \
            f"Taxa de fraude inesperada: {fraud_rate:.5f}"

    @pytest.mark.skipif(not is_prepared("mnist"), reason="MNIST nao preparado")
    def test_mnist_no_constant_features(self):
        """Nenhuma feature deve ter variancia zero apos preprocessing."""
        x_path, _ = npy_paths("mnist")
        X = np.load(x_path)
        variances = np.var(X, axis=0)
        assert np.all(variances > 0), \
            f"Features constantes encontradas: {np.sum(variances == 0)}"
