"""Testes para models/factory.py e datasets/registry.py."""

import numpy as np
import pytest

from models.factory import ModelFactory
from datasets.registry import DatasetRegistry


# ======================================================================
# ModelFactory
# ======================================================================

class TestModelFactory:
    @pytest.mark.parametrize("model_type", ["xgboost", "lightgbm", "catboost"])
    def test_train_all_model_types(self, sample_data, model_type):
        X, y = sample_data
        model = ModelFactory.train(
            model_type, X, y,
            client_id=0, server_round=1, local_epochs=10,
        )
        assert model is not None
        pred = model.predict(X)
        assert len(pred) == len(y)

    @pytest.mark.parametrize("model_type", ["xgboost", "lightgbm", "catboost"])
    def test_predict_proba_works(self, sample_data, model_type):
        X, y = sample_data
        model = ModelFactory.train(
            model_type, X, y,
            client_id=0, server_round=1, local_epochs=10,
        )
        proba = model.predict_proba(X)
        assert proba.shape == (len(y), 2)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_invalid_model_raises(self, sample_data):
        X, y = sample_data
        with pytest.raises(ValueError, match="Modelo desconhecido"):
            ModelFactory.train("invalid", X, y, client_id=0,
                             server_round=1, local_epochs=10)

    def test_warm_start_xgboost(self, sample_data):
        X, y = sample_data
        model1 = ModelFactory.train(
            "xgboost", X, y,
            client_id=0, server_round=1, local_epochs=10,
        )
        model2 = ModelFactory.train(
            "xgboost", X, y,
            client_id=0, server_round=2, local_epochs=10,
            warm_start_model=model1,
        )
        assert model2 is not None
        pred = model2.predict(X)
        assert len(pred) == len(y)

    def test_warm_start_lightgbm(self, sample_data):
        X, y = sample_data
        model1 = ModelFactory.train(
            "lightgbm", X, y,
            client_id=0, server_round=1, local_epochs=10,
        )
        model2 = ModelFactory.train(
            "lightgbm", X, y,
            client_id=0, server_round=2, local_epochs=10,
            warm_start_model=model1,
        )
        assert model2 is not None

    def test_model_is_serializable(self, sample_data):
        """Modelo treinado deve ser serializavel via pickle (requisito do FL)."""
        import pickle
        X, y = sample_data
        model = ModelFactory.train(
            "xgboost", X, y,
            client_id=0, server_round=1, local_epochs=10,
        )
        raw = pickle.dumps(model)
        restored = pickle.loads(raw)
        np.testing.assert_array_equal(model.predict(X), restored.predict(X))


# ======================================================================
# DatasetRegistry
# ======================================================================

class TestDatasetRegistry:
    def test_higgs_is_registered(self):
        # Importar higgs.py para registrar o dataset
        import datasets.higgs
        assert "higgs" in DatasetRegistry.available()

    def test_available_returns_list(self):
        avail = DatasetRegistry.available()
        assert isinstance(avail, list)

    def test_invalid_dataset_raises(self):
        with pytest.raises(ValueError, match="Dataset desconhecido"):
            DatasetRegistry.load("inexistente", role="server")

    def test_register_custom_dataset(self):
        @DatasetRegistry.register("test_dummy")
        def load_dummy(role, client_id=0, **kwargs):
            X = np.random.randn(50, 5)
            y = np.random.randint(0, 2, 50)
            if role == "server":
                return X, y
            return X, y, X, y

        assert "test_dummy" in DatasetRegistry.available()
        X_test, y_test = DatasetRegistry.load("test_dummy", role="server")
        assert X_test.shape == (50, 5)

        result = DatasetRegistry.load("test_dummy", role="client", client_id=0)
        assert len(result) == 4
