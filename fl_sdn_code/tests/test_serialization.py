"""Testes para core/serialization.py — roundtrip pickle de modelos."""

import numpy as np
import pytest

from core.serialization import serialize_model, deserialize_model


class TestSerialization:
    def test_roundtrip_xgboost(self, sample_data):
        import xgboost as xgb
        X, y = sample_data
        model = xgb.XGBClassifier(n_estimators=10, random_state=42, verbosity=0)
        model.fit(X, y)

        raw = serialize_model(model)
        restored = deserialize_model(raw)

        pred_orig = model.predict(X)
        pred_rest = restored.predict(X)
        np.testing.assert_array_equal(pred_orig, pred_rest)

    def test_roundtrip_lightgbm(self, sample_data):
        import lightgbm as lgb
        X, y = sample_data
        model = lgb.LGBMClassifier(n_estimators=10, random_state=42, verbosity=-1)
        model.fit(X, y)

        raw = serialize_model(model)
        restored = deserialize_model(raw)

        pred_orig = model.predict(X)
        pred_rest = restored.predict(X)
        np.testing.assert_array_equal(pred_orig, pred_rest)

    def test_roundtrip_catboost(self, sample_data):
        from catboost import CatBoostClassifier
        X, y = sample_data
        model = CatBoostClassifier(iterations=10, random_seed=42, verbose=0)
        model.fit(X, y)

        raw = serialize_model(model)
        restored = deserialize_model(raw)

        pred_orig = model.predict(X)
        pred_rest = restored.predict(X)
        np.testing.assert_array_equal(pred_orig, pred_rest)

    def test_serialized_is_bytes(self, sample_data):
        import xgboost as xgb
        X, y = sample_data
        model = xgb.XGBClassifier(n_estimators=5, random_state=42, verbosity=0)
        model.fit(X, y)

        raw = serialize_model(model)
        assert isinstance(raw, bytes)
        assert len(raw) > 0

    def test_roundtrip_preserves_predict_proba(self, sample_data):
        import xgboost as xgb
        X, y = sample_data
        model = xgb.XGBClassifier(n_estimators=10, random_state=42, verbosity=0)
        model.fit(X, y)

        raw = serialize_model(model)
        restored = deserialize_model(raw)

        prob_orig = model.predict_proba(X)
        prob_rest = restored.predict_proba(X)
        np.testing.assert_array_almost_equal(prob_orig, prob_rest)
