"""
Testes para o schedule de decaimento exponencial do warm start.

Verifica que:
  - n_estimators e learning_rate decaem corretamente a cada round
  - Os pisos (WARM_START_TREE_MIN_RATIO e WARM_START_LR_MIN_RATIO) sao respeitados
  - extra_params["learning_rate"] e usado como base para o decay do LR
  - Round 1 (sem warm start) usa valores completos
  - Os 3 modelos (xgboost, lightgbm, catboost) treinam corretamente com o schedule
"""

import math

import numpy as np
import pytest

from models.factory import ModelFactory, _compute_warm_start_schedule
from config import (
    WARM_START_TREE_DECAY,
    WARM_START_TREE_MIN_RATIO,
    WARM_START_LR_DECAY,
    WARM_START_LR_MIN_RATIO,
    XGBOOST_PARAMS,
    LIGHTGBM_PARAMS,
    CATBOOST_PARAMS,
)


BASE_LR    = 0.10
LOCAL_EPS  = 100


# ======================================================================
# _compute_warm_start_schedule — logica do schedule
# ======================================================================

class TestWarmStartScheduleLogic:
    """Testa a funcao auxiliar _compute_warm_start_schedule diretamente."""

    def test_round1_full_epochs(self):
        """Round 1: nao ha warm start, schedule deve retornar local_epochs completo."""
        n_new, lr = _compute_warm_start_schedule(BASE_LR, LOCAL_EPS, server_round=1)
        # decay ^ 0 = 1.0, portanto n_new = local_epochs * 1.0 = 100
        assert n_new == LOCAL_EPS
        assert abs(lr - BASE_LR) < 1e-6

    def test_round2_reduced(self):
        """Round 2: primeia reducao — decay^1."""
        n_new, lr = _compute_warm_start_schedule(BASE_LR, LOCAL_EPS, server_round=2)
        expected_ratio = WARM_START_TREE_DECAY ** 1
        expected_n = max(int(LOCAL_EPS * expected_ratio), 10)
        expected_lr = round(max(BASE_LR * WARM_START_LR_MIN_RATIO,
                                BASE_LR * WARM_START_LR_DECAY ** 1), 6)
        assert n_new == expected_n
        assert abs(lr - expected_lr) < 1e-5

    def test_monotone_decrease_n_estimators(self):
        """n_estimators deve ser nao-crescente com o aumento dos rounds."""
        values = [
            _compute_warm_start_schedule(BASE_LR, LOCAL_EPS, r)[0]
            for r in range(1, 15)
        ]
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1], (
                f"n_new deveria ser nao-crescente: round {i+1}={values[i]} "
                f"> round {i+2}={values[i+1]}"
            )

    def test_monotone_decrease_learning_rate(self):
        """learning_rate deve ser nao-crescente com o aumento dos rounds."""
        values = [
            _compute_warm_start_schedule(BASE_LR, LOCAL_EPS, r)[1]
            for r in range(1, 15)
        ]
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1] - 1e-9, (
                f"lr deveria ser nao-crescente: round {i+1}={values[i]:.6f} "
                f"> round {i+2}={values[i+1]:.6f}"
            )

    def test_floor_n_estimators_respected(self):
        """n_new nunca deve cair abaixo de WARM_START_TREE_MIN_RATIO * local_epochs."""
        floor = max(int(LOCAL_EPS * WARM_START_TREE_MIN_RATIO), 10)
        for r in range(1, 21):
            n_new, _ = _compute_warm_start_schedule(BASE_LR, LOCAL_EPS, r)
            assert n_new >= floor, (
                f"Round {r}: n_new={n_new} abaixo do piso={floor}"
            )

    def test_floor_lr_respected(self):
        """lr nunca deve cair abaixo de WARM_START_LR_MIN_RATIO * base_lr."""
        floor = BASE_LR * WARM_START_LR_MIN_RATIO
        for r in range(1, 21):
            _, lr = _compute_warm_start_schedule(BASE_LR, LOCAL_EPS, r)
            assert lr >= floor - 1e-8, (
                f"Round {r}: lr={lr:.6f} abaixo do piso={floor:.6f}"
            )

    def test_stabilizes_after_floor(self):
        """Apos atingir o piso, n_new e lr devem ser constantes."""
        # A partir de um round suficientemente alto ambos devem estar no piso
        n_high, lr_high = _compute_warm_start_schedule(BASE_LR, LOCAL_EPS, server_round=20)
        n_very_high, lr_very_high = _compute_warm_start_schedule(BASE_LR, LOCAL_EPS, server_round=50)
        assert n_high == n_very_high, "n_new deve estar estabilizado no piso"
        assert abs(lr_high - lr_very_high) < 1e-8, "lr deve estar estabilizado no piso"

    def test_extra_params_lr_used_as_base(self):
        """Se extra_params contem learning_rate, ele e usado como base para o decay."""
        tuned_lr = 0.05
        extra = {"learning_rate": tuned_lr, "max_depth": 4}

        _, lr_no_extra = _compute_warm_start_schedule(BASE_LR, LOCAL_EPS, server_round=3)
        _, lr_with_extra = _compute_warm_start_schedule(BASE_LR, LOCAL_EPS, server_round=3,
                                                         extra_params=extra)

        # LR com extra_params deve ser baseado em 0.05, nao em 0.10
        expected = max(tuned_lr * WARM_START_LR_MIN_RATIO,
                       tuned_lr * WARM_START_LR_DECAY ** 2)
        assert abs(lr_with_extra - round(expected, 6)) < 1e-6
        # Deve diferir do sem-extra
        assert abs(lr_with_extra - lr_no_extra) > 1e-4

    def test_extra_params_without_lr_uses_base(self):
        """extra_params sem learning_rate nao interfere no decay do LR."""
        extra = {"max_depth": 4}  # sem lr
        _, lr_no_extra = _compute_warm_start_schedule(BASE_LR, LOCAL_EPS, server_round=5)
        _, lr_with_extra = _compute_warm_start_schedule(BASE_LR, LOCAL_EPS, server_round=5,
                                                        extra_params=extra)
        assert abs(lr_no_extra - lr_with_extra) < 1e-8

    def test_small_local_epochs_floor_of_10(self):
        """Com local_epochs muito pequeno, n_new nunca deve ser < 10."""
        for r in range(1, 10):
            n_new, _ = _compute_warm_start_schedule(BASE_LR, local_epochs=5, server_round=r)
            assert n_new >= 10, f"Round {r}: n_new={n_new} menor que piso absoluto 10"

    def test_decay_formula_correctness(self):
        """Verifica a formula exata para rounds 2, 5, 10."""
        cases = [2, 5, 10]
        for r in cases:
            n_new, lr = _compute_warm_start_schedule(BASE_LR, LOCAL_EPS, server_round=r)
            exp_ratio = max(WARM_START_TREE_MIN_RATIO, WARM_START_TREE_DECAY ** (r - 1))
            exp_n = max(int(LOCAL_EPS * exp_ratio), 10)
            exp_lr = round(max(BASE_LR * WARM_START_LR_MIN_RATIO,
                               BASE_LR * WARM_START_LR_DECAY ** (r - 1)), 6)
            assert n_new == exp_n, f"Round {r}: n_new={n_new}, esperado={exp_n}"
            assert abs(lr - exp_lr) < 1e-6, f"Round {r}: lr={lr}, esperado={exp_lr:.6f}"


# ======================================================================
# Integracao: warm start real com os 3 modelos
# ======================================================================

class TestWarmStartDecayIntegration:
    """Testa que os 3 modelos aceitam o schedule de warm start sem erros."""

    @pytest.fixture
    def train_data(self):
        rng = np.random.RandomState(42)
        X = rng.randn(200, 10).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        return X, y

    @pytest.mark.parametrize("model_type", ["xgboost", "lightgbm", "catboost"])
    def test_warm_start_multi_rounds(self, train_data, model_type):
        """Treina 5 rounds com warm start — modelo deve resultar em predicoes validas."""
        X, y = train_data
        model = None
        for r in range(1, 6):
            model = ModelFactory.train(
                model_type, X, y,
                client_id=0, server_round=r, local_epochs=20,
                warm_start_model=model,
            )
        assert model is not None
        proba = model.predict_proba(X)
        assert proba.shape == (len(y), 2)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    @pytest.mark.parametrize("model_type", ["xgboost", "lightgbm", "catboost"])
    def test_warm_start_with_extra_params(self, train_data, model_type):
        """Warm start com extra_params (lr tunado) funciona para os 3 modelos."""
        X, y = train_data
        extra = {"learning_rate": 0.05}

        model1 = ModelFactory.train(
            model_type, X, y,
            client_id=0, server_round=1, local_epochs=20,
            extra_params=extra,
        )
        model2 = ModelFactory.train(
            model_type, X, y,
            client_id=0, server_round=3, local_epochs=20,
            warm_start_model=model1,
            extra_params=extra,
        )
        assert model2 is not None
        proba = model2.predict_proba(X)
        assert proba.shape == (len(y), 2)

    @pytest.mark.parametrize("model_type", ["xgboost", "lightgbm", "catboost"])
    def test_warm_start_late_round_still_works(self, train_data, model_type):
        """Round 15 (apos estabilizacao no piso) deve funcionar sem erros."""
        X, y = train_data
        model1 = ModelFactory.train(
            model_type, X, y,
            client_id=0, server_round=1, local_epochs=20,
        )
        # Simula round tardio (piso atingido)
        model2 = ModelFactory.train(
            model_type, X, y,
            client_id=0, server_round=15, local_epochs=20,
            warm_start_model=model1,
        )
        assert model2 is not None

    @pytest.mark.parametrize("model_type", ["xgboost", "lightgbm", "catboost"])
    def test_warm_start_serializable(self, train_data, model_type):
        """Modelo treinado com warm start + decay deve ser serializavel via pickle."""
        import pickle
        X, y = train_data
        model1 = ModelFactory.train(
            model_type, X, y,
            client_id=0, server_round=1, local_epochs=12,
        )
        model2 = ModelFactory.train(
            model_type, X, y,
            client_id=0, server_round=4, local_epochs=12,
            warm_start_model=model1,
        )
        raw = pickle.dumps(model2)
        restored = pickle.loads(raw)
        np.testing.assert_array_equal(
            model2.predict(X), restored.predict(X),
            err_msg=f"{model_type}: predicoes divergem apos pickle roundtrip",
        )


# ======================================================================
# Tabela de decaimento (helper para doc/debug)
# ======================================================================

def test_decay_table_snapshot():
    """
    Gera e valida um snapshot da tabela de decaimento.
    Garante que regressions no schedule sao detectadas.
    """
    base_lr = 0.10
    local_epochs = 100

    # Valores esperados calculados manualmente com as constantes atuais
    # (TREE_DECAY=0.85, TREE_MIN=0.30, LR_DECAY=0.93, LR_MIN=0.40)
    expected = {
        # round: (n_new, lr_approx)
        1:  (100, 0.100000),
        2:  (85,  0.093000),
        3:  (72,  0.086490),
        5:  (52,  0.074806),
        8:  (32,  0.060270),
        9:  (30,  0.056051),   # n_new atinge piso (30)
        15: (30,  0.040000),   # lr atinge piso (0.04)
        20: (30,  0.040000),
    }

    for r, (exp_n, exp_lr) in expected.items():
        n_new, lr = _compute_warm_start_schedule(base_lr, local_epochs, server_round=r)
        assert n_new == exp_n, f"Round {r}: n_new={n_new}, esperado={exp_n}"
        assert abs(lr - exp_lr) < 5e-4, f"Round {r}: lr={lr:.6f}, esperado≈{exp_lr:.6f}"
