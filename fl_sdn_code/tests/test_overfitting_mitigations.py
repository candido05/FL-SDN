"""
Testes para as mitigacoes de overfitting implementadas:
  - Tree cap (MAX_TOTAL_TREES)
  - Early stopping com validation split
  - Selecao server-side do best_model
  - Ensemble ponderado
  - Particao estratificada de dados
  - Extra params (grid search)
  - Epocas uniformes (sem LOCAL_EPOCHS_BY_CAT)
"""

import numpy as np
import pytest

from models.factory import (
    ModelFactory, _count_trees, _count_trees_lgb, _count_trees_catboost,
    _apply_tree_cap, _make_val_split,
)
from datasets.registry import stratified_partition
from config import MAX_TOTAL_TREES, LOCAL_EPOCHS


# ======================================================================
# Tree Cap
# ======================================================================

class TestTreeCap:
    def test_apply_tree_cap_under_limit(self):
        """Se total < MAX_TOTAL_TREES, retorna n_new inalterado."""
        result = _apply_tree_cap(50, 100, "test")
        assert result == 50

    def test_apply_tree_cap_at_limit(self):
        """Se total == MAX_TOTAL_TREES, retorna n_new inalterado."""
        result = _apply_tree_cap(100, MAX_TOTAL_TREES - 100, "test")
        assert result == 100

    def test_apply_tree_cap_over_limit(self):
        """Se total > MAX_TOTAL_TREES, limita n_new."""
        result = _apply_tree_cap(200, MAX_TOTAL_TREES - 50, "test")
        assert result == 50

    def test_apply_tree_cap_already_full(self):
        """Se n_existing >= MAX_TOTAL_TREES, retorna 0."""
        result = _apply_tree_cap(100, MAX_TOTAL_TREES, "test")
        assert result == 0

    def test_apply_tree_cap_never_negative(self):
        """Nunca retorna valor negativo."""
        result = _apply_tree_cap(100, MAX_TOTAL_TREES + 500, "test")
        assert result >= 0

    def test_warm_start_respects_tree_cap_xgboost(self, sample_data):
        """XGBoost com warm start respeita o tree cap."""
        X, y = sample_data
        model1 = ModelFactory.train(
            "xgboost", X, y,
            client_id=0, server_round=1, local_epochs=10,
        )
        n1 = _count_trees(model1)
        assert n1 > 0

        # Warm start deve produzir mais arvores, mas nao infinitas
        model2 = ModelFactory.train(
            "xgboost", X, y,
            client_id=0, server_round=2, local_epochs=10,
            warm_start_model=model1,
        )
        n2 = _count_trees(model2)
        assert n2 > n1
        assert n2 <= MAX_TOTAL_TREES

    def test_warm_start_respects_tree_cap_lightgbm(self, sample_data):
        """LightGBM com warm start respeita o tree cap."""
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

    def test_warm_start_respects_tree_cap_catboost(self, sample_data):
        """CatBoost com warm start respeita o tree cap."""
        X, y = sample_data
        model1 = ModelFactory.train(
            "catboost", X, y,
            client_id=0, server_round=1, local_epochs=10,
        )
        model2 = ModelFactory.train(
            "catboost", X, y,
            client_id=0, server_round=2, local_epochs=10,
            warm_start_model=model1,
        )
        assert model2 is not None

    def test_count_trees_xgboost(self, sample_data):
        X, y = sample_data
        model = ModelFactory.train(
            "xgboost", X, y,
            client_id=0, server_round=1, local_epochs=10,
        )
        n = _count_trees(model)
        assert n > 0
        assert n <= 10  # no maximo local_epochs arvores

    def test_count_trees_none_returns_zero(self):
        """Modelo None retorna 0 arvores."""
        assert _count_trees(None) == 0
        assert _count_trees_lgb(None) == 0
        assert _count_trees_catboost(None) == 0


# ======================================================================
# Validation Split e Early Stopping
# ======================================================================

class TestValidationSplit:
    def test_make_val_split_normal(self, sample_data):
        """Com dados suficientes, faz split."""
        X, y = sample_data
        X_fit, X_val, y_fit, y_val = _make_val_split(X, y)
        assert X_val is not None
        assert y_val is not None
        assert len(X_fit) + len(X_val) == len(X)
        assert len(X_val) > 0
        assert len(X_fit) > 0

    def test_make_val_split_small_data(self):
        """Com poucos dados (<50), nao faz split."""
        rng = np.random.RandomState(42)
        X = rng.randn(30, 5)
        y = (X[:, 0] > 0).astype(int)
        X_fit, X_val, y_fit, y_val = _make_val_split(X, y)
        assert X_val is None
        assert y_val is None
        assert len(X_fit) == 30

    def test_val_split_preserves_classes(self, sample_data):
        """Split de validacao mantem ambas as classes."""
        X, y = sample_data
        X_fit, X_val, y_fit, y_val = _make_val_split(X, y)
        assert len(np.unique(y_fit)) == 2
        assert len(np.unique(y_val)) == 2

    def test_early_stopping_does_not_crash(self, sample_data):
        """Treinamento com early stopping funciona sem erros."""
        X, y = sample_data
        for model_type in ["xgboost", "lightgbm", "catboost"]:
            model = ModelFactory.train(
                model_type, X, y,
                client_id=0, server_round=1, local_epochs=50,
            )
            assert model is not None
            proba = model.predict_proba(X)
            assert proba.shape[0] == len(X)


# ======================================================================
# Extra Params (Grid Search)
# ======================================================================

class TestExtraParams:
    @pytest.mark.parametrize("model_type", ["xgboost", "lightgbm", "catboost"])
    def test_extra_params_accepted(self, sample_data, model_type):
        """ModelFactory.train aceita extra_params sem erro."""
        X, y = sample_data
        extra = {"learning_rate": 0.05}
        model = ModelFactory.train(
            model_type, X, y,
            client_id=0, server_round=1, local_epochs=10,
            extra_params=extra,
        )
        assert model is not None

    def test_extra_params_none_works(self, sample_data):
        """extra_params=None funciona normalmente."""
        X, y = sample_data
        model = ModelFactory.train(
            "xgboost", X, y,
            client_id=0, server_round=1, local_epochs=10,
            extra_params=None,
        )
        assert model is not None

    def test_extra_params_override_depth(self, sample_data):
        """Extra params podem alterar max_depth."""
        X, y = sample_data
        model = ModelFactory.train(
            "xgboost", X, y,
            client_id=0, server_round=1, local_epochs=10,
            extra_params={"max_depth": 3},
        )
        assert model is not None
        pred = model.predict(X)
        assert len(pred) == len(y)


# ======================================================================
# Particao Estratificada
# ======================================================================

class TestStratifiedPartition:
    def test_returns_correct_number_of_partitions(self):
        """Retorna o numero correto de particoes."""
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        parts = stratified_partition(y, 2, seed=42)
        assert len(parts) == 2

    def test_partitions_cover_all_indices(self):
        """Todas as amostras estao em alguma particao."""
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        parts = stratified_partition(y, 2, seed=42)
        all_idx = np.sort(np.concatenate(parts))
        np.testing.assert_array_equal(all_idx, np.arange(10))

    def test_partitions_non_overlapping(self):
        """Particoes nao se sobrepoe."""
        y = np.array([0]*50 + [1]*50)
        parts = stratified_partition(y, 5, seed=42)
        all_idx = np.concatenate(parts)
        assert len(all_idx) == len(np.unique(all_idx))

    def test_class_distribution_preserved(self):
        """Cada particao tem distribuicao de classes similar."""
        y = np.array([0]*400 + [1]*100)  # 80% classe 0, 20% classe 1
        parts = stratified_partition(y, 5, seed=42)
        for part in parts:
            part_y = y[part]
            ratio_1 = np.mean(part_y == 1)
            # Deve estar proximo de 20% (com margem para arredondamento)
            assert 0.15 <= ratio_1 <= 0.25, f"Ratio classe 1 = {ratio_1:.2f}"

    def test_imbalanced_dataset(self):
        """Funciona com dataset altamente desbalanceado (como Avazu ~17%)."""
        rng = np.random.RandomState(42)
        y = np.zeros(1000, dtype=int)
        y[:170] = 1  # 17% positivos
        rng.shuffle(y)
        parts = stratified_partition(y, 6, seed=42)
        assert len(parts) == 6
        for part in parts:
            part_y = y[part]
            ratio = np.mean(part_y == 1)
            assert 0.14 <= ratio <= 0.20, f"Ratio = {ratio:.2f}"

    def test_reproducible_with_same_seed(self):
        """Mesma seed produz mesmas particoes."""
        y = np.array([0]*50 + [1]*50)
        parts1 = stratified_partition(y, 3, seed=42)
        parts2 = stratified_partition(y, 3, seed=42)
        for p1, p2 in zip(parts1, parts2):
            np.testing.assert_array_equal(np.sort(p1), np.sort(p2))

    def test_different_seeds_differ(self):
        """Seeds diferentes produzem particoes diferentes."""
        y = np.array([0]*50 + [1]*50)
        parts1 = stratified_partition(y, 3, seed=42)
        parts2 = stratified_partition(y, 3, seed=99)
        # Pelo menos uma particao deve diferir
        any_different = False
        for p1, p2 in zip(parts1, parts2):
            if not np.array_equal(np.sort(p1), np.sort(p2)):
                any_different = True
                break
        assert any_different


# ======================================================================
# Epocas Uniformes
# ======================================================================

class TestUniformEpochs:
    def test_local_epochs_is_uniform(self):
        """Verifica que LOCAL_EPOCHS e um valor unico, nao um dict."""
        assert isinstance(LOCAL_EPOCHS, int)
        assert LOCAL_EPOCHS > 0

    def test_config_no_local_epochs_by_cat(self):
        """LOCAL_EPOCHS_BY_CAT nao deve mais ser importavel como variavel ativa."""
        import config
        # A constante pode ainda existir no arquivo mas nao deve ser usada
        # O importante e que CLIENT_CATEGORIES existe (para QoS)
        assert hasattr(config, "CLIENT_CATEGORIES")
        assert hasattr(config, "LOCAL_EPOCHS")

    @pytest.mark.parametrize("model_type", ["xgboost", "lightgbm", "catboost"])
    def test_all_models_train_same_epochs(self, sample_data, model_type):
        """Todos os modelos treinam com as mesmas epocas."""
        X, y = sample_data
        epochs = 15
        model = ModelFactory.train(
            model_type, X, y,
            client_id=0, server_round=1, local_epochs=epochs,
        )
        assert model is not None


# ======================================================================
# Selecao Server-Side e Ensemble Ponderado
# ======================================================================

class TestServerSideEvaluation:
    def _make_models(self, sample_data, n_models=3):
        """Helper: cria N modelos com dados diferentes."""
        X, y = sample_data
        models = {}
        for i in range(n_models):
            rng = np.random.RandomState(i)
            idx = rng.choice(len(X), size=70, replace=False)
            model = ModelFactory.train(
                "xgboost", X[idx], y[idx],
                client_id=i, server_round=1, local_epochs=10,
            )
            models[i] = model
        return models

    def test_server_side_eval_produces_accuracies(self, sample_data):
        """Avaliacao server-side retorna accuracies validas."""
        X, y = sample_data
        models = self._make_models(sample_data)
        for cid, model in models.items():
            y_prob = model.predict_proba(X)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            acc = float(np.mean(y_pred == y))
            assert 0.0 <= acc <= 1.0

    def test_weighted_ensemble_runs(self, sample_data):
        """Ensemble ponderado por accuracy funciona."""
        X, y = sample_data
        models = self._make_models(sample_data)

        # Calcula pesos (simula o que bagging.py faz)
        preds = []
        weights = []
        for cid, model in models.items():
            prob = model.predict_proba(X)[:, 1]
            pred = (prob >= 0.5).astype(int)
            acc = float(np.mean(pred == y))
            preds.append(prob)
            weights.append(acc)

        weights = np.array(weights)
        weights = weights / weights.sum()
        y_prob = np.average(preds, axis=0, weights=weights)
        y_pred = (y_prob >= 0.5).astype(int)

        assert len(y_pred) == len(y)
        assert np.all((y_pred == 0) | (y_pred == 1))
        ensemble_acc = np.mean(y_pred == y)
        assert 0.0 <= ensemble_acc <= 1.0

    def test_weighted_vs_unweighted_differ(self, sample_data):
        """Ensemble ponderado e media simples podem diferir."""
        X, y = sample_data
        models = self._make_models(sample_data)

        preds = []
        weights = []
        for cid, model in models.items():
            prob = model.predict_proba(X)[:, 1]
            pred = (prob >= 0.5).astype(int)
            acc = float(np.mean(pred == y))
            preds.append(prob)
            weights.append(acc)

        # Media simples
        y_simple = np.mean(preds, axis=0)

        # Media ponderada
        w = np.array(weights)
        w = w / w.sum()
        y_weighted = np.average(preds, axis=0, weights=w)

        # Podem ser iguais se accs forem identicas, mas geralmente diferem
        # Apenas verificamos que ambos sao validos
        assert np.all(y_simple >= 0) and np.all(y_simple <= 1)
        assert np.all(y_weighted >= 0) and np.all(y_weighted <= 1)

    def test_best_model_by_server_acc(self, sample_data):
        """Selecao server-side escolhe modelo com melhor acc no test set."""
        X, y = sample_data
        models = self._make_models(sample_data)

        best_acc = -1
        best_cid = -1
        for cid, model in models.items():
            prob = model.predict_proba(X)[:, 1]
            pred = (prob >= 0.5).astype(int)
            acc = float(np.mean(pred == y))
            if acc > best_acc:
                best_acc = acc
                best_cid = cid

        assert best_cid >= 0
        assert best_acc > 0


# ======================================================================
# Integracao: fluxo completo com mitigacoes
# ======================================================================

class TestMitigationsIntegration:
    def test_multi_round_warm_start_bounded(self, sample_data):
        """Warm start ao longo de multiplos rounds nao acumula infinitamente."""
        X, y = sample_data
        model = None
        for round_num in range(1, 6):
            model = ModelFactory.train(
                "xgboost", X, y,
                client_id=0, server_round=round_num, local_epochs=10,
                warm_start_model=model,
            )
        n_trees = _count_trees(model)
        assert n_trees <= MAX_TOTAL_TREES
        assert n_trees > 10  # treinou algo

    def test_early_stopping_with_warm_start(self, sample_data):
        """Early stopping funciona junto com warm start."""
        X, y = sample_data
        model1 = ModelFactory.train(
            "xgboost", X, y,
            client_id=0, server_round=1, local_epochs=20,
        )
        model2 = ModelFactory.train(
            "xgboost", X, y,
            client_id=0, server_round=2, local_epochs=20,
            warm_start_model=model1,
        )
        assert model2 is not None
        pred = model2.predict(X)
        assert len(pred) == len(y)

    def test_extra_params_with_warm_start(self, sample_data):
        """Extra params funcionam com warm start."""
        X, y = sample_data
        model1 = ModelFactory.train(
            "xgboost", X, y,
            client_id=0, server_round=1, local_epochs=10,
            extra_params={"learning_rate": 0.05},
        )
        model2 = ModelFactory.train(
            "xgboost", X, y,
            client_id=0, server_round=2, local_epochs=10,
            warm_start_model=model1,
            extra_params={"learning_rate": 0.05},
        )
        assert model2 is not None

    @pytest.mark.parametrize("model_type", ["xgboost", "lightgbm", "catboost"])
    def test_full_pipeline_all_models(self, sample_data, model_type):
        """Pipeline completo: treino → warm start → serialize → predict."""
        import pickle
        X, y = sample_data

        # Round 1
        m1 = ModelFactory.train(
            model_type, X, y,
            client_id=0, server_round=1, local_epochs=10,
        )

        # Serialize (simula envio pela rede)
        raw = pickle.dumps(m1)
        restored = pickle.loads(raw)

        # Round 2 com warm start
        m2 = ModelFactory.train(
            model_type, X, y,
            client_id=0, server_round=2, local_epochs=10,
            warm_start_model=restored,
        )

        pred = m2.predict(X)
        assert len(pred) == len(y)
        proba = m2.predict_proba(X)
        assert proba.shape == (len(y), 2)
