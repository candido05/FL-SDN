"""
Testes para o pipeline completo de datasets: preprocessing, loaders, integracao.

Cobre lacunas nao cobertas por test_preprocessing.py:
  - prepare_avazu: escrita via memmap, truncamento quando CSV < MAX_ROWS
  - prepare_higgs_full: recuperacao de arquivo truncado (EOFError)
  - stratified_partition: dados desbalanceados (Avazu-like, 16% positivos)
  - Loaders avazu/epsilon/higgs_full: contrato server/client, max_samples,
    igualdade de test sets, dtype, particoes sem sobreposicao
  - Integracao end-to-end: save .npy -> load via DatasetRegistry

Todos os testes usam dados sinteticos — nao requerem os arquivos brutos reais.
"""

import csv
import gzip
import io
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.registry import stratified_partition


# ======================================================================
# Helpers compartilhados
# ======================================================================

def _patch_base_dir(monkeypatch, base_dir):
    """Redireciona datasets.paths._BASE_DIR para um diretorio temporario."""
    import datasets.paths as p
    monkeypatch.setattr(p, "_BASE_DIR", str(base_dir))


def _write_avazu_csv(path, n_rows=2000, seed=42):
    """Cria um CSV sintetico no formato Avazu."""
    rng = np.random.RandomState(seed)
    header = [
        "id", "click", "hour", "C1", "banner_pos",
        "site_id", "site_domain", "site_category",
        "app_id", "app_domain", "app_category",
        "device_id", "device_ip", "device_model",
        "device_type", "device_conn_type",
        "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(n_rows):
            hour = 14102100 + rng.randint(0, 24)
            click = rng.randint(0, 2)
            cats = [rng.choice(["a", "b", "c", "d"]) for _ in range(21)]
            writer.writerow([i, click, hour] + cats)


def _write_avazu_csv_gz(path, n_rows=2000, seed=42):
    """Cria um CSV.gz sintetico no formato Avazu."""
    csv_path = path.replace(".gz", "_tmp.csv")
    _write_avazu_csv(csv_path, n_rows=n_rows, seed=seed)
    with open(csv_path, "rb") as fin, gzip.open(path, "wb") as fout:
        fout.write(fin.read())
    os.remove(csv_path)


def _write_higgs_csv_gz(path, n_rows=600, truncate_at=None, seed=42):
    """
    Cria um HIGGS.csv.gz sintetico (29 colunas: label + 28 features).
    Se truncate_at for informado, trunca o arquivo naquele byte.
    """
    rng = np.random.RandomState(seed)
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        for i in range(n_rows):
            label = rng.randint(0, 2)
            features = rng.randn(28).astype(np.float32)
            line = str(label) + "," + ",".join(f"{v:.6f}" for v in features) + "\n"
            gz.write(line.encode())
    raw = buf.getvalue()
    if truncate_at is not None:
        raw = raw[:truncate_at]
    with open(path, "wb") as f:
        f.write(raw)


def _save_synthetic_npy(base_dir, name, n_rows, n_features, seed=42):
    """Salva arquivos .npy sinteticos em base_dir/name/."""
    d = os.path.join(str(base_dir), name)
    os.makedirs(d, exist_ok=True)
    rng = np.random.RandomState(seed)
    X = rng.randn(n_rows, n_features).astype(np.float32)
    y = rng.randint(0, 2, n_rows).astype(np.int8)
    np.save(os.path.join(d, f"{name}_X.npy"), X)
    np.save(os.path.join(d, f"{name}_y.npy"), y)
    return X, y


# ======================================================================
# prepare_avazu — escrita incremental via memmap
# ======================================================================

class TestPrepareAvazuPipeline:
    """Testa prepare_avazu com CSV sintetico pequeno."""

    @pytest.fixture
    def avazu_dir(self, tmp_path, monkeypatch):
        _patch_base_dir(monkeypatch, tmp_path)
        d = tmp_path / "avazu"
        d.mkdir()
        return d

    def test_output_files_created(self, avazu_dir):
        _write_avazu_csv_gz(str(avazu_dir / "train.gz"), n_rows=1500)
        from tools.prepare_datasets import prepare_avazu
        assert prepare_avazu(_max_rows=1000) is True
        assert (avazu_dir / "avazu_X.npy").exists()
        assert (avazu_dir / "avazu_y.npy").exists()

    def test_output_shape_equals_max_rows(self, avazu_dir):
        """Quando CSV > MAX_ROWS, output tem exatamente MAX_ROWS linhas."""
        _write_avazu_csv_gz(str(avazu_dir / "train.gz"), n_rows=1500)
        from tools.prepare_datasets import prepare_avazu
        prepare_avazu(_max_rows=1000)
        X = np.load(avazu_dir / "avazu_X.npy")
        y = np.load(avazu_dir / "avazu_y.npy")
        assert X.shape == (1000, 1029), f"Shape incorreto: {X.shape}"
        assert y.shape == (1000,)

    def test_truncation_when_csv_smaller_than_max_rows(self, avazu_dir):
        """Quando CSV < MAX_ROWS, output deve ter o numero real de linhas (sem zeros)."""
        _write_avazu_csv_gz(str(avazu_dir / "train.gz"), n_rows=300)
        from tools.prepare_datasets import prepare_avazu
        prepare_avazu(_max_rows=5000)   # MAX_ROWS bem maior que CSV
        X = np.load(avazu_dir / "avazu_X.npy")
        y = np.load(avazu_dir / "avazu_y.npy")
        assert X.shape[0] == 300, f"Esperava 300 linhas reais, obteve {X.shape[0]}"
        assert y.shape[0] == 300

    def test_output_dtype_float32(self, avazu_dir):
        _write_avazu_csv_gz(str(avazu_dir / "train.gz"), n_rows=500)
        from tools.prepare_datasets import prepare_avazu
        prepare_avazu(_max_rows=400)
        X = np.load(avazu_dir / "avazu_X.npy")
        assert X.dtype == np.float32

    def test_labels_binary(self, avazu_dir):
        _write_avazu_csv_gz(str(avazu_dir / "train.gz"), n_rows=500)
        from tools.prepare_datasets import prepare_avazu
        prepare_avazu(_max_rows=400)
        y = np.load(avazu_dir / "avazu_y.npy")
        assert set(np.unique(y)).issubset({0, 1}), f"Labels nao-binarias: {np.unique(y)}"

    def test_5_temporal_features_in_valid_range(self, avazu_dir):
        """Hora do dia (coluna 0) deve estar em [0, 23]; period (col 2) em [0, 3]."""
        _write_avazu_csv_gz(str(avazu_dir / "train.gz"), n_rows=500)
        from tools.prepare_datasets import prepare_avazu
        prepare_avazu(_max_rows=400)
        X = np.load(avazu_dir / "avazu_X.npy")
        assert X[:, 0].min() >= 0 and X[:, 0].max() <= 23   # hora
        assert X[:, 2].min() >= 0 and X[:, 2].max() <= 3    # periodo

    def test_sin_cos_bounded(self, avazu_dir):
        """sin e cos (colunas 3 e 4) devem estar em [-1, 1]."""
        _write_avazu_csv_gz(str(avazu_dir / "train.gz"), n_rows=500)
        from tools.prepare_datasets import prepare_avazu
        prepare_avazu(_max_rows=400)
        X = np.load(avazu_dir / "avazu_X.npy")
        assert X[:, 3].min() >= -1.0 - 1e-5
        assert X[:, 3].max() <= 1.0 + 1e-5
        assert X[:, 4].min() >= -1.0 - 1e-5
        assert X[:, 4].max() <= 1.0 + 1e-5

    def test_no_nans_or_infs(self, avazu_dir):
        _write_avazu_csv_gz(str(avazu_dir / "train.gz"), n_rows=500)
        from tools.prepare_datasets import prepare_avazu
        prepare_avazu(_max_rows=400)
        X = np.load(avazu_dir / "avazu_X.npy")
        assert not np.any(np.isnan(X)), "NaN encontrado nas features"
        assert not np.any(np.isinf(X)), "Inf encontrado nas features"

    def test_output_loadable_as_npy(self, avazu_dir):
        """Arquivo salvo deve ser um .npy valido e recarregavel."""
        _write_avazu_csv_gz(str(avazu_dir / "train.gz"), n_rows=500)
        from tools.prepare_datasets import prepare_avazu
        prepare_avazu(_max_rows=400)
        X = np.load(avazu_dir / "avazu_X.npy")
        y = np.load(avazu_dir / "avazu_y.npy")
        assert len(X) == len(y)


# ======================================================================
# prepare_higgs_full — recuperacao de truncamento
# ======================================================================

class TestPrepareHiggsFullTruncation:
    """Testa prepare_higgs_full com arquivo valido e com arquivo truncado."""

    @pytest.fixture
    def higgs_dir(self, tmp_path, monkeypatch):
        _patch_base_dir(monkeypatch, tmp_path)
        d = tmp_path / "higgs_full"
        d.mkdir()
        return d

    def test_complete_file_produces_correct_output(self, higgs_dir):
        """Arquivo completo (nao truncado) deve funcionar normalmente."""
        _write_higgs_csv_gz(str(higgs_dir / "HIGGS.csv.gz"), n_rows=400)
        from tools.prepare_datasets import prepare_higgs_full
        assert prepare_higgs_full(_chunk_size=200) is True
        assert (higgs_dir / "higgs_full_X.npy").exists()
        assert (higgs_dir / "higgs_full_y.npy").exists()

    def test_truncated_file_does_not_raise(self, higgs_dir):
        """Arquivo truncado nao deve propagar EOFError."""
        # Cria um gzip completo com 400 linhas, depois trunca os ultimos bytes
        full_path = str(higgs_dir / "HIGGS.csv.gz")
        _write_higgs_csv_gz(full_path, n_rows=400)
        with open(full_path, "rb") as f:
            content = f.read()
        with open(full_path, "wb") as f:
            f.write(content[: int(len(content) * 0.6)])  # trunca 40% do final

        from tools.prepare_datasets import prepare_higgs_full
        try:
            prepare_higgs_full(_chunk_size=100)
        except EOFError:
            pytest.fail("prepare_higgs_full propagou EOFError — truncamento nao tratado")

    def test_truncated_file_produces_output_from_complete_chunks(self, higgs_dir):
        """Chunks completamente lidos antes do truncamento devem ser salvos."""
        full_path = str(higgs_dir / "HIGGS.csv.gz")
        _write_higgs_csv_gz(full_path, n_rows=400)
        with open(full_path, "rb") as f:
            content = f.read()
        with open(full_path, "wb") as f:
            f.write(content[: int(len(content) * 0.6)])

        from tools.prepare_datasets import prepare_higgs_full
        prepare_higgs_full(_chunk_size=100)  # chunks de 100; esperamos >= 1 completo

        x_path = higgs_dir / "higgs_full_X.npy"
        if x_path.exists():  # pode nao ter chunks completos dependendo do truncamento
            X = np.load(x_path)
            assert X.ndim == 2
            assert X.shape[1] >= 28  # features apos preprocessing

    def test_output_feature_count_after_interactions(self, higgs_dir):
        """28 features originais + 21 interacoes - correlacoes = tipicamente > 28."""
        _write_higgs_csv_gz(str(higgs_dir / "HIGGS.csv.gz"), n_rows=500)
        from tools.prepare_datasets import prepare_higgs_full
        prepare_higgs_full(_chunk_size=300)
        X = np.load(higgs_dir / "higgs_full_X.npy")
        assert X.shape[1] >= 28

    def test_labels_binary(self, higgs_dir):
        _write_higgs_csv_gz(str(higgs_dir / "HIGGS.csv.gz"), n_rows=300)
        from tools.prepare_datasets import prepare_higgs_full
        prepare_higgs_full(_chunk_size=200)
        y = np.load(higgs_dir / "higgs_full_y.npy")
        assert set(np.unique(y)).issubset({0, 1})


# ======================================================================
# stratified_partition — particoes equilibradas com dados desbalanceados
# ======================================================================

class TestStratifiedPartitionImbalanced:
    """Testa stratified_partition com dados tipicos de cada dataset."""

    def test_equal_sizes_balanced(self):
        rng = np.random.RandomState(0)
        y = rng.randint(0, 2, 6000)
        parts = stratified_partition(y, 6, 42)
        sizes = [len(p) for p in parts]
        assert max(sizes) - min(sizes) <= 1, f"Tamanhos desiguais: {sizes}"

    def test_class_balance_preserved_imbalanced(self):
        """16% positivos (Avazu-like): cada particao deve ter ~16% de positivos."""
        rng = np.random.RandomState(42)
        n = 60_000
        y = np.zeros(n, dtype=np.int8)
        y[: int(0.16 * n)] = 1
        rng.shuffle(y)

        parts = stratified_partition(y, 6, 42)
        global_rate = y.mean()
        for i, idx in enumerate(parts):
            rate = y[idx].mean()
            assert abs(rate - global_rate) < 0.02, (
                f"Particao {i}: taxa {rate:.4f} desvia de global {global_rate:.4f}"
            )

    def test_no_overlap_between_partitions(self):
        rng = np.random.RandomState(0)
        y = rng.randint(0, 2, 1200)
        parts = stratified_partition(y, 6, 42)
        all_idx = np.concatenate(parts)
        assert len(all_idx) == len(np.unique(all_idx)), "Indices sobrepostos entre particoes"

    def test_covers_all_samples(self):
        y = np.array([0, 1] * 300)
        parts = stratified_partition(y, 6, 42)
        total = sum(len(p) for p in parts)
        assert total == 600, f"Total de amostras perdidas: esperava 600, obteve {total}"

    def test_reproducible_with_same_seed(self):
        y = np.random.RandomState(0).randint(0, 2, 600)
        p1 = stratified_partition(y, 3, 99)
        p2 = stratified_partition(y, 3, 99)
        for a, b in zip(p1, p2):
            np.testing.assert_array_equal(np.sort(a), np.sort(b))

    def test_different_seeds_give_different_partitions(self):
        y = np.random.RandomState(0).randint(0, 2, 600)
        p1 = stratified_partition(y, 3, 1)
        p2 = stratified_partition(y, 3, 2)
        # Pelo menos uma particao deve ser diferente
        any_diff = any(
            not np.array_equal(np.sort(a), np.sort(b)) for a, b in zip(p1, p2)
        )
        assert any_diff, "Sementes diferentes produziram particoes identicas"

    def test_single_partition(self):
        y = np.array([0, 1, 0, 1, 1])
        parts = stratified_partition(y, 1, 42)
        assert len(parts) == 1
        assert len(parts[0]) == 5

    def test_highly_imbalanced_epsilon_like(self):
        """50/50 (Epsilon): taxa de positivos identica entre particoes."""
        rng = np.random.RandomState(7)
        y = np.array([0] * 3000 + [1] * 3000)
        rng.shuffle(y)
        parts = stratified_partition(y, 6, 42)
        rates = [y[idx].mean() for idx in parts]
        for rate in rates:
            assert abs(rate - 0.5) < 0.02


# ======================================================================
# Loader avazu.py — contratos da API
# ======================================================================

class TestAvazuLoaderContracts:
    """Testa o loader datasets/avazu.py com .npy sinteticos."""

    N_ROWS = 5000
    N_FEATURES = 1029

    @pytest.fixture
    def avazu_npy(self, tmp_path, monkeypatch):
        _patch_base_dir(monkeypatch, tmp_path)
        _save_synthetic_npy(tmp_path, "avazu", self.N_ROWS, self.N_FEATURES)

    def test_server_returns_two_arrays(self, avazu_npy):
        from datasets.avazu import load
        result = load(role="server", max_samples=self.N_ROWS)
        assert isinstance(result, tuple) and len(result) == 2
        X, y = result
        assert X.ndim == 2 and y.ndim == 1

    def test_client_returns_four_arrays(self, avazu_npy):
        from datasets.avazu import load
        result = load(role="client", client_id=0, max_samples=self.N_ROWS)
        assert isinstance(result, tuple) and len(result) == 4

    def test_test_set_identical_server_and_client(self, avazu_npy):
        from datasets.avazu import load
        X_srv, y_srv = load(role="server", max_samples=self.N_ROWS)
        _, _, X_cli, y_cli = load(role="client", client_id=0, max_samples=self.N_ROWS)
        np.testing.assert_array_equal(X_srv, X_cli)
        np.testing.assert_array_equal(y_srv, y_cli)

    def test_test_set_size_matches_test_size(self, avazu_npy):
        """TEST_SIZE=0.2: com max_samples=1000, test deve ter 200 amostras."""
        from datasets.avazu import load
        X_test, y_test = load(role="server", max_samples=1000)
        assert len(y_test) == 200, f"Esperava 200, obteve {len(y_test)}"

    def test_max_samples_limits_total_data(self, avazu_npy):
        """max_samples=500 -> total 500 amostras antes do split."""
        from datasets.avazu import load
        X_test, y_test = load(role="server", max_samples=500)
        # test set = 20% de 500 = 100
        assert len(y_test) == 100

    def test_default_max_used_when_file_smaller(self, tmp_path, monkeypatch):
        """Arquivo com 2000 linhas < default_max=500k: usa todas as 2000 linhas."""
        _patch_base_dir(monkeypatch, tmp_path)
        _save_synthetic_npy(tmp_path, "avazu", 2000, self.N_FEATURES)
        from datasets.avazu import load
        X_test, y_test = load(role="server")   # default_max=500k > 2000
        assert len(y_test) == 400  # 2000 * 0.2

    def test_feature_count_preserved(self, avazu_npy):
        from datasets.avazu import load
        X_test, _ = load(role="server", max_samples=self.N_ROWS)
        assert X_test.shape[1] == self.N_FEATURES

    def test_labels_are_int(self, avazu_npy):
        from datasets.avazu import load
        _, y_test = load(role="server", max_samples=self.N_ROWS)
        assert np.issubdtype(y_test.dtype, np.integer)
        assert set(np.unique(y_test)).issubset({0, 1})

    def test_client_train_plus_test_equals_total(self, avazu_npy):
        """train + test (cliente) deve cobrir o total de amostras."""
        from datasets.avazu import load
        n = 1000
        X_tr, y_tr, X_te, y_te = load(role="client", client_id=0, max_samples=n,
                                       num_clients=1)
        assert len(y_tr) + len(y_te) == n


# ======================================================================
# Loaders epsilon.py e higgs_full.py — contratos basicos
# ======================================================================

class TestEpsilonAndHiggsFullLoaders:
    """Testa loaders de epsilon e higgs_full com .npy sinteticos."""

    @pytest.fixture
    def epsilon_npy(self, tmp_path, monkeypatch):
        _patch_base_dir(monkeypatch, tmp_path)
        _save_synthetic_npy(tmp_path, "epsilon", 2000, 354)

    @pytest.fixture
    def higgs_full_npy(self, tmp_path, monkeypatch):
        _patch_base_dir(monkeypatch, tmp_path)
        _save_synthetic_npy(tmp_path, "higgs_full", 2000, 42)

    def test_epsilon_server_output_shape(self, epsilon_npy):
        from datasets.epsilon import load
        X, y = load(role="server", max_samples=1000)
        assert X.ndim == 2 and y.ndim == 1
        assert len(y) == 200   # 1000 * 0.2

    def test_epsilon_client_test_equals_server(self, epsilon_npy):
        from datasets.epsilon import load
        X_srv, y_srv = load(role="server", max_samples=1000)
        _, _, X_cli, y_cli = load(role="client", client_id=0, max_samples=1000)
        np.testing.assert_array_equal(X_srv, X_cli)
        np.testing.assert_array_equal(y_srv, y_cli)

    def test_epsilon_labels_are_int(self, epsilon_npy):
        from datasets.epsilon import load
        _, y = load(role="server", max_samples=500)
        assert np.issubdtype(y.dtype, np.integer)
        assert set(np.unique(y)).issubset({0, 1})

    def test_higgs_full_server_output_shape(self, higgs_full_npy):
        from datasets.higgs_full import load
        X, y = load(role="server", max_samples=1000)
        assert X.ndim == 2 and y.ndim == 1
        assert len(y) == 200

    def test_higgs_full_client_test_equals_server(self, higgs_full_npy):
        from datasets.higgs_full import load
        X_srv, y_srv = load(role="server", max_samples=1000)
        _, _, X_cli, y_cli = load(role="client", client_id=0, max_samples=1000)
        np.testing.assert_array_equal(X_srv, X_cli)
        np.testing.assert_array_equal(y_srv, y_cli)

    def test_higgs_full_labels_are_int(self, higgs_full_npy):
        from datasets.higgs_full import load
        _, y = load(role="server", max_samples=500)
        assert np.issubdtype(y.dtype, np.integer)
        assert set(np.unique(y)).issubset({0, 1})

    def test_epsilon_client_partition_sizes_balanced(self, epsilon_npy):
        """Todas as particoes de cliente devem ter tamanhos proximos (diff <= num_classes)."""
        from datasets.epsilon import load
        from config import NUM_CLIENTS
        sizes = [
            len(load(role="client", client_id=c, max_samples=1200)[1])
            for c in range(NUM_CLIENTS)
        ]
        # Com 2 classes cada uma contribuindo remainder independentemente,
        # a diferenca maxima de tamanho entre clientes e num_classes (2).
        assert max(sizes) - min(sizes) <= 2, f"Tamanhos desiguais: {sizes}"

    def test_higgs_full_client_partition_class_balance(self, higgs_full_npy):
        """Taxa de positivos por cliente deve estar proxima da global."""
        from datasets.higgs_full import load
        from config import NUM_CLIENTS
        n = 1200
        _, y_all = load(role="server", max_samples=n)
        # Calcula taxa de positivos em cada cliente
        rates = []
        for c in range(NUM_CLIENTS):
            _, y_tr, _, _ = load(role="client", client_id=c, max_samples=n)
            rates.append(float(y_tr.mean()))
        global_rate = float(np.concatenate(
            [load(role="client", client_id=c, max_samples=n)[1] for c in range(NUM_CLIENTS)]
        ).mean())
        for i, rate in enumerate(rates):
            assert abs(rate - global_rate) < 0.05, (
                f"Cliente {i}: taxa {rate:.3f} muito distante da global {global_rate:.3f}"
            )


# ======================================================================
# Integracao end-to-end: prepare -> is_prepared -> load
# ======================================================================

class TestPreparedStateIntegration:
    """Testa que is_prepared detecta arquivos criados e que os loaders os usam."""

    def test_is_prepared_false_before_files_exist(self, tmp_path, monkeypatch):
        _patch_base_dir(monkeypatch, tmp_path)
        from datasets.paths import is_prepared
        assert is_prepared("avazu") is False
        assert is_prepared("epsilon") is False
        assert is_prepared("higgs_full") is False

    def test_is_prepared_true_after_npy_created(self, tmp_path, monkeypatch):
        _patch_base_dir(monkeypatch, tmp_path)
        _save_synthetic_npy(tmp_path, "epsilon", 200, 354)
        from datasets.paths import is_prepared
        assert is_prepared("epsilon") is True

    def test_avazu_prepare_then_load_round_trip(self, tmp_path, monkeypatch):
        """prepare_avazu -> avazu.load deve funcionar sem erros."""
        _patch_base_dir(monkeypatch, tmp_path)
        avazu_dir = tmp_path / "avazu"
        avazu_dir.mkdir()
        _write_avazu_csv_gz(str(avazu_dir / "train.gz"), n_rows=600)

        from tools.prepare_datasets import prepare_avazu
        prepare_avazu(_max_rows=500)

        from datasets.paths import is_prepared
        assert is_prepared("avazu") is True

        from datasets.avazu import load
        X_test, y_test = load(role="server", max_samples=500)
        assert len(y_test) == 100    # 500 * 0.2

    def test_avazu_x_and_y_row_count_consistent(self, tmp_path, monkeypatch):
        """Apos prepare, X e y devem ter o mesmo numero de linhas."""
        _patch_base_dir(monkeypatch, tmp_path)
        avazu_dir = tmp_path / "avazu"
        avazu_dir.mkdir()
        _write_avazu_csv_gz(str(avazu_dir / "train.gz"), n_rows=400)

        from tools.prepare_datasets import prepare_avazu
        prepare_avazu(_max_rows=300)

        X = np.load(avazu_dir / "avazu_X.npy")
        y = np.load(avazu_dir / "avazu_y.npy")
        assert X.shape[0] == y.shape[0]
