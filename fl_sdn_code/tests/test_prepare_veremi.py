"""Testes obrigatórios do M0 (seção 8 da especificação).

Rodar de dentro de fl_sdn_code/:
    python -m pytest tests/test_prepare_veremi.py -v

Todos usam CSVs sintéticos pequenos, criados em tmp_path. Nenhum depende do
CSV real de 1,2 GB.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Importa os módulos pelo caminho do arquivo. Assim os testes não dependem de
# tools/ ou datasets/ serem pacotes, nem colidem com a biblioteca "datasets".
# ---------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # fl_sdn_code/


def _load(name: str, relpath: str):
    path = os.path.join(ROOT, relpath)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Não foi possível carregar {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module   # necessário para @dataclass
    spec.loader.exec_module(module)
    return module


prep = _load("prepare_veremi", "tools/prepare_veremi.py")
reader = _load("veremi_prepared", "datasets/veremi_prepared.py")

# ---------------------------------------------------------------------------
# Utilitários: montar CSVs sintéticos com as 30 colunas do CSV real
# ---------------------------------------------------------------------------

ALL_COLUMNS = ["type", "sendTime", "sender", "senderPseudo", "messageID", "class"]
for _q in ["pos", "spd", "acl", "hed"]:
    ALL_COLUMNS += [_q + a for a in "xyz"] + [_q + a + "_n" for a in "xyz"]


def msg(sender, pseudo, t, cls=0, mid=None, pos=(0.0, 0.0), spd=(0.0, 0.0), acl=(0.0, 0.0)):
    """Uma mensagem (linha do CSV). Campos não informados ficam em 0."""
    return {
        "sender": sender, "senderPseudo": pseudo, "sendTime": t, "class": cls,
        "messageID": mid,
        "posx": pos[0], "posy": pos[1],
        "spdx": spd[0], "spdy": spd[1],
        "aclx": acl[0], "acly": acl[1],
    }


def write_csv(path, messages, shuffle_seed=None) -> str:
    rows = []
    for i, m in enumerate(messages):
        r = dict.fromkeys(ALL_COLUMNS, 0.0)
        r["type"] = 4
        r.update({k: v for k, v in m.items() if v is not None})
        if m.get("messageID") is None:
            r["messageID"] = i
        rows.append(r)
    df = pd.DataFrame(rows, columns=ALL_COLUMNS)
    if shuffle_seed is not None:
        df = df.sample(frac=1, random_state=shuffle_seed)
    df.to_csv(path, index=False)
    return str(path)


def random_messages(n_pseudos=6, per_pseudo=12, seed=0, cls_of=lambda s: 0):
    """Vários pseudônimos (um por veículo) com tempos e valores aleatórios."""
    rng = np.random.default_rng(seed)
    out = []
    # messageIDs embaralhados: não seguem a ordem do tempo.
    mids = iter(rng.permutation(n_pseudos * per_pseudo * 10)[: n_pseudos * per_pseudo])
    for k in range(n_pseudos):
        sender, pseudo = 100 + k, 1000 + k
        times = 72000 + np.cumsum(rng.choice([0.1, 0.5, 1.0, 2.0], size=per_pseudo))
        for t in times:
            out.append(msg(sender, pseudo, float(t), cls=cls_of(sender), mid=int(next(mids)),
                           pos=tuple(rng.normal(size=2) * 100),
                           spd=tuple(rng.normal(size=2) * 10),
                           acl=tuple(rng.normal(size=2))))
    return out


def run_prepare(tmp_path, csv_path, outname="out"):
    out = str(tmp_path / outname)
    manifest = prep.prepare(
        csv_path, out, prep.sha256_file(csv_path), expected_rows=None, verbose=False
    )
    return out, manifest


def feats(df_or_csv) -> pd.DataFrame:
    df = prep.read_csv(df_or_csv, expected_rows=None) if isinstance(df_or_csv, str) else df_or_csv
    return prep.compute_temporal_features(df)


# ---------------------------------------------------------------------------
# Teste 1 — Fórmulas (exemplo resolvido da seção 8.3)
# ---------------------------------------------------------------------------

def test_1_formulas_exemplo_8_3(tmp_path):
    messages = [
        # pseudônimo do exemplo: A, B, C, D
        # messageID em ordem INVERSA à do tempo: a ordenação tem que ser pelo
        # sendTime (o messageID só desempata).
        msg(7, 70, 10.0, mid=4, pos=(0, 0), spd=(10, 0), acl=(0, 0)),     # A
        msg(7, 70, 11.0, mid=3, pos=(10, 0), spd=(10, 0), acl=(2, 0)),    # B
        msg(7, 70, 12.0, mid=2, pos=(20, 3), spd=(12, 0), acl=(0, 0)),    # C
        msg(7, 70, 12.2, mid=1, pos=(22.4, 3), spd=(12, 0), acl=(0, 0)),  # D
        # segundo pseudônimo, 2 mensagens
        msg(9, 90, 50.0, mid=6, pos=(0, 0), spd=(1, 0)),                  # E
        msg(9, 90, 51.0, mid=5, pos=(1, 0), spd=(1, 0)),                  # F
    ]
    # Embaralha as linhas: o cálculo não pode depender da ordem do CSV.
    csv = write_csv(tmp_path / "ex.csv", messages, shuffle_seed=3)
    df = prep.read_csv(csv, expected_rows=None)
    result = prep.compute_temporal_features(df)
    result.index = df["messageID"].to_numpy()

    nan = np.nan
    expected = {  # dt_prev, rate_5s, pos_resid, spd_resid
        4: [nan, 0, nan, nan],    # A
        3: [1.0, 1, 0.0, 0.0],    # B
        2: [1.0, 2, 3.0, 0.0],    # C
        1: [0.2, 3, 0.0, 0.0],    # D
        6: [nan, 0, nan, nan],    # E
        5: [1.0, 1, 0.0, 0.0],    # F
    }
    for mid, values in expected.items():
        np.testing.assert_allclose(
            result.loc[mid, prep.TEMPORAL_FEATURES].to_numpy(dtype=float),
            values, atol=1e-5, err_msg=f"mensagem {mid}",
        )


# ---------------------------------------------------------------------------
# Teste 2 — Causalidade (truncamento)
# ---------------------------------------------------------------------------

def test_2_causalidade_truncamento(tmp_path):
    csv = write_csv(tmp_path / "c.csv", random_messages(n_pseudos=6, per_pseudo=12),
                    shuffle_seed=1)
    df = prep.read_csv(csv, expected_rows=None)
    assert len(df) >= 50

    full = prep.compute_temporal_features(df)

    T = float(np.median(df["sendTime"]))
    before = (df["sendTime"] < T).to_numpy()
    assert 0 < before.sum() < len(df)

    df_cut = df[before].reset_index(drop=True)
    cut = prep.compute_temporal_features(df_cut)

    pd.testing.assert_frame_equal(
        full[before].reset_index(drop=True), cut, check_exact=True
    )


# ---------------------------------------------------------------------------
# Teste 3 — Regras dos passos 3 e 6
# ---------------------------------------------------------------------------

def test_3a_veiculo_com_duas_classes_aborta(tmp_path):
    messages = [
        msg(1, 11, 1.0, cls=0),
        msg(1, 11, 2.0, cls=5),   # mesmo veículo, outra classe
        msg(2, 22, 1.0, cls=0),
    ]
    csv = write_csv(tmp_path / "a.csv", messages)
    with pytest.raises(prep.PreparationError, match="1 classe por veículo"):
        run_prepare(tmp_path, csv)


def test_3b_pseudonimo_compartilhado_aborta(tmp_path):
    messages = [
        msg(1, 55, 1.0),
        msg(2, 55, 2.0),          # pseudônimo 55 (≠ 1) usado por dois veículos
        msg(3, 33, 1.0),
    ]
    csv = write_csv(tmp_path / "b.csv", messages)
    with pytest.raises(prep.PreparationError, match="1 veículo por pseudônimo"):
        run_prepare(tmp_path, csv)


# ---------------------------------------------------------------------------
# Teste 4 — Exclusões (tipo 2 e pseudônimo 1)
# ---------------------------------------------------------------------------

def test_4_exclusoes(tmp_path):
    messages = [
        msg(1, 11, 1.0, cls=0), msg(1, 11, 2.0, cls=0),
        msg(2, 22, 1.0, cls=2), msg(2, 22, 2.0, cls=2), msg(2, 22, 3.0, cls=2),  # tipo 2
        msg(3, 1, 1.0, cls=16), msg(3, 33, 1.5, cls=16), msg(3, 1, 2.0, cls=16),
        msg(3, 33, 2.5, cls=16),                                                 # usa 1 e 33
        msg(4, 1, 1.2, cls=16), msg(4, 44, 1.7, cls=16),                         # usa 1 e 44
    ]
    csv = write_csv(tmp_path / "e.csv", messages)
    out, manifest = run_prepare(tmp_path, csv)
    data = reader.load_prepared(out)

    assert not (data.tipo == 2).any(), "sobrou linha de tipo 2"
    assert 2 not in data.sender, "veículo do tipo 2 não foi removido inteiro"
    # o veículo 3 continua, só com as linhas do pseudônimo 33
    assert (data.sender == 3).sum() == 2
    assert (data.sender == 4).sum() == 1
    assert len(data.y) == 2 + 2 + 1

    assert manifest["exclusions"]["attack_type_2"] == {"rows_removed": 3, "vehicles_removed": 1}
    assert manifest["exclusions"]["pseudonym_1"] == {"rows_removed": 3, "vehicles_affected": 2}


# ---------------------------------------------------------------------------
# Teste 5 — Determinismo
# ---------------------------------------------------------------------------

def test_5_determinismo(tmp_path):
    csv = write_csv(tmp_path / "d.csv",
                    random_messages(cls_of=lambda s: [0, 13, 16][s % 3]), shuffle_seed=2)
    _, m1 = run_prepare(tmp_path, csv, "run1")
    _, m2 = run_prepare(tmp_path, csv, "run2")
    assert set(m1["files"]) == {"X.npy", "y.npy", "tipo.npy", "familia.npy", "sender.npy"}
    for fname in m1["files"]:
        assert m1["files"][fname]["sha256"] == m2["files"][fname]["sha256"], fname


# ---------------------------------------------------------------------------
# Teste 6 — Formato da saída
# ---------------------------------------------------------------------------

def test_6_formato_da_saida(tmp_path):
    messages = random_messages(cls_of=lambda s: [0, 5, 13][s % 3])
    csv = write_csv(tmp_path / "f.csv", messages)
    out, manifest = run_prepare(tmp_path, csv)
    n = len(messages)

    expected = {
        "X.npy": ((n, 20), np.float32),
        "y.npy": ((n,), np.int8),
        "tipo.npy": ((n,), np.int8),
        "familia.npy": ((n,), np.int8),
        "sender.npy": ((n,), np.int64),
    }
    for fname, (shape, dtype) in expected.items():
        arr = np.load(os.path.join(out, fname))
        assert arr.shape == shape, fname
        assert arr.dtype == dtype, fname
        assert manifest["files"][fname]["shape"] == list(shape)
        assert manifest["files"][fname]["dtype"] == np.dtype(dtype).name

    assert manifest["feature_names"] == [
        "posx", "posy", "spdx", "spdy", "aclx", "acly", "hedx", "hedy",
        "posx_n", "posy_n", "spdx_n", "spdy_n", "aclx_n", "acly_n", "hedx_n", "hedy_n",
        "dt_prev", "rate_5s", "pos_resid", "spd_resid",
    ]
    forbidden = {"sender", "senderPseudo", "sendTime", "messageID", "class"}
    assert not forbidden & set(manifest["feature_names"])

    # X tem exatamente o conteúdo do CSV nas colunas 0-15, na ordem do CSV,
    # e nenhuma coluna igual a um identificador ou ao tempo.
    X = np.load(os.path.join(out, "X.npy"))
    src = prep.read_csv(csv, expected_rows=None)
    for j, c in enumerate(prep.CONTENT_FEATURES):
        np.testing.assert_array_equal(X[:, j], src[c].to_numpy(dtype=np.float32), err_msg=c)
    for c in ["sender", "senderPseudo", "sendTime", "messageID"]:
        col = src[c].to_numpy(dtype=np.float32)
        assert not any(np.array_equal(X[:, j], col) for j in range(20)), c

    # Rótulos coerentes com a tabela de famílias.
    y = np.load(os.path.join(out, "y.npy"))
    tipo = np.load(os.path.join(out, "tipo.npy"))
    familia = np.load(os.path.join(out, "familia.npy"))
    np.testing.assert_array_equal(y, (tipo != 0).astype(np.int8))
    np.testing.assert_array_equal(familia, [prep.FAMILY_MAP[int(t)] for t in tipo])

    # Nada de .tmp sobrando, e a saída não pode ser sobrescrita.
    assert not os.path.exists(out + ".tmp")
    with pytest.raises(prep.PreparationError, match="já existe"):
        prep.prepare(csv, out, prep.sha256_file(csv), expected_rows=None, verbose=False)


def test_6b_hash_errado_aborta_sem_criar_saida(tmp_path):
    csv = write_csv(tmp_path / "h.csv", random_messages(n_pseudos=2, per_pseudo=3))
    out = str(tmp_path / "out")
    with pytest.raises(prep.PreparationError, match="SHA-256"):
        prep.prepare(csv, out, "0" * 64, expected_rows=None, verbose=False)
    assert not os.path.exists(out)
    assert not os.path.exists(out + ".tmp")


# ---------------------------------------------------------------------------
# Teste 7 — Leitor
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_output(tmp_path):
    csv = write_csv(tmp_path / "r.csv", random_messages(cls_of=lambda s: [0, 13][s % 2]))
    out, _ = run_prepare(tmp_path, csv)
    return out


def _copy(src, tmp_path, name):
    dst = str(tmp_path / name)
    shutil.copytree(src, dst)
    return dst


def _edit_manifest(path, fn):
    mpath = os.path.join(path, "manifest.json")
    with open(mpath, encoding="utf-8") as f:
        manifest = json.load(f)
    fn(manifest)
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True, ensure_ascii=False)


def test_7a_leitor_aceita_saida_valida(valid_output):
    for mmap in (True, False):
        data = reader.load_prepared(valid_output, mmap=mmap)
        assert data.X.shape[1] == 20
        assert len(data.feature_names) == 20
        assert data.X.shape[0] == data.y.shape[0] == data.sender.shape[0]
    # com mmap, os arrays são somente leitura
    data = reader.load_prepared(valid_output, mmap=True)
    with pytest.raises((ValueError, OSError)):
        data.X[0, 0] = 1.0


def test_7b_leitor_rejeita_byte_alterado(valid_output, tmp_path):
    bad = _copy(valid_output, tmp_path, "byte")
    fpath = os.path.join(bad, "y.npy")
    with open(fpath, "r+b") as f:
        f.seek(-1, os.SEEK_END)
        b = f.read(1)
        f.seek(-1, os.SEEK_END)
        f.write(bytes([b[0] ^ 0xFF]))
    with pytest.raises(reader.PreparedDataError, match="y.npy"):
        reader.load_prepared(bad)


def test_7c_leitor_rejeita_schema_desconhecido(valid_output, tmp_path):
    bad = _copy(valid_output, tmp_path, "schema")
    _edit_manifest(bad, lambda m: m.update(schema_version="9.9"))
    with pytest.raises(reader.PreparedDataError, match="schema_version"):
        reader.load_prepared(bad)


def test_7d_leitor_rejeita_n_diferente(valid_output, tmp_path):
    bad = _copy(valid_output, tmp_path, "n")
    fpath = os.path.join(bad, "tipo.npy")
    tipo = np.load(fpath)
    np.save(fpath, tipo[:-1])   # uma linha a menos
    # Atualiza o hash e a forma no manifesto, para que só a conferência de N
    # possa pegar o problema (senão o hash pegaria antes).
    sha = reader._sha256_file(fpath)

    def fix(m):
        m["files"]["tipo.npy"]["sha256"] = sha
        m["files"]["tipo.npy"]["shape"] = [len(tipo) - 1]

    _edit_manifest(bad, fix)
    with pytest.raises(reader.PreparedDataError, match="desalinhados"):
        reader.load_prepared(bad)


def test_7e_leitor_rejeita_features_em_outra_ordem(valid_output, tmp_path):
    bad = _copy(valid_output, tmp_path, "ordem")

    def swap(m):
        names = m["feature_names"]
        assert names[18:20] == ["pos_resid", "spd_resid"]
        names[18], names[19] = names[19], names[18]

    # Só o manifesto muda: os hashes dos .npy continuam batendo, então só a
    # conferência de nomes/ordem pode pegar o problema.
    _edit_manifest(bad, swap)
    with pytest.raises(reader.PreparedDataError, match="feature_names"):
        reader.load_prepared(bad)
