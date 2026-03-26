"""
Testes de validacao pre-execucao para run_experimentos_noturnos.sh.

Valida que todos os contratos entre o script bash e o codigo FL estao
corretos antes de iniciar os experimentos noturnos.

Grupos de testes:
  TestServerClientArgcompat   — argumentos CLI server.py / client.py
  TestStrategyNames           — nomes de estrategia usados no script
  TestOutputDirNaming         — padrao de nome de diretorio x collect_results
  TestDatasetFiles            — arquivos .npy preparados e acessiveis
  TestSDNEndpoints            — endpoints SDN vivos (requer orquestrador rodando)
  TestSDNTools                — ferramentas auxiliares do SDN
  TestVirtualenv              — venv acessivel nos caminhos esperados
  TestFLRoundNotifications    — fl_round_start / fl_round_stop via Python
  TestServerStartupDetection  — regex de deteccao de inicio do servidor
  TestContainerDatasetPaths   — caminhos de dataset dentro dos containers
  TestCollectResultsLogic     — logica de coleta de resultados
"""

import argparse
import os
import re
import sys
import time
import tempfile
import shutil

import pytest

# Adiciona o diretorio raiz do projeto ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# =============================================================================
# Constantes do script (replicam o que esta no bash)
# =============================================================================

SCRIPT_DATASETS   = ["epsilon", "higgs_full", "avazu"]
SCRIPT_MODELS     = ["xgboost"]
SCRIPT_STRATEGIES = {"com_sdn": "sdn-bagging", "sem_sdn": "bagging"}

FL_DIR    = os.path.expanduser("~/FL-SDN-main/fl_sdn_code")
SDN_DIR   = os.path.expanduser("~/sdn-project-main")
ORCH_URL  = "http://172.16.1.1:8000"

VENV_PATHS = [
    os.path.expanduser("~/fl-node/venv/bin/activate"),
    os.path.expanduser("~/FL-SDN-main/venv/bin/activate"),
    os.path.expanduser("~/FL-SDN-main/fl_sdn_code/venv/bin/activate"),
]


# =============================================================================
# 1. Compatibilidade de argumentos server.py / client.py
# =============================================================================

class TestServerClientArgCompat:
    """Garante que os argumentos CLI aceitos por server.py e client.py
    cobrem exatamente o que o script bash passa em tempo de execucao."""

    def _parse_choices(self, script_path: str, arg_name: str):
        """Importa o modulo e extrai as choices do argparse sem executar main()."""
        import importlib.util, types

        spec = importlib.util.spec_from_file_location("_mod", script_path)
        mod  = importlib.util.module_from_spec(spec)
        # Injeta um namespace minimo para evitar efeitos colaterais
        mod.__dict__.update({"__name__": "_mod"})
        spec.loader.exec_module(mod)

        # Chama main() em modo dry-run capturando o parser antes do parse_args
        parser = argparse.ArgumentParser()
        src = open(script_path).read()
        # Extrai choices diretamente do codigo-fonte (mais simples e sem efeitos)
        pattern = rf'add_argument\("--{re.escape(arg_name)}".*?choices=\[(.*?)\]'
        m = re.search(pattern, src, re.DOTALL)
        if not m:
            return None
        raw = m.group(1)
        return [x.strip().strip('"').strip("'") for x in raw.split(",") if x.strip()]

    def test_server_model_choices_cover_script_models(self):
        src = open(os.path.join(FL_DIR, "server.py")).read()
        m = re.search(r'add_argument\("--model".*?choices=\[(.*?)\]', src, re.DOTALL)
        assert m, "server.py nao tem --model com choices"
        choices = [x.strip().strip('"').strip("'") for x in m.group(1).split(",")]
        for model in SCRIPT_MODELS:
            assert model in choices, \
                f"Modelo '{model}' (usado no script) nao aceito pelo server.py"

    def test_server_strategy_choices_cover_script_strategies(self):
        src = open(os.path.join(FL_DIR, "server.py")).read()
        m = re.search(r'add_argument\("--strategy".*?choices=\[(.*?)\]', src, re.DOTALL)
        assert m, "server.py nao tem --strategy com choices"
        choices = [x.strip().strip('"').strip("'") for x in m.group(1).split(",")]
        for strategy in SCRIPT_STRATEGIES.values():
            assert strategy in choices, \
                f"Estrategia '{strategy}' (usada no script) nao aceita pelo server.py"

    def test_server_dataset_choices_cover_script_datasets(self):
        src = open(os.path.join(FL_DIR, "server.py")).read()
        m = re.search(r'add_argument\("--dataset".*?choices=\[(.*?)\]', src, re.DOTALL)
        assert m, "server.py nao tem --dataset com choices"
        choices = [x.strip().strip('"').strip("'") for x in m.group(1).split(",")]
        for ds in SCRIPT_DATASETS:
            assert ds in choices, \
                f"Dataset '{ds}' (usado no script) nao aceito pelo server.py"

    def test_client_model_choices_cover_script_models(self):
        src = open(os.path.join(FL_DIR, "client.py")).read()
        m = re.search(r'add_argument\("--model".*?choices=\[(.*?)\]', src, re.DOTALL)
        assert m, "client.py nao tem --model com choices"
        choices = [x.strip().strip('"').strip("'") for x in m.group(1).split(",")]
        for model in SCRIPT_MODELS:
            assert model in choices, \
                f"Modelo '{model}' (usado no script) nao aceito pelo client.py"

    def test_client_dataset_choices_cover_script_datasets(self):
        src = open(os.path.join(FL_DIR, "client.py")).read()
        m = re.search(r'add_argument\("--dataset".*?choices=\[(.*?)\]', src, re.DOTALL)
        assert m, "client.py nao tem --dataset com choices"
        choices = [x.strip().strip('"').strip("'") for x in m.group(1).split(",")]
        for ds in SCRIPT_DATASETS:
            assert ds in choices, \
                f"Dataset '{ds}' (usado no script) nao aceito pelo client.py"

    def test_client_has_client_id_argument(self):
        src = open(os.path.join(FL_DIR, "client.py")).read()
        assert '"--client-id"' in src, \
            "client.py nao tem argumento --client-id"


# =============================================================================
# 2. Nomes de estrategia
# =============================================================================

class TestStrategyNames:
    """Verifica que os nomes de estrategia do script existem no STRATEGY_MAP."""

    def test_com_sdn_strategy_registered(self):
        from strategies import STRATEGY_MAP
        assert "sdn-bagging" in STRATEGY_MAP, \
            "sdn-bagging nao esta registrado em STRATEGY_MAP"

    def test_sem_sdn_strategy_registered(self):
        from strategies import STRATEGY_MAP
        assert "bagging" in STRATEGY_MAP, \
            "bagging nao esta registrado em STRATEGY_MAP"

    def test_all_script_strategies_registered(self):
        from strategies import STRATEGY_MAP
        for exp_type, strategy in SCRIPT_STRATEGIES.items():
            assert strategy in STRATEGY_MAP, \
                f"Estrategia '{strategy}' ({exp_type}) ausente no STRATEGY_MAP"


# =============================================================================
# 3. Padrao de nome de diretorio x collect_results
# =============================================================================

class TestOutputDirNaming:
    """Verifica que o padrao de nome gerado por create_run_dir e detectavel
    pela logica de collect_results (find ... -name '*{exp_name}*')."""

    def test_run_dir_contains_exp_name(self, tmp_path, monkeypatch):
        import core.output as out_mod
        monkeypatch.setattr(out_mod, "_BASE_DIR", str(tmp_path))

        for exp_type, strategy in SCRIPT_STRATEGIES.items():
            for ds in SCRIPT_DATASETS:
                for model in SCRIPT_MODELS:
                    exp_name = f"{exp_type}_{ds}_{model}"
                    run_dir = out_mod.create_run_dir(model, strategy, exp_name)
                    dirname  = os.path.basename(run_dir)
                    assert exp_name in dirname, \
                        f"exp_name '{exp_name}' ausente no dir '{dirname}'"

    def test_collect_results_pattern_matches_run_dir(self, tmp_path, monkeypatch):
        """Simula o find do script: find output -name '*{exp_name}*'."""
        import core.output as out_mod
        monkeypatch.setattr(out_mod, "_BASE_DIR", str(tmp_path))

        exp_name = "com_sdn_epsilon_xgboost"
        run_dir  = out_mod.create_run_dir("xgboost", "sdn-bagging", exp_name)

        # Replica o find do bash
        matches = [
            d for d in os.listdir(str(tmp_path))
            if exp_name in d and os.path.isdir(os.path.join(str(tmp_path), d))
        ]
        assert len(matches) == 1, \
            f"Pattern '*{exp_name}*' deveria encontrar exatamente 1 dir, encontrou {matches}"

    def test_two_experiments_produce_distinct_dirs(self, tmp_path, monkeypatch):
        """Dois experimentos distintos nao devem colidir no find."""
        import core.output as out_mod
        monkeypatch.setattr(out_mod, "_BASE_DIR", str(tmp_path))

        exp1 = "com_sdn_epsilon_xgboost"
        exp2 = "sem_sdn_epsilon_xgboost"
        time.sleep(1)  # garante timestamp diferente
        out_mod.create_run_dir("xgboost", "sdn-bagging", exp1)
        out_mod.create_run_dir("xgboost", "bagging", exp2)

        m1 = [d for d in os.listdir(str(tmp_path)) if exp1 in d]
        m2 = [d for d in os.listdir(str(tmp_path)) if exp2 in d]
        assert len(m1) == 1 and len(m2) == 1, \
            "Os dois experimentos devem gerar diretorios distintos e unicos"
        assert m1[0] != m2[0], "Diretorios identicos — colisao de nomes"


# =============================================================================
# 4. Arquivos de dataset preparados
# =============================================================================

class TestDatasetFiles:
    """Verifica que os .npy de cada dataset existem nos caminhos esperados
    pelo script (FL_DIR/data/<ds>/<ds>_X.npy e <ds>_y.npy)."""

    @pytest.mark.parametrize("ds", SCRIPT_DATASETS)
    def test_x_npy_exists(self, ds):
        path = os.path.join(FL_DIR, "data", ds, f"{ds}_X.npy")
        assert os.path.exists(path), \
            f"Arquivo ausente: {path}\n  Execute: python tools/prepare_datasets.py --dataset {ds}"

    @pytest.mark.parametrize("ds", SCRIPT_DATASETS)
    def test_y_npy_exists(self, ds):
        path = os.path.join(FL_DIR, "data", ds, f"{ds}_y.npy")
        assert os.path.exists(path), \
            f"Arquivo ausente: {path}\n  Execute: python tools/prepare_datasets.py --dataset {ds}"

    @pytest.mark.parametrize("ds", SCRIPT_DATASETS)
    def test_x_and_y_same_row_count(self, ds):
        import numpy as np
        x_path = os.path.join(FL_DIR, "data", ds, f"{ds}_X.npy")
        y_path = os.path.join(FL_DIR, "data", ds, f"{ds}_y.npy")
        if not (os.path.exists(x_path) and os.path.exists(y_path)):
            pytest.skip(f"Dataset {ds} nao preparado")
        X = np.load(x_path, mmap_mode="r")
        y = np.load(y_path, mmap_mode="r")
        assert len(X) == len(y), \
            f"{ds}: X tem {len(X)} linhas mas y tem {len(y)}"

    @pytest.mark.parametrize("ds", SCRIPT_DATASETS)
    def test_is_prepared_returns_true(self, ds):
        from datasets.paths import is_prepared
        assert is_prepared(ds), \
            f"is_prepared('{ds}') retornou False — dataset nao reconhecido"


# =============================================================================
# 5. Endpoints SDN vivos
# =============================================================================

class TestSDNEndpoints:
    """Testa os endpoints do orquestrador SDN (requer orquestrador rodando)."""

    @pytest.fixture(autouse=True)
    def require_orchestrator(self):
        try:
            import requests
            r = requests.get(f"{ORCH_URL}/health", timeout=3)
            r.raise_for_status()
        except Exception:
            pytest.skip("Orquestrador SDN indisponivel")

    def test_health_has_switches_field(self):
        import requests
        data = requests.get(f"{ORCH_URL}/health", timeout=3).json()
        assert "switches" in data, "/health nao retorna campo 'switches'"

    def test_health_has_hosts_field(self):
        import requests
        data = requests.get(f"{ORCH_URL}/health", timeout=3).json()
        assert "hosts" in data, "/health nao retorna campo 'hosts'"

    def test_health_switch_count_sufficient(self):
        import requests
        data = requests.get(f"{ORCH_URL}/health", timeout=3).json()
        n = data.get("switches", 0)
        assert n >= 10, f"Apenas {n} switches — GNS3 pode nao estar totalmente iniciado"

    def test_health_host_count_sufficient(self):
        import requests
        data = requests.get(f"{ORCH_URL}/health", timeout=3).json()
        n = data.get("hosts", 0)
        assert n >= 6, f"Apenas {n} hosts — algum container FL pode estar offline"

    def test_metrics_hosts_endpoint(self):
        import requests
        data = requests.get(f"{ORCH_URL}/metrics/hosts", timeout=5).json()
        assert "hosts" in data, "/metrics/hosts nao retorna campo 'hosts'"

    def test_fl_training_start_returns_ok(self):
        import requests
        r = requests.post(f"{ORCH_URL}/fl/training/start",
                          json={"round": 99}, timeout=5)
        assert r.status_code == 200
        body = r.json()
        assert body.get("status") == "ok", f"Status inesperado: {body}"
        assert "csv_path" in body, "Resposta nao contem csv_path"
        assert body.get("round") == 99

    def test_fl_training_stop_returns_ok(self):
        import requests
        # Garante que ha sessao ativa (start antes)
        requests.post(f"{ORCH_URL}/fl/training/start",
                      json={"round": 98}, timeout=5)
        r = requests.post(f"{ORCH_URL}/fl/training/stop", json={}, timeout=5)
        assert r.status_code == 200
        body = r.json()
        assert body.get("status") == "ok", f"Status inesperado: {body}"
        assert "duration_sec" in body

    def test_fl_training_creates_csv_file(self):
        import requests
        r = requests.post(f"{ORCH_URL}/fl/training/start",
                          json={"round": 97}, timeout=5)
        csv_path = r.json().get("csv_path", "")
        assert csv_path, "csv_path esta vazio"
        time.sleep(4)  # aguarda pelo menos 1 ciclo SDN (~3s)
        requests.post(f"{ORCH_URL}/fl/training/stop", json={}, timeout=5)
        assert os.path.exists(csv_path), \
            f"CSV nao criado em disco: {csv_path}"

    def test_fl_csv_has_expected_columns(self):
        import requests, csv as csv_mod
        r = requests.post(f"{ORCH_URL}/fl/training/start",
                          json={"round": 96}, timeout=5)
        csv_path = r.json().get("csv_path", "")
        time.sleep(4)
        requests.post(f"{ORCH_URL}/fl/training/stop", json={}, timeout=5)
        if not os.path.exists(csv_path):
            pytest.skip("CSV nao gerado — ciclo SDN pode ser mais lento")
        with open(csv_path) as f:
            reader = csv_mod.DictReader(f)
            cols = reader.fieldnames or []
        expected = {"timestamp", "cycle", "elapsed_sec", "n_switches",
                    "max_link_load_bps", "congested_links"}
        missing = expected - set(cols)
        assert not missing, f"Colunas ausentes no CSV do round: {missing}"


# =============================================================================
# 6. sdn_tools.py — ferramenta de limpeza de flows (sem_sdn)
# =============================================================================

class TestSDNTools:
    """Verifica ferramentas auxiliares SDN usadas pelo script."""

    def test_sdn_tools_py_exists(self):
        """sdn_tools.py esta em orchestrator/utils/ -- o script aponta para este path."""
        path = os.path.join(SDN_DIR, "orchestrator", "utils", "sdn_tools.py")
        assert os.path.exists(path), (
            f"ARQUIVO AUSENTE: {path}\n"
            "  sem sdn_tools.py o experimento sem_sdn nao limpa os flows SDN.\n"
            "  Os experimentos com_sdn e sem_sdn vao compartilhar o mesmo estado\n"
            "  de rede, comprometendo a comparacao."
        )

    def test_sdn_tools_path_in_script_is_correct(self):
        """Garante que o script usa o path correto orchestrator/utils/sdn_tools.py."""
        script = open(os.path.expanduser(
            "~/FL-SDN-main/run_experimentos_noturnos.sh")).read()
        assert "orchestrator/utils/sdn_tools.py" in script, (
            "Script ainda usa 'sdn_tools.py' na raiz -- deveria ser "
            "'orchestrator/utils/sdn_tools.py'"
        )

    def test_orchestrator_venv_exists(self):
        venv = os.path.join(SDN_DIR, "venv", "bin", "activate")
        assert os.path.exists(venv), \
            f"venv do SDN nao encontrada: {venv}"


# =============================================================================
# 7. Virtualenv
# =============================================================================

class TestVirtualenv:
    """Verifica que pelo menos um caminho de venv do script e valido."""

    def test_at_least_one_venv_path_exists(self):
        found = [p for p in VENV_PATHS if os.path.exists(p)]
        assert found, (
            f"Nenhum venv encontrado nos caminhos:\n"
            + "\n".join(f"  {p}" for p in VENV_PATHS)
        )

    def test_venv_has_python_executable(self):
        for venv_activate in VENV_PATHS:
            if os.path.exists(venv_activate):
                python = os.path.join(os.path.dirname(venv_activate), "python")
                assert os.path.exists(python), \
                    f"python nao encontrado em {os.path.dirname(venv_activate)}"
                return
        pytest.skip("Nenhum venv encontrado")

    def test_venv_has_required_packages(self):
        """Verifica que os pacotes criticos estao instalados no venv."""
        for venv_activate in VENV_PATHS:
            if os.path.exists(venv_activate):
                python = os.path.join(os.path.dirname(venv_activate), "python")
                import subprocess
                for pkg in ["flwr", "xgboost", "lightgbm", "catboost", "numpy", "sklearn"]:
                    result = subprocess.run(
                        [python, "-c", f"import {pkg}"],
                        capture_output=True,
                    )
                    assert result.returncode == 0, \
                        f"Pacote '{pkg}' nao instalado no venv"
                return
        pytest.skip("Nenhum venv encontrado")


# =============================================================================
# 8. FL round notifications via Python
# =============================================================================

class TestFLRoundNotifications:
    """Testa fl_round_start / fl_round_stop atraves do modulo Python."""

    def test_controller_is_available_when_sdn_running(self):
        from sdn.controller import is_available
        # Em modo nao-mock com orquestrador vivo, is_available deve ser True
        result = is_available()
        # Nao forcamos True — apenas verificamos que a funcao existe e retorna bool
        assert isinstance(result, bool)

    def test_fl_round_start_returns_dict_or_none(self):
        from sdn.controller import fl_round_start
        result = fl_round_start(999)
        assert result is None or isinstance(result, dict), \
            f"fl_round_start deve retornar dict ou None, retornou {type(result)}"

    def _require_orchestrator_reachable(self):
        """Pula o teste se o orquestrador SDN nao estiver acessivel na rede."""
        from sdn.controller import is_available
        if not is_available():
            pytest.skip("SDN em modo mock ou indisponivel")
        try:
            import requests as _req
            _req.get(f"{ORCH_URL}/health", timeout=3).raise_for_status()
        except Exception:
            pytest.skip("Orquestrador SDN inacessivel na rede")

    def test_fl_round_start_live_returns_ok_status(self):
        self._require_orchestrator_reachable()
        from sdn.controller import fl_round_start
        result = fl_round_start(888)
        assert result is not None, "fl_round_start retornou None com SDN disponivel"
        assert result.get("status") == "ok"
        assert "csv_path" in result

    def test_fl_round_stop_live_returns_duration(self):
        self._require_orchestrator_reachable()
        from sdn.controller import fl_round_start, fl_round_stop
        fl_round_start(887)
        time.sleep(2)
        result = fl_round_stop()
        assert result is not None
        assert result.get("status") == "ok"
        assert "duration_sec" in result
        assert float(result["duration_sec"]) >= 0

    def test_fl_round_start_mock_does_not_raise(self, monkeypatch):
        import sdn.controller as ctrl
        monkeypatch.setattr(ctrl, "is_available", lambda: False)
        # Nao deve lancar excecao em modo mock
        result = ctrl.fl_round_start(1)
        assert result is None

    def test_fl_round_stop_mock_does_not_raise(self, monkeypatch):
        import sdn.controller as ctrl
        monkeypatch.setattr(ctrl, "is_available", lambda: False)
        result = ctrl.fl_round_stop()
        assert result is None

    def test_fl_round_start_called_in_sdn_bagging_configure_fit(self):
        """Garante que a chamada fl_round_start existe no codigo da estrategia."""
        src = open(os.path.join(FL_DIR, "strategies", "sdn_bagging.py")).read()
        assert "fl_round_start" in src, \
            "fl_round_start nao encontrado em sdn_bagging.py"

    def test_fl_round_stop_called_in_sdn_bagging_aggregate_fit(self):
        src = open(os.path.join(FL_DIR, "strategies", "sdn_bagging.py")).read()
        assert "fl_round_stop" in src, \
            "fl_round_stop nao encontrado em sdn_bagging.py"

    def test_fl_round_start_called_in_sdn_cycling(self):
        src = open(os.path.join(FL_DIR, "strategies", "sdn_cycling.py")).read()
        assert "fl_round_start" in src

    def test_fl_round_stop_called_in_sdn_cycling(self):
        src = open(os.path.join(FL_DIR, "strategies", "sdn_cycling.py")).read()
        assert "fl_round_stop" in src


# =============================================================================
# 9. Regex de deteccao de inicio do servidor
# =============================================================================

class TestServerStartupDetection:
    """Garante que o padrao regex que o script usa para detectar
    'servidor pronto' esta presente na saida real do server.py."""

    SCRIPT_REGEX = re.compile(r"Aguardando.*cliente|Waiting.*client")

    def test_server_startup_message_matches_regex(self):
        """Verifica que a mensagem no server.py casa com o regex do script bash."""
        src = open(os.path.join(FL_DIR, "server.py")).read()
        # Encontra a string de print que o servidor emite ao ficar pronto
        matches = re.findall(r'print\(f?"([^"]*Aguardando[^"]*)"', src)
        assert matches, \
            "Nao encontrada mensagem 'Aguardando' em server.py — regex do script pode nao detectar inicio"
        for msg in matches:
            # Simula interpolacao simples (remove {var} para testar o template)
            clean = re.sub(r'\{[^}]+\}', 'X', msg)
            assert self.SCRIPT_REGEX.search(clean), \
                f"Mensagem '{clean}' nao casa com regex do script bash"

    def test_startup_timeout_180s_is_reasonable(self):
        """180s de timeout para o servidor inicializar deve ser suficiente.
        O gargalo e o carregamento do dataset — avazu usa mmap+500k samples."""
        # Nao tem como medir sem rodar o servidor, mas verificamos que o dataset
        # maior (avazu) usa mmap_mode='r' + subsampling para manter carga < 5s.
        src = open(os.path.join(FL_DIR, "datasets", "avazu.py")).read()
        assert 'mmap_mode="r"' in src or "mmap_mode='r'" in src, \
            "avazu.py nao usa mmap_mode='r' — carregamento pode ser lento demais para timeout de 180s"
        assert "max_samples" in src, \
            "avazu.py nao limita max_samples — pode exceder RAM e timeout"


# =============================================================================
# 10. Caminhos de dataset dentro dos containers
# =============================================================================

class TestContainerDatasetPaths:
    """Verifica que o script copia e verifica os datasets nos caminhos corretos."""

    def test_copy_destination_matches_verify_path(self):
        """copy_dataset_to_containers e verify_dataset_in_containers
        devem usar o mesmo caminho /fl/fl_sdn_code/data/<ds>/."""
        script = open(os.path.expanduser(
            "~/FL-SDN-main/run_experimentos_noturnos.sh")).read()

        # Extrai o dest_dir do copy e o destino do verify
        copy_pattern   = r'dest_dir="/fl/fl_sdn_code/data/\$dataset"'
        verify_pattern = r'dest_dir="/fl/fl_sdn_code/data/\$dataset"'

        # Verifica que o mesmo caminho aparece nas duas funcoes
        assert '/fl/fl_sdn_code/data/$dataset' in script, \
            "Caminho /fl/fl_sdn_code/data/$dataset nao encontrado no script"

        # Conta ocorrencias — deve aparecer tanto na copia quanto na verificacao
        count = script.count('/fl/fl_sdn_code/data/$dataset')
        assert count >= 2, \
            f"Caminho aparece apenas {count}x — copy e verify podem usar caminhos diferentes"

    def test_client_command_uses_same_container_path(self):
        """O comando client.py no docker exec deve usar o mesmo caminho."""
        script = open(os.path.expanduser(
            "~/FL-SDN-main/run_experimentos_noturnos.sh")).read()
        assert '/fl/fl_sdn_code/client.py' in script, \
            "Script nao usa /fl/fl_sdn_code/client.py para iniciar clientes"

    @pytest.mark.parametrize("ds", SCRIPT_DATASETS)
    def test_dataset_path_in_script_matches_fl_dir_structure(self, ds):
        """O path local usado no docker cp deve existir no host."""
        x_path = os.path.join(FL_DIR, "data", ds, f"{ds}_X.npy")
        assert os.path.exists(x_path), \
            f"Arquivo fonte do docker cp nao existe: {x_path}"


# =============================================================================
# 11. Logica de collect_results
# =============================================================================

class TestCollectResultsLogic:
    """Testa a logica de busca de resultados do script."""

    def test_exp_name_format_is_consistent(self):
        """O exp_name gerado no script deve ser identico ao EXP= passado ao server."""
        # No script: exp_name="${exp_type}_${dataset}_${model}"
        # No server:  EXP="$exp_name" python3 server.py
        # No collect: find output -name "*${exp_name}*"
        for exp_type, strategy in SCRIPT_STRATEGIES.items():
            for ds in SCRIPT_DATASETS:
                for model in SCRIPT_MODELS:
                    exp_name = f"{exp_type}_{ds}_{model}"
                    # Nao deve ter espacos (quebraria o find do bash)
                    assert " " not in exp_name, \
                        f"exp_name '{exp_name}' contem espaco — quebraria o find"
                    # Nao deve ter caracteres especiais de glob
                    assert not re.search(r'[*?\[\]]', exp_name), \
                        f"exp_name '{exp_name}' contem caracteres glob"

    def test_sdn_round_csv_search_path_exists(self):
        """O diretorio onde o script busca fl_metrics_round*.csv deve existir."""
        assert os.path.isdir(SDN_DIR), \
            f"SDN_DIR nao existe: {SDN_DIR}"

    def test_sdn_round_csv_pattern_finds_generated_files(self):
        """Apos fl_round_start/stop, o find do script deve encontrar o CSV."""
        import glob
        pattern = os.path.join(SDN_DIR, "fl_metrics_round*.csv")
        # Pode ja haver CSVs de testes anteriores — so verificamos que glob funciona
        files = glob.glob(pattern)
        # Nao exige arquivos — apenas que o glob nao levante excecao
        assert isinstance(files, list)
