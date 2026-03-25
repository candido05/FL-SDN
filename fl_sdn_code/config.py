"""
Configuracoes centralizadas do FL-SDN.
Modifique os valores abaixo conforme necessario.
"""

# ---------------------------------------------------------------------------
# Rede — Topologia GNS3
# ---------------------------------------------------------------------------

# Endereco do SDN Orchestrator (FastAPI no host Ubuntu)
# Substituiu a conexao direta ao ODL (porta 8181) para evitar duplo polling
SDN_ORCHESTRATOR_IP   = "172.16.1.1"
SDN_ORCHESTRATOR_PORT = "8000"

# Variaveis antigas do ODL:
# SDN_CONTROLLER_IP   = "172.16.1.1"
# SDN_CONTROLLER_PORT = "8181"
# SDN_CONTROLLER_USER = "admin"
# SDN_CONTROLLER_PASS = "admin"

# Mapeamento client_id → IP na rede GNS3
SDN_CLIENT_IPS = {
    0: "172.16.1.10",   # FL-Node-1
    1: "172.16.1.16",   # FL-Node-5
    2: "172.16.1.11",   # FL-Node-2
    3: "172.16.1.14",   # FL-Node-4
    4: "172.16.1.13",   # FL-Node-3
    5: "172.16.1.17",   # FL-Node-6
}

# Categorias de clientes (usadas para QoS/priorizacao de trafego SDN)
# Todos treinam com o mesmo numero de epocas (LOCAL_EPOCHS)
CLIENT_CATEGORIES = {
    0: "cat1",  # FL-Node-1
    1: "cat1",  # FL-Node-5
    2: "cat2",  # FL-Node-2
    3: "cat2",  # FL-Node-4
    4: "cat3",  # FL-Node-3
    5: "cat3",  # FL-Node-6
}



CLIENT_CONNECT_ADDRESS = "172.16.1.1:8080"
NUM_CLIENTS = 6
SERVER_HOST = "0.0.0.0"       # IP em que o servidor escuta (0.0.0.0 = todas as interfaces)
SERVER_PORT = 8080             # Porta gRPC do servidor
SERVER_ADDRESS = f"{SERVER_HOST}:{SERVER_PORT}"

# Endereco que os CLIENTES usam para conectar ao servidor
# Se rodar tudo na mesma maquina, use 127.0.0.1
# Se rodar em maquinas diferentes, coloque o IP real do servidor
#CLIENT_CONNECT_ADDRESS = f"127.0.0.1:{SERVER_PORT}"

# ---------------------------------------------------------------------------
# Treinamento Federado
# ---------------------------------------------------------------------------
#NUM_CLIENTS = 3                # Numero de clientes federados
NUM_ROUNDS = 20                 # Rounds de comunicacao servidor-clientes
LOCAL_EPOCHS = 100             # Numero de estimadores/iteracoes por treino local
                               # (n_estimators para XGBoost/LightGBM, iterations para CatBoost)
LOG_EVERY = 10                 # Mostrar progresso a cada N epocas locais

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
N_SAMPLES = 50_000             # Amostras do Higgs a carregar (total)
TEST_SIZE = 0.2                # Fracao para teste
RANDOM_SEED = 42               # Seed para reproducibilidade

# ---------------------------------------------------------------------------
# Modelos - Hiperparametros base
# ---------------------------------------------------------------------------
XGBOOST_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": LOCAL_EPOCHS,
    "random_state": RANDOM_SEED,
}

LIGHTGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "verbosity": -1,
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": LOCAL_EPOCHS,
    "random_state": RANDOM_SEED,
}

CATBOOST_PARAMS = {
    "loss_function": "Logloss",
    "depth": 6,
    "learning_rate": 0.1,
    "iterations": LOCAL_EPOCHS,
    "random_seed": RANDOM_SEED,
    "verbose": False,
}

# ---------------------------------------------------------------------------
# Controle de Overfitting — Tree Cap e Early Stopping
# ---------------------------------------------------------------------------
MAX_TOTAL_TREES = 500              # Maximo de arvores acumuladas com warm start
                                   # Evita crescimento ilimitado ao longo dos rounds
EARLY_STOPPING_ROUNDS = 10         # Parar se validacao nao melhorar em N rounds
VALIDATION_SPLIT = 0.15            # Fracao do treino local reservada para validacao
                                   # Usada para early stopping (nao treina nesta parcela)

# ---------------------------------------------------------------------------
# Warm Start — Taxa de decaimento de n_estimators e learning_rate
# ---------------------------------------------------------------------------
# Com warm start, cada round parte do modelo ja treinado do round anterior.
# Para evitar overfitting, tanto n_estimators quanto learning_rate devem
# decair gradualmente: o modelo precisa de ajustes cada vez menores e mais
# refinados conforme os rounds avancam.
#
# Decaimento exponencial por round (server_round comecando em 1):
#   valor_round = max(piso, base * decay_rate ^ (server_round - 1))
#
# Tabela de exemplo (LOCAL_EPOCHS=100, learning_rate=0.10, NUM_ROUNDS=20):
#
#  Round | n_new  | n_ratio | lr     | Observacao
#  ------+--------+---------+--------+------------------------------
#    1   |  100   |  100%   | 0.100  | sem warm start
#    2   |   85   |   85%   | 0.093  | primeiro warm start
#    3   |   72   |   72%   | 0.086  |
#    4   |   61   |   61%   | 0.080  |
#    5   |   52   |   52%   | 0.075  |
#    6   |   44   |   44%   | 0.069  |
#    7   |   37   |   37%   | 0.064  |
#    8   |   32   |   32%   | 0.060  |
#    9   |   30   |   30%   | 0.056  | piso de n_estimators atingido
#   10   |   30   |   30%   | 0.052  |
#   12   |   30   |   30%   | 0.046  |
#   15   |   30   |   30%   | 0.040  | piso de lr atingido
#   20   |   30   |   30%   | 0.040  | estabilizado
#
# O piso garante que o modelo continue aprendendo dos dados locais.

WARM_START_TREE_DECAY    = 0.85   # Taxa de decaimento exponencial do n_estimators
WARM_START_TREE_MIN_RATIO = 0.30  # Piso: minimo de epocas locais como fracao de LOCAL_EPOCHS

WARM_START_LR_DECAY      = 0.93   # Taxa de decaimento exponencial do learning_rate
WARM_START_LR_MIN_RATIO  = 0.40   # Piso: minimo de learning_rate como fracao do valor base


# Hiperparametros tunados (preenchido pelo grid_search.py)
# Se None, usa os parametros base (XGBOOST_PARAMS, etc.)
# Formato: {"learning_rate": 0.05, "max_depth": 4, ...}
#
# Prioridade: env TUNED_PARAMS_JSON > valor manual abaixo
import json as _json, os as _os
_tuned_env = _os.environ.get("TUNED_PARAMS_JSON", "")
TUNED_PARAMS = _json.loads(_tuned_env) if _tuned_env else None
# Para definir manualmente:
# TUNED_PARAMS = {"learning_rate": 0.05, "max_depth": 4, ...}

# ---------------------------------------------------------------------------
# Integracao SDN — Limiares e scoring
# ---------------------------------------------------------------------------

# Modo mock: True = gera metricas simuladas (sem orquestrador real)
# Util para testes locais e validacao da logica
SDN_MOCK_MODE = False
#SDN_MOCK_MODE = True   # descomentar para testes locais sem SDN

# Adaptacao de epocas locais baseada em rede
# True = ajusta epocas conforme efficiency_score do cliente
# False = todos usam LOCAL_EPOCHS (uniforme)
# ATENCAO: manter False nos experimentos com/sem SDN para comparacao justa
SDN_ADAPTIVE_EPOCHS = False
#SDN_ADAPTIVE_EPOCHS = True   # descomentar se quiser adaptacao por rede

# Limiares de elegibilidade de clientes
# Alinhado com REROUTE_THRESH do orquestrador (0.75 × 20 Mbps = 15 Mbps)
SDN_MIN_BANDWIDTH_MBPS = 15.0
#SDN_MIN_BANDWIDTH_MBPS = 10.0  # valor anterior (topologia antiga 100 Mbps)
SDN_MAX_LATENCY_MS     = 50.0   # Latencia maxima aceita
SDN_MAX_PACKET_LOSS    = 0.10   # Perda de pacotes maxima (10%)

# Pesos para calculo do efficiency_score (devem somar 1.0)
SDN_SCORE_WEIGHTS = {
    "bandwidth":   0.5,
    "latency":     0.3,
    "packet_loss": 0.2,
}

# ---------------------------------------------------------------------------
# Client Health Score — selecao/exclusao dinamica de clientes
# ---------------------------------------------------------------------------

# Perfil de pesos para o health score.
# Opcoes: "balanced", "contribution", "resource", "network", "custom"
#   - balanced:     contribuicao=0.40, recurso=0.30, rede=0.30
#   - contribution: contribuicao=0.70, recurso=0.15, rede=0.15
#   - resource:     contribuicao=0.15, recurso=0.70, rede=0.15
#   - network:      contribuicao=0.15, recurso=0.15, rede=0.70
#   - custom:       usa HEALTH_SCORE_CUSTOM_WEIGHTS abaixo
HEALTH_SCORE_PROFILE = "balanced"

# Pesos customizados (so usados quando HEALTH_SCORE_PROFILE = "custom")
# Devem somar 1.0
HEALTH_SCORE_CUSTOM_WEIGHTS = {
    "contribution": 0.40,
    "resource": 0.30,
    "network": 0.30,
}

# Maximo de clientes excluidos por round
HEALTH_SCORE_MAX_EXCLUDE = 2

# Rounds minimos antes de comecar a excluir (para acumular historico)
HEALTH_SCORE_MIN_ROUNDS = 2

# Clientes com score abaixo deste limiar sao candidatos a exclusao
HEALTH_SCORE_THRESHOLD = 0.50

# Habilitar/desabilitar o sistema de health score
# True = ativa exclusao dinamica nas estrategias SDN
# False = comportamento original (sem exclusao)
HEALTH_SCORE_ENABLED = True
