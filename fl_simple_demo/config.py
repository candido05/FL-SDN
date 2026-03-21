"""
Configuracoes centralizadas do FL Simple Demo.
Modifique os valores abaixo conforme necessario.
"""

# ---------------------------------------------------------------------------
# Rede
# ---------------------------------------------------------------------------

# Épocas locais por categoria (simula diferença de capacidade computacional)
LOCAL_EPOCHS_BY_CAT = {
    "cat1": 50,   # low — treina menos, modelo menor
    "cat2": 100,  # medium
    "cat3": 150,  # high — modelo maior, mais bytes na rede
}

# Mapeamento client_id → categoria
CLIENT_CATEGORIES = {
    0: "cat1",
    1: "cat1",
    2: "cat2",
    3: "cat2",
    4: "cat3",
    5: "cat3",
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
# Integracao SDN (OpenDaylight)
# ---------------------------------------------------------------------------

# Conexao com o controlador SDN
SDN_CONTROLLER_IP   = "172.16.1.1"   # IP do OpenDaylight
SDN_CONTROLLER_PORT = 8181            # Porta REST do ODL
SDN_CONTROLLER_USER = "admin"         # Usuario ODL
SDN_CONTROLLER_PASS = "admin"         # Senha ODL

# Modo mock: True = gera metricas simuladas (sem ODL real)
# Util para testes locais e validacao da logica
SDN_MOCK_MODE = True

# Limiares de elegibilidade de clientes
SDN_MIN_BANDWIDTH_MBPS = 10.0   # Largura de banda minima para participar
SDN_MAX_LATENCY_MS     = 50.0   # Latencia maxima aceita
SDN_MAX_PACKET_LOSS    = 0.10   # Perda de pacotes maxima (10%)

# Pesos para calculo do efficiency_score (devem somar 1.0)
SDN_SCORE_WEIGHTS = {
    "bandwidth":   0.5,   # Peso da largura de banda
    "latency":     0.3,   # Peso da latencia
    "packet_loss": 0.2,   # Peso da perda de pacotes
}

# Adaptacao de epocas locais baseada em rede
# True = ajusta epocas conforme efficiency_score do cliente
# False = usa epocas fixas por categoria (comportamento original)
SDN_ADAPTIVE_EPOCHS = True

# Mapeamento client_id → IP na rede GNS3
SDN_CLIENT_IPS = {
    0: "172.16.1.12",
    1: "172.16.1.12",
    2: "172.16.1.13",
    3: "172.16.1.13",
    4: "172.16.1.14",
    5: "172.16.1.14",
}
