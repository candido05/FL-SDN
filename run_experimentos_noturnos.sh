#!/bin/bash
# =============================================================================
#  run_experimentos_noturnos.sh
#  Executa todos os experimentos FL-SDN de forma sequencial e automatica.
#
#  Ordem: epsilon -> higgs_full -> avazu
#  Para cada dataset: com_sdn -> sem_sdn
#
#  PRE-REQUISITOS (antes de rodar este script):
#    1. ODL Karaf iniciado e estavel (~90s apos boot)
#    2. GNS3 aberto com topologia iniciada (todos os nos verdes)
#    3. sudo bash ~/Downloads/setup_switch.sh     (ja executado)
#    4. sudo ./Downloads/setup_experimento.sh     (ja executado)
#    5. python3 sdn_orchestrator.py rodando em terminal separado
#
#  Uso:
#    chmod +x run_experimentos_noturnos.sh
#    nohup sudo bash run_experimentos_noturnos.sh > experimentos.log 2>&1 &
#    tail -f experimentos.log   -- acompanhe o progresso
# =============================================================================

set -euo pipefail

# -- Configuracoes ------------------------------------------------------------
FL_DIR="$HOME/FL-SDN-main/fl_sdn_code"
SDN_DIR="$HOME/sdn-project-main"
RESULTS_DIR="$HOME/resultados_$(date +%Y%m%d)"
LOG_DIR="$RESULTS_DIR/logs"
ORCHESTRATOR_URL="http://127.0.0.1:8000"
EXPERIMENT_TIMEOUT=86400  # segundos max por experimento (24h)

# Virtualenv -- tenta os dois caminhos possiveis
VENV_PATHS=(
    "$HOME/fl-node/venv/bin/activate"
    "$FL_DIR/../venv/bin/activate"
    "$FL_DIR/venv/bin/activate"
)
ACTIVE_VENV=""

# Datasets na ordem de execucao (menor -> maior)
DATASETS=("epsilon" "higgs_full" "avazu")

# Modelos a testar
MODELS=("xgboost")

# Estrategias: com_sdn e sem_sdn
declare -A STRATEGY_MAP=(
    ["com_sdn"]="sdn-bagging"
    ["sem_sdn"]="bagging"
)

# IPs dos containers FL por client-id
declare -A CLIENT_IPS=(
    [0]="172.16.1.10"
    [1]="172.16.1.16"
    [2]="172.16.1.11"
    [3]="172.16.1.14"
    [4]="172.16.1.13"
    [5]="172.16.1.17"
)

# -- Cores --------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

log()  { echo -e "$(date '+%Y-%m-%d %H:%M:%S') ${CYAN}[INFO]${NC}  $1"; }
ok()   { echo -e "$(date '+%Y-%m-%d %H:%M:%S') ${GREEN}[OK]${NC}    $1"; }
warn() { echo -e "$(date '+%Y-%m-%d %H:%M:%S') ${YELLOW}[WARN]${NC}  $1"; }
err()  { echo -e "$(date '+%Y-%m-%d %H:%M:%S') ${RED}[ERRO]${NC}  $1"; }
step() {
    echo ""
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') ${BOLD}${BLUE}---- $1 ----${NC}"
}

# -- Utilitarios Docker -------------------------------------------------------

get_fl_containers() {
    sudo docker ps --format '{{.Names}}' | grep -E 'GNS3\.FL-Node' 2>/dev/null || true
}

get_bg_containers() {
    sudo docker ps --format '{{.Names}}' | grep -E 'GNS3\.BG-Node' 2>/dev/null || true
}

get_container_by_ip() {
    local target_ip="$1"
    while IFS= read -r container; do
        local ip
        ip=$(sudo docker exec "$container" \
            ip addr show eth0 2>/dev/null | grep -oP 'inet \K[\d.]+' || true)
        if [ "$ip" = "$target_ip" ]; then
            echo "$container"
            return
        fi
    done < <(get_fl_containers)
}

# -- Utilitarios SDN ----------------------------------------------------------

check_orchestrator() {
    curl -sf "$ORCHESTRATOR_URL/health" >/dev/null 2>&1
}

get_switch_count() {
    curl -sf "$ORCHESTRATOR_URL/health" 2>/dev/null \
        | python3 -c \
          "import sys,json; print(json.load(sys.stdin).get('switches',0))" \
          2>/dev/null || echo "0"
}

get_host_count() {
    curl -sf "$ORCHESTRATOR_URL/health" 2>/dev/null \
        | python3 -c \
          "import sys,json; print(json.load(sys.stdin).get('hosts',0))" \
          2>/dev/null || echo "0"
}

# =============================================================================
# INICIALIZACAO
# =============================================================================

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

echo ""
echo -e "${BOLD}${BLUE}"
echo "================================================================"
echo "=   EXPERIMENTOS FL-SDN -- EXECUCAO AUTOMATICA NOTURNA        ="
echo "================================================================"
echo -e "${NC}"
log "Inicio: $(date)"
log "Resultados em: $RESULTS_DIR"
log "Datasets: ${DATASETS[*]}"

# =============================================================================
# PASSO 0 -- Validacoes iniciais
# =============================================================================
step "PASSO 0 -- Validacoes do ambiente"

# Localiza virtualenv
for venv_path in "${VENV_PATHS[@]}"; do
    if [ -f "$venv_path" ]; then
        ACTIVE_VENV="$venv_path"
        ok "Virtualenv: $ACTIVE_VENV"
        break
    fi
done
if [ -z "$ACTIVE_VENV" ]; then
    err "Nenhum virtualenv encontrado. Caminhos tentados:"
    for p in "${VENV_PATHS[@]}"; do echo "  $p"; done
    exit 1
fi

# Verifica orquestrador
if ! check_orchestrator; then
    err "SDN Orchestrator nao esta respondendo em $ORCHESTRATOR_URL"
    err "Inicie: cd ~/sdn-project-main && source venv/bin/activate && python3 sdn_orchestrator.py"
    exit 1
fi
ok "SDN Orchestrator: respondendo"

# Verifica switches
N_SW=$(get_switch_count)
if [ "$N_SW" -lt 10 ]; then
    warn "Apenas $N_SW switches. Aguardando 30s para ODL estabilizar..."
    sleep 30
    N_SW=$(get_switch_count)
    if [ "$N_SW" -lt 10 ]; then
        err "Switches insuficientes ($N_SW). Verifique GNS3 e setup_switch.sh."
        exit 1
    fi
fi
ok "Switches descobertos: $N_SW"

# Verifica hosts
N_HOSTS=$(get_host_count)
if [ "$N_HOSTS" -lt 6 ]; then
    warn "Apenas $N_HOSTS hosts descobertos. Aguardando 20s..."
    sleep 20
    N_HOSTS=$(get_host_count)
fi
ok "Hosts descobertos: $N_HOSTS"

# Verifica containers FL
N_FL=$(get_fl_containers | grep -c . || true)
if [ "$N_FL" -lt 6 ]; then
    err "Apenas $N_FL containers FL ativos. Esperado: 6."
    exit 1
fi
ok "Containers FL ativos: $N_FL"

# Verifica endpoints de sessao FL (integracao SDN -> FL rounds)
log "Verificando endpoint /fl/training/start..."
FL_ENDPOINT_OK=$(curl -sf -X POST "$ORCHESTRATOR_URL/fl/training/start" \
    -H "Content-Type: application/json" \
    -d '{"round": 0}' 2>/dev/null \
    | python3 -c \
      "import sys,json; print(json.load(sys.stdin).get('status','erro'))" \
      2>/dev/null || echo "erro")

if [ "$FL_ENDPOINT_OK" = "ok" ]; then
    curl -sf -X POST "$ORCHESTRATOR_URL/fl/training/stop" >/dev/null 2>&1 || true
    ok "Endpoint /fl/training/start: disponivel -- CSVs por round serao gerados"
else
    warn "Endpoint /fl/training/start indisponivel -- verifique api.py do orquestrador"
    warn "Experimentos continuarao mas sem CSVs SDN por round"
fi
FL_ENDPOINT_AVAILABLE="$FL_ENDPOINT_OK"

# Verifica e prepara datasets no host
for ds in "${DATASETS[@]}"; do
    x_file="$FL_DIR/data/$ds/${ds}_X.npy"
    y_file="$FL_DIR/data/$ds/${ds}_y.npy"
    if [ ! -f "$x_file" ] || [ ! -f "$y_file" ]; then
        warn "Dataset '$ds' nao encontrado em $FL_DIR/data/$ds/. Preparando..."
        cd "$FL_DIR"
        # shellcheck disable=SC1090
        source "$ACTIVE_VENV"
        python3 tools/prepare_datasets.py --dataset "$ds" \
            >> "$LOG_DIR/prepare_${ds}.log" 2>&1
        deactivate 2>/dev/null || true
        ok "Dataset '$ds' preparado."
    else
        ok "Dataset '$ds': $(du -sh "$x_file" | cut -f1) (X) + $(du -sh "$y_file" | cut -f1) (y)"
    fi
done

# =============================================================================
# FUNCAO: Limpar containers e CSVs de rounds anteriores
# =============================================================================
cleanup_before_experiment() {
    local dataset="$1"
    local exp_type="$2"
    step "Limpeza -- $exp_type / $dataset"

    # Limpa cada container FL
    while IFS= read -r container; do
        log "Limpando $container..."
        sudo docker exec "$container" bash -c "
            pkill -9 -f 'python.*client.py' 2>/dev/null || true
            pkill -9 -f 'iperf3'            2>/dev/null || true
            sleep 2
            find /fl -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
            find /fl -name '*.pyc' -delete 2>/dev/null || true
            rm -rf /fl/fl_sdn_code/output/* 2>/dev/null || true
            rm -f /tmp/*.tar.gz /tmp/*.npy /tmp/*.pkl /tmp/client_*.log 2>/dev/null || true
            echo \"Espaco /fl: \$(df -h /fl | awk 'NR==2{print \$4}') disponivel\"
        " 2>/dev/null || warn "  Erro ao limpar $container (continuando)"
    done < <(get_fl_containers)

    # Mata iperf3 nos BG-Nodes
    while IFS= read -r container; do
        sudo docker exec "$container" bash -c \
            "pkill -9 -f 'iperf3' 2>/dev/null; true" 2>/dev/null || true
    done < <(get_bg_containers)

    # Remove CSVs de rounds SDN anteriores
    local n_removed=0
    while IFS= read -r csv_file; do
        rm -f "$csv_file"
        n_removed=$((n_removed + 1))
    done < <(find "$SDN_DIR" -maxdepth 3 \
        -name "fl_metrics_round*.csv" 2>/dev/null || true)
    [ "$n_removed" -gt 0 ] && log "  $n_removed fl_metrics_round*.csv antigos removidos"

    ok "Limpeza concluida"
}

# =============================================================================
# FUNCAO: Copiar dataset para todos os containers FL
# =============================================================================
copy_dataset_to_containers() {
    local dataset="$1"
    step "Copiando dataset '$dataset' para containers"

    local x_file="$FL_DIR/data/$dataset/${dataset}_X.npy"
    local y_file="$FL_DIR/data/$dataset/${dataset}_y.npy"
    local dest_dir="/fl/fl_sdn_code/data/$dataset"

    if [ ! -f "$x_file" ]; then
        err "Arquivo nao encontrado: $x_file"
        return 1
    fi

    local size_x size_y local_size_x
    size_x=$(du -sh "$x_file" | cut -f1)
    size_y=$(du -sh "$y_file" | cut -f1)
    local_size_x=$(stat -c%s "$x_file")
    log "Tamanho a copiar: X=$size_x  y=$size_y"

    local copied=0 skipped=0
    while IFS= read -r container; do
        sudo docker exec "$container" mkdir -p "$dest_dir" 2>/dev/null

        local remote_size
        remote_size=$(sudo docker exec "$container" \
            stat -c%s "$dest_dir/${dataset}_X.npy" 2>/dev/null || echo "0")

        if [ "$remote_size" = "$local_size_x" ]; then
            log "  $container: ja presente (mesmo tamanho) -- pulando"
            skipped=$((skipped + 1))
        else
            log "  $container: copiando X ($size_x)..."
            sudo docker cp "$x_file" "$container:$dest_dir/${dataset}_X.npy"
            log "  $container: copiando y ($size_y)..."
            sudo docker cp "$y_file" "$container:$dest_dir/${dataset}_y.npy"
            copied=$((copied + 1))
        fi
    done < <(get_fl_containers)

    ok "Dataset '$dataset': $copied copiado(s), $skipped ja presentes"
}

# =============================================================================
# FUNCAO: Verificar dataset nos containers
# =============================================================================
verify_dataset_in_containers() {
    local dataset="$1"
    local dest_dir="/fl/fl_sdn_code/data/$dataset"
    local errors=0

    while IFS= read -r container; do
        local result
        result=$(sudo docker exec "$container" python3 -c "
import numpy as np, sys
try:
    X = np.load('$dest_dir/${dataset}_X.npy')
    y = np.load('$dest_dir/${dataset}_y.npy')
    print(f'OK shape={X.shape}')
except Exception as e:
    print(f'ERRO:{e}')
" 2>/dev/null || echo "ERRO:exec_falhou")
        if [[ "$result" != OK* ]]; then
            warn "  $container: $result"
            errors=$((errors + 1))
        fi
    done < <(get_fl_containers)

    if [ "$errors" -gt 0 ]; then
        err "$errors container(s) com problema no dataset '$dataset'"
        return 1
    fi
    ok "Dataset '$dataset' verificado em todos os containers"
}

# =============================================================================
# FUNCAO: Iniciar background traffic
# =============================================================================
start_background_traffic() {
    step "Iniciando background traffic (iperf3)"

    local bg2 bg1 bg3
    bg2=$(sudo docker ps --format '{{.Names}}' | grep 'BG-Node-2' | head -1 || true)
    bg1=$(sudo docker ps --format '{{.Names}}' | grep 'BG-Node-1' | head -1 || true)
    bg3=$(sudo docker ps --format '{{.Names}}' | grep 'BG-Node-3' | head -1 || true)

    if [ -z "$bg2" ]; then
        warn "BG-Node-2 nao encontrado -- experimento sem congestionamento"
        return 0
    fi

    sudo docker exec "$bg2" bash -c "pkill -9 -f iperf3 2>/dev/null; sleep 1" || true
    [ -n "$bg1" ] && sudo docker exec "$bg1" bash -c "pkill -9 -f iperf3 2>/dev/null; true" || true
    [ -n "$bg3" ] && sudo docker exec "$bg3" bash -c "pkill -9 -f iperf3 2>/dev/null; true" || true

    sudo docker exec -d "$bg2" bash -c "iperf3 -s -p 5201 -D"
    sudo docker exec -d "$bg2" bash -c "iperf3 -s -p 5202 -D"
    sleep 2

    [ -n "$bg1" ] && sudo docker exec -d "$bg1" bash -c \
        "iperf3 -c 172.16.1.15 -p 5201 -t 999999 -b 40M -P 2 >/dev/null 2>&1 &"
    [ -n "$bg3" ] && sudo docker exec -d "$bg3" bash -c \
        "iperf3 -c 172.16.1.15 -p 5202 -t 999999 -b 40M -P 2 >/dev/null 2>&1 &"

    sleep 3
    ok "Background traffic ativo: BG1->BG2 (:5201) e BG3->BG2 (:5202) -- ~80 Mbps total"
}

# =============================================================================
# FUNCAO: Parar background traffic
# =============================================================================
stop_background_traffic() {
    while IFS= read -r container; do
        sudo docker exec "$container" bash -c \
            "pkill -9 -f iperf3 2>/dev/null; true" 2>/dev/null || true
    done < <(get_bg_containers)
    log "Background traffic parado"
}

# =============================================================================
# FUNCAO: Coletar resultados do experimento
# =============================================================================
collect_results() {
    local exp_name="$1"
    local out_base="$RESULTS_DIR/${exp_name}_output"
    mkdir -p "$out_base"

    # 1. Resultados FL (CSV de rounds do servidor Flower)
    local fl_output_dir
    fl_output_dir=$(find "$FL_DIR/output" -maxdepth 1 -type d \
        -name "*${exp_name}*" 2>/dev/null | tail -1 || true)

    if [ -n "$fl_output_dir" ] && [ -d "$fl_output_dir" ]; then
        cp -r "$fl_output_dir/." "$out_base/"
        ok "CSVs FL copiados de: $fl_output_dir"
    else
        warn "Diretorio FL output nao encontrado para '$exp_name'"
        fl_output_dir=$(ls -td "$FL_DIR/output/"*/ 2>/dev/null | head -1 || true)
        if [ -n "$fl_output_dir" ]; then
            cp -r "$fl_output_dir/." "$out_base/"
            warn "Usando output mais recente: $fl_output_dir"
        fi
    fi

    # 2. CSVs de rounds SDN (fl_metrics_roundN_*.csv)
    if [ "$FL_ENDPOINT_AVAILABLE" = "ok" ]; then
        local sdn_round_dir="$out_base/sdn_rounds"
        mkdir -p "$sdn_round_dir"
        local n_round_csvs=0

        while IFS= read -r csv_file; do
            cp "$csv_file" "$sdn_round_dir/"
            n_round_csvs=$((n_round_csvs + 1))
        done < <(find "$SDN_DIR" -maxdepth 3 \
            -name "fl_metrics_round*.csv" 2>/dev/null || true)

        if [ "$n_round_csvs" -gt 0 ]; then
            ok "CSVs SDN por round: $n_round_csvs arquivo(s) -> $sdn_round_dir"
        else
            warn "Nenhum fl_metrics_round*.csv encontrado"
        fi
    fi

    # 3. sdn_metrics.csv geral da sessao
    while IFS= read -r csv_file; do
        cp "$csv_file" "$out_base/"
        ok "sdn_metrics copiado: $(basename "$csv_file")"
    done < <(find "$SDN_DIR" -maxdepth 3 \
        -name "sdn_metrics*.csv" \
        -newer "$RESULTS_DIR" 2>/dev/null || true)

    # 4. Logs dos clientes FL
    local client_log_dir="$out_base/client_logs"
    mkdir -p "$client_log_dir"
    for client_id in 0 1 2 3 4 5; do
        local target_ip="${CLIENT_IPS[$client_id]}"
        local container
        container=$(get_container_by_ip "$target_ip")
        [ -z "$container" ] && continue
        sudo docker cp \
            "$container:/tmp/client_${client_id}.log" \
            "$client_log_dir/client_${client_id}.log" 2>/dev/null || true
    done
    ok "Logs de clientes copiados -> $client_log_dir"
}

# =============================================================================
# FUNCAO PRINCIPAL: Rodar um experimento completo
# =============================================================================
run_experiment() {
    local dataset="$1"
    local model="$2"
    local exp_type="$3"   # "com_sdn" ou "sem_sdn"
    local strategy="${STRATEGY_MAP[$exp_type]}"
    local exp_name="${exp_type}_${dataset}_${model}"
    local server_log="$LOG_DIR/server_${exp_name}.log"

    step "EXPERIMENTO: $exp_name"
    log "Dataset: $dataset | Modelo: $model | Estrategia: $strategy"
    log "Log servidor: $server_log"

    local exp_start_time
    exp_start_time=$(date +%s)

    # Configura SDN conforme o tipo de experimento
    if [ "$exp_type" = "sem_sdn" ]; then
        log "Modo sem_sdn: removendo flows SDN..."
        cd "$SDN_DIR" && source venv/bin/activate 2>/dev/null
        # sdn_tools.py esta em orchestrator/utils/ -- nao na raiz do projeto
        python3 orchestrator/utils/sdn_tools.py clean \
            >> "$LOG_DIR/sdn_clean_${exp_name}.log" 2>&1 || true
        deactivate 2>/dev/null || true
        ok "Flows SDN removidos"
    else
        if ! check_orchestrator; then
            err "Orquestrador SDN nao esta respondendo! Abortando $exp_name."
            return 1
        fi
        ok "Modo com_sdn: orquestrador ativo ($N_SW switches / $N_HOSTS hosts)"
    fi

    # Inicia servidor FL em background
    log "Iniciando servidor FL..."
    cd "$FL_DIR"
    # shellcheck disable=SC1090
    source "$ACTIVE_VENV"

    EXP="$exp_name" python3 server.py \
        --model "$model" \
        --strategy "$strategy" \
        --dataset "$dataset" \
        >> "$server_log" 2>&1 &
    local SERVER_PID=$!
    log "Servidor FL PID: $SERVER_PID"

    # Aguarda servidor anunciar que esta pronto
    log "Aguardando servidor inicializar..."
    local wait_sec=0
    while ! grep -qE "Aguardando.*cliente|Waiting.*client" "$server_log" 2>/dev/null; do
        sleep 3
        wait_sec=$((wait_sec + 3))

        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            err "Servidor FL encerrou inesperadamente. Ultimas linhas do log:"
            tail -30 "$server_log"
            return 1
        fi

        if [ $wait_sec -gt 180 ]; then
            err "Servidor FL nao ficou pronto em 180s. Log:"
            tail -20 "$server_log"
            kill "$SERVER_PID" 2>/dev/null || true
            return 1
        fi
    done
    ok "Servidor FL pronto -- iniciando clientes"

    # Inicia os 6 clientes FL nos containers
    local started_clients=0
    for client_id in 0 1 2 3 4 5; do
        local target_ip="${CLIENT_IPS[$client_id]}"
        local container
        container=$(get_container_by_ip "$target_ip")

        if [ -z "$container" ]; then
            warn "Container com IP $target_ip (client-id=$client_id) nao encontrado -- pulando"
            continue
        fi

        sudo docker exec -d "$container" bash -c \
            "python3 /fl/fl_sdn_code/client.py \
                --client-id $client_id \
                --model $model \
                --dataset $dataset \
                > /tmp/client_${client_id}.log 2>&1"

        log "  client-id=$client_id iniciado em $container ($target_ip)"
        started_clients=$((started_clients + 1))
    done
    ok "$started_clients clientes FL iniciados"

    # Monitora o servidor FL ate concluir (ou timeout)
    log "Monitorando execucao (timeout: ${EXPERIMENT_TIMEOUT}s)..."
    local elapsed=0
    local last_round="0"

    while kill -0 "$SERVER_PID" 2>/dev/null; do
        sleep 30
        elapsed=$((elapsed + 30))

        local current_round
        current_round=$(grep -oP 'Round\s+\K[0-9]+' "$server_log" 2>/dev/null \
            | tail -1 || echo "0")
        if [ "$current_round" != "$last_round" ]; then
            local acc
            acc=$(grep -oP 'acc=\K[0-9.]+' "$server_log" 2>/dev/null \
                | tail -1 || echo "?")
            log "Round $current_round/20 | acc=$acc | ${elapsed}s decorridos"
            last_round="$current_round"
        fi

        if [ "$elapsed" -gt "$EXPERIMENT_TIMEOUT" ]; then
            err "TIMEOUT: $exp_name excedeu ${EXPERIMENT_TIMEOUT}s -- encerrando"
            kill "$SERVER_PID" 2>/dev/null || true
            break
        fi
    done

    local exp_end_time
    exp_end_time=$(date +%s)
    local duration_min=$(( (exp_end_time - exp_start_time) / 60 ))
    ok "Experimento $exp_name finalizado em ${duration_min} minutos"

    collect_results "$exp_name"

    {
        echo "exp=$exp_name"
        echo "dataset=$dataset"
        echo "model=$model"
        echo "strategy=$strategy"
        echo "inicio=$(date -d @"$exp_start_time" '+%Y-%m-%d %H:%M:%S')"
        echo "fim=$(date '+%Y-%m-%d %H:%M:%S')"
        echo "duracao_min=$duration_min"
    } > "$RESULTS_DIR/${exp_name}.done"

    deactivate 2>/dev/null || true
    return 0
}

# =============================================================================
# LOOP PRINCIPAL
# =============================================================================

TOTAL_START=$(date +%s)
EXPERIMENT_COUNT=0
FAILED_EXPERIMENTS=()

for dataset in "${DATASETS[@]}"; do
    echo ""
    echo -e "${BOLD}${YELLOW}============================================================${NC}"
    echo -e "${BOLD}${YELLOW}  DATASET: $dataset${NC}"
    echo -e "${BOLD}${YELLOW}============================================================${NC}"

    for model in "${MODELS[@]}"; do

        # -- Experimento COM SDN ----------------------------------------------
        cleanup_before_experiment "$dataset" "com_sdn"

        if ! copy_dataset_to_containers "$dataset"; then
            err "Falha ao copiar '$dataset' -- pulando experimentos deste dataset"
            FAILED_EXPERIMENTS+=("com_sdn_${dataset}_${model}:copia_falhou")
            FAILED_EXPERIMENTS+=("sem_sdn_${dataset}_${model}:copia_falhou")
            continue 2
        fi

        verify_dataset_in_containers "$dataset" || \
            warn "Alguns containers com problema -- experimento pode falhar"

        start_background_traffic

        if run_experiment "$dataset" "$model" "com_sdn"; then
            ok "OK com_sdn / $dataset / $model"
            EXPERIMENT_COUNT=$((EXPERIMENT_COUNT + 1))
        else
            err "FALHOU: com_sdn / $dataset / $model"
            FAILED_EXPERIMENTS+=("com_sdn_${dataset}_${model}")
        fi

        stop_background_traffic
        log "Pausa de 90s entre experimentos..."
        sleep 90

        # -- Experimento SEM SDN ----------------------------------------------
        cleanup_before_experiment "$dataset" "sem_sdn"

        start_background_traffic

        if run_experiment "$dataset" "$model" "sem_sdn"; then
            ok "OK sem_sdn / $dataset / $model"
            EXPERIMENT_COUNT=$((EXPERIMENT_COUNT + 1))
        else
            err "FALHOU: sem_sdn / $dataset / $model"
            FAILED_EXPERIMENTS+=("sem_sdn_${dataset}_${model}")
        fi

        stop_background_traffic
        log "Pausa de 90s antes do proximo dataset..."
        sleep 90

    done
done

# =============================================================================
# RELATORIO FINAL
# =============================================================================

TOTAL_END=$(date +%s)
TOTAL_MIN=$(( (TOTAL_END - TOTAL_START) / 60 ))

echo ""
echo -e "${BOLD}${BLUE}"
echo "================================================================"
echo "=                    RELATORIO FINAL                          ="
echo "================================================================"
echo -e "${NC}"
log "Inicio:  $(date -d @"$TOTAL_START" '+%Y-%m-%d %H:%M:%S')"
log "Fim:     $(date '+%Y-%m-%d %H:%M:%S')"
log "Duracao: ${TOTAL_MIN} minutos"
echo ""
ok "Experimentos concluidos: $EXPERIMENT_COUNT"

if [ ${#FAILED_EXPERIMENTS[@]} -gt 0 ]; then
    err "Falhas (${#FAILED_EXPERIMENTS[@]}):"
    for f in "${FAILED_EXPERIMENTS[@]}"; do
        echo "  x $f"
    done
fi

echo ""
log "Estrutura de resultados:"
ls -lh "$RESULTS_DIR" 2>/dev/null

echo ""
echo -e "${CYAN}Para gerar graficos comparativos:${NC}"
for ds in "${DATASETS[@]}"; do
    model="${MODELS[0]}"
    echo "  python3 ~/FL-SDN-main/fl_sdn_code/plot_resultados.py \\"
    echo "    --com $RESULTS_DIR/com_sdn_${ds}_${model}_output/*resultados.csv \\"
    echo "    --sem $RESULTS_DIR/sem_sdn_${ds}_${model}_output/*resultados.csv \\"
    echo "    --titulo '$ds'"
    echo ""
done

{
    echo "=== RELATORIO FINAL -- FL-SDN EXPERIMENTOS NOTURNOS ==="
    echo "Inicio:   $(date -d @"$TOTAL_START" '+%Y-%m-%d %H:%M:%S')"
    echo "Fim:      $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Duracao:  ${TOTAL_MIN} minutos"
    echo "Concluidos: $EXPERIMENT_COUNT"
    echo "Falhas: ${FAILED_EXPERIMENTS[*]:-nenhuma}"
    echo ""
    echo "Arquivos gerados:"
    find "$RESULTS_DIR" -name "*.csv" | sort
} > "$RESULTS_DIR/relatorio_final.txt"

ok "Relatorio salvo: $RESULTS_DIR/relatorio_final.txt"
ok "Script concluido."
