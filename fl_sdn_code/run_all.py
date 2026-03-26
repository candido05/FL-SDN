"""
Script auxiliar que lanca servidor + clientes em processos separados.

Integra grid search automatico antes do treinamento federado e
suporta execucao completa overnight sem intervencao.

Uso individual:
    python run_all.py --model xgboost --strategy bagging
    python run_all.py --model xgboost --strategy bagging --dataset higgs_full

Uso completo (todos modelos x todas estrategias x um dataset):
    python run_all.py --run-all --dataset higgs

Uso TOTAL overnight (todos modelos x todas estrategias x todos datasets):
    python run_all.py --run-all --all-datasets

Com grid search automatico antes do FL:
    python run_all.py --run-all --all-datasets --grid-search

Teste rapido:
    python run_all.py --test

Apenas grid search (sem FL):
    python run_all.py --grid-search-only --dataset higgs
    python run_all.py --grid-search-only --all-datasets

Apenas verificacao de particoes:
    python run_all.py --verify-only --dataset higgs
"""

import argparse
import json
import os
import subprocess
import sys
import time

from config import NUM_CLIENTS, NUM_ROUNDS, LOCAL_EPOCHS, CLIENT_CONNECT_ADDRESS

ALL_MODELS = ["xgboost", "lightgbm", "catboost"]
ALL_STRATEGIES = ["bagging", "cycling", "sdn-bagging", "sdn-cycling"]
ALL_DATASETS = ["higgs", "higgs_full", "epsilon", "avazu", "mnist", "creditcard"]

# Diretorio para salvar resultados do grid search
TUNED_DIR = os.path.join(os.path.dirname(__file__), "tuned_params")


# ---------------------------------------------------------------------------
# Grid Search integrado
# ---------------------------------------------------------------------------

def run_grid_search_for_dataset(dataset, sample_size=30000):
    """
    Roda grid search para todos os modelos em um dataset.
    Salva JSONs em tuned_params/<dataset>/.
    Retorna dict {model_type: params_dict}.
    """
    out_dir = os.path.join(TUNED_DIR, dataset)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"# GRID SEARCH — Dataset: {dataset}")
    print(f"# Modelos: {ALL_MODELS}")
    print(f"# Sample size: {sample_size:,}")
    print(f"# Output: {out_dir}/")
    print(f"{'#'*60}\n")
    sys.stdout.flush()

    cmd = [
        sys.executable, os.path.join("tools", "grid_search.py"),
        "--dataset", dataset,
        "--all-models",
        "--sample-size", str(sample_size),
        "--output-dir", out_dir,
    ]

    t0 = time.time()
    ret = subprocess.run(cmd)
    elapsed = time.time() - t0

    if ret.returncode != 0:
        print(f"\n[Grid Search] FALHA para dataset {dataset} (exit={ret.returncode})")
        return None

    # Carrega resultados
    results = {}
    for model in ALL_MODELS:
        json_path = os.path.join(out_dir, f"tuned_{model}.json")
        if os.path.exists(json_path):
            with open(json_path) as f:
                results[model] = json.load(f)
            print(f"  [{model}] Params carregados: {results[model]}")
        else:
            print(f"  [{model}] AVISO: JSON nao encontrado em {json_path}")

    print(f"\n[Grid Search] Dataset {dataset} concluido em {elapsed:.1f}s")
    return results


def load_tuned_params(dataset, model):
    """Carrega params tunados de um JSON salvo, se existir."""
    json_path = os.path.join(TUNED_DIR, dataset, f"tuned_{model}.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Verificacao de particoes
# ---------------------------------------------------------------------------

def run_verify_partitions(dataset):
    """Roda verificacao de particoes para um dataset."""
    print(f"\n[Verificacao] Verificando particoes do dataset {dataset}...")
    cmd = [sys.executable, os.path.join("tools", "verify_partitions.py"), "--dataset", dataset]
    ret = subprocess.run(cmd)
    return ret.returncode == 0


# ---------------------------------------------------------------------------
# Experimento FL
# ---------------------------------------------------------------------------

def run_experiment(model, strategy, dataset, exp_prefix="", tuned_params=None):
    """Executa um experimento: servidor + N clientes."""
    exp_name = f"{exp_prefix}{dataset}" if exp_prefix else dataset

    print(f"\n{'#'*60}")
    print(f"# EXPERIMENTO: {model} + {strategy} + {dataset}")
    print(f"# EXP={exp_name}")
    if tuned_params:
        print(f"# Params tunados: {tuned_params}")
    print(f"{'#'*60}\n")

    env = os.environ.copy()
    env["EXP"] = exp_name
    if tuned_params:
        env["TUNED_PARAMS_JSON"] = json.dumps(tuned_params)

    # Lancar servidor
    server_cmd = [
        sys.executable, "server.py",
        "--model", model,
        "--strategy", strategy,
        "--dataset", dataset,
    ]
    print(f"[Launcher] Iniciando servidor...")
    server_proc = subprocess.Popen(server_cmd, env=env)

    # Tempo de espera depende do dataset
    wait_time = 15 if dataset == "higgs" else 60
    print(f"[Launcher] Aguardando servidor carregar dataset ({wait_time}s)...")
    time.sleep(wait_time)

    # Lancar clientes
    client_procs = []
    for cid in range(NUM_CLIENTS):
        client_cmd = [
            sys.executable, "client.py",
            "--client-id", str(cid),
            "--model", model,
            "--dataset", dataset,
        ]
        print(f"[Launcher] Iniciando cliente {cid}...")
        proc = subprocess.Popen(client_cmd, env=env)
        client_procs.append(proc)
        time.sleep(2)

    # Aguardar todos
    print(f"\n[Launcher] Todos os processos iniciados. Aguardando conclusao...\n")
    server_proc.wait()
    for proc in client_procs:
        proc.wait()

    ret = server_proc.returncode
    status = "OK" if ret == 0 else f"FALHA (exit={ret})"
    print(f"\n[Launcher] {model} + {strategy} + {dataset} → {status}")
    return ret == 0


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_full_pipeline(datasets, do_grid_search=False, do_verify=True,
                      grid_sample_size=30000):
    """
    Pipeline completo: grid search (opcional) + verificacao + todos os experimentos.
    """
    total_experiments = len(ALL_MODELS) * len(ALL_STRATEGIES) * len(datasets)

    print(f"\n{'='*60}")
    print(f" PIPELINE COMPLETO — Execucao Overnight")
    print(f"{'='*60}")
    print(f" Datasets:       {datasets}")
    print(f" Modelos:        {ALL_MODELS}")
    print(f" Estrategias:    {ALL_STRATEGIES}")
    print(f" Experimentos:   {total_experiments}")
    print(f" Clientes:       {NUM_CLIENTS}")
    print(f" Rounds:         {NUM_ROUNDS}")
    print(f" Epocas locais:  {LOCAL_EPOCHS}")
    print(f" Grid search:    {'SIM' if do_grid_search else 'NAO'}")
    print(f" Verificacao:    {'SIM' if do_verify else 'NAO'}")
    print(f"{'='*60}\n")
    sys.stdout.flush()

    t_pipeline = time.time()
    tuned_cache = {}  # {dataset: {model: params}}
    results = []
    current = 0

    for dataset in datasets:
        print(f"\n{'*'*60}")
        print(f"* DATASET: {dataset}")
        print(f"{'*'*60}")

        # 1. Verificacao de particoes
        if do_verify:
            verify_ok = run_verify_partitions(dataset)
            if not verify_ok:
                print(f"[Pipeline] AVISO: Verificacao de particoes falhou para {dataset}")
                print(f"[Pipeline] Continuando mesmo assim...")

        # 2. Grid search (uma vez por dataset, cobre todos os modelos)
        if do_grid_search:
            gs_results = run_grid_search_for_dataset(dataset, grid_sample_size)
            if gs_results:
                tuned_cache[dataset] = gs_results
            else:
                print(f"[Pipeline] Grid search falhou para {dataset}, usando params default")
                tuned_cache[dataset] = {}
        else:
            # Tenta carregar JSONs existentes
            tuned_cache[dataset] = {}
            for model in ALL_MODELS:
                params = load_tuned_params(dataset, model)
                if params:
                    tuned_cache[dataset][model] = params
                    print(f"  [{model}] Params tunados carregados de cache: {params}")

        # 3. Experimentos FL
        for model in ALL_MODELS:
            tuned = tuned_cache.get(dataset, {}).get(model, None)

            for strategy in ALL_STRATEGIES:
                current += 1
                print(f"\n{'='*60}")
                print(f" [{current}/{total_experiments}] {model} + {strategy} + {dataset}")
                elapsed_so_far = time.time() - t_pipeline
                print(f" Tempo decorrido: {elapsed_so_far/60:.1f} min")
                print(f"{'='*60}")
                sys.stdout.flush()

                success = run_experiment(
                    model, strategy, dataset,
                    tuned_params=tuned,
                )
                results.append({
                    "model": model,
                    "strategy": strategy,
                    "dataset": dataset,
                    "success": success,
                    "tuned_params": tuned,
                })

                # Pausa entre experimentos para liberar recursos
                if current < total_experiments:
                    print(f"\n[Launcher] Pausa de 10s entre experimentos...")
                    time.sleep(10)

    # Resumo final
    elapsed = time.time() - t_pipeline
    ok_count = sum(1 for r in results if r["success"])
    fail_count = len(results) - ok_count

    print(f"\n\n{'='*60}")
    print(f" RESUMO FINAL — {len(results)} EXPERIMENTOS")
    print(f" Tempo total: {elapsed/60:.1f} minutos ({elapsed/3600:.1f} horas)")
    print(f" Sucesso: {ok_count} | Falhas: {fail_count}")
    print(f"{'='*60}")

    if fail_count > 0:
        print(f"\n Experimentos com falha:")
        for r in results:
            if not r["success"]:
                print(f"   - {r['model']} + {r['strategy']} + {r['dataset']}")

    print(f"\n Resultados salvos em: output/")
    if tuned_cache:
        print(f" Params tunados salvos em: {TUNED_DIR}/")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Lanca servidor + clientes automaticamente",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--model", type=str, choices=ALL_MODELS,
                        help="Modelo a utilizar")
    parser.add_argument("--strategy", type=str, choices=ALL_STRATEGIES,
                        help="Estrategia a utilizar")
    parser.add_argument("--dataset", type=str, default="higgs",
                        choices=ALL_DATASETS,
                        help="Dataset a utilizar (default: higgs)")
    parser.add_argument("--run-all", action="store_true",
                        help="Executa todos modelos x todas estrategias")
    parser.add_argument("--all-datasets", action="store_true",
                        help="Usado com --run-all: executa para TODOS os datasets")
    parser.add_argument("--grid-search", action="store_true",
                        help="Roda grid search antes dos experimentos FL")
    parser.add_argument("--grid-sample-size", type=int, default=30000,
                        help="Amostras para grid search (default: 30000)")
    parser.add_argument("--grid-search-only", action="store_true",
                        help="Apenas roda grid search (sem FL)")
    parser.add_argument("--verify-only", action="store_true",
                        help="Apenas verifica particoes de dados")
    parser.add_argument("--no-verify", action="store_true",
                        help="Pula verificacao de particoes")
    parser.add_argument("--test", action="store_true",
                        help="Teste rapido: xgboost + bagging + higgs (1 round)")
    args = parser.parse_args()

    # Modo teste rapido
    if args.test:
        print(f"{'='*60}")
        print(f" TESTE RAPIDO — Verificando se o codigo funciona")
        print(f" Modelo: xgboost | Estrategia: bagging | Dataset: higgs")
        print(f"{'='*60}\n")
        success = run_experiment("xgboost", "bagging", "higgs", exp_prefix="teste_")
        if success:
            print(f"\n{'='*60}")
            print(f" TESTE OK — Codigo funcionando corretamente!")
            print(f"{'='*60}")
        else:
            print(f"\n{'='*60}")
            print(f" TESTE FALHOU — Verifique os logs acima")
            print(f"{'='*60}")
        return

    # Modo apenas verificacao
    if args.verify_only:
        datasets = ALL_DATASETS if args.all_datasets else [args.dataset]
        all_ok = True
        for ds in datasets:
            ok = run_verify_partitions(ds)
            if not ok:
                all_ok = False
        sys.exit(0 if all_ok else 1)

    # Modo apenas grid search
    if args.grid_search_only:
        datasets = ALL_DATASETS if args.all_datasets else [args.dataset]
        for ds in datasets:
            run_grid_search_for_dataset(ds, args.grid_sample_size)
        return

    # Modo --run-all (com ou sem grid search)
    if args.run_all:
        datasets = ALL_DATASETS if args.all_datasets else [args.dataset]
        run_full_pipeline(
            datasets,
            do_grid_search=args.grid_search,
            do_verify=not args.no_verify,
            grid_sample_size=args.grid_sample_size,
        )
        return

    # Modo individual
    if not args.model or not args.strategy:
        parser.print_help()
        print(f"\n\nExemplos:")
        print(f"  python run_all.py --model xgboost --strategy bagging")
        print(f"  python run_all.py --run-all --dataset higgs")
        print(f"  python run_all.py --run-all --all-datasets --grid-search")
        print(f"  python run_all.py --grid-search-only --dataset higgs")
        print(f"  python run_all.py --verify-only --all-datasets")
        print(f"  python run_all.py --test")
        return

    # Carrega params tunados se existem
    tuned = load_tuned_params(args.dataset, args.model)
    if tuned:
        print(f"[Launcher] Params tunados carregados: {tuned}")

    print(f"{'='*60}")
    print(f" FL-SDN — Federated Learning + Software-Defined Networking")
    print(f" Modelo:      {args.model} | Estrategia: {args.strategy}")
    print(f" Dataset:     {args.dataset}")
    print(f" Clientes:    {NUM_CLIENTS}")
    print(f" Rounds:      {NUM_ROUNDS}")
    print(f" Epocas loc:  {LOCAL_EPOCHS}")
    print(f" Servidor:    {CLIENT_CONNECT_ADDRESS}")
    if tuned:
        print(f" Tunados:     {tuned}")
    print(f"{'='*60}\n")

    run_experiment(args.model, args.strategy, args.dataset, tuned_params=tuned)

    print(f"\n{'='*60}")
    print(f" CONCLUIDO!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
