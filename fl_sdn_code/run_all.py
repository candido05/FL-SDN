"""
Script auxiliar que lanca servidor + clientes em processos separados.

Uso individual:
    python run_all.py --model xgboost --strategy bagging
    python run_all.py --model xgboost --strategy bagging --dataset higgs_full
    python run_all.py --model catboost --strategy sdn-bagging --dataset epsilon

Uso completo (todos modelos × todas estrategias × um dataset):
    python run_all.py --run-all --dataset higgs
    python run_all.py --run-all --dataset higgs_full

Uso TOTAL (todos modelos × todas estrategias × todos datasets):
    python run_all.py --run-all --all-datasets

Teste rapido (verifica se tudo funciona com higgs reduzido):
    python run_all.py --test
"""

import argparse
import subprocess
import sys
import time

from config import NUM_CLIENTS, NUM_ROUNDS, LOCAL_EPOCHS, CLIENT_CONNECT_ADDRESS

ALL_MODELS = ["xgboost", "lightgbm", "catboost"]
ALL_STRATEGIES = ["bagging", "cycling", "sdn-bagging", "sdn-cycling"]
ALL_DATASETS = ["higgs", "higgs_full", "epsilon", "avazu"]


def run_experiment(model, strategy, dataset, exp_prefix=""):
    """Executa um experimento: servidor + N clientes."""
    exp_name = f"{exp_prefix}{dataset}" if exp_prefix else dataset

    print(f"\n{'#'*60}")
    print(f"# EXPERIMENTO: {model} + {strategy} + {dataset}")
    print(f"# EXP={exp_name}")
    print(f"{'#'*60}\n")

    import os
    env = os.environ.copy()
    env["EXP"] = exp_name

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
                        help="Executa todos modelos × todas estrategias para o dataset escolhido")
    parser.add_argument("--all-datasets", action="store_true",
                        help="Usado com --run-all: executa para TODOS os datasets")
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

    # Modo --run-all
    if args.run_all:
        datasets = ALL_DATASETS if args.all_datasets else [args.dataset]
        total = len(ALL_MODELS) * len(ALL_STRATEGIES) * len(datasets)
        current = 0
        results = []

        print(f"{'='*60}")
        print(f" EXECUCAO COMPLETA")
        print(f" Modelos:     {ALL_MODELS}")
        print(f" Estrategias: {ALL_STRATEGIES}")
        print(f" Datasets:    {datasets}")
        print(f" Total:       {total} experimentos")
        print(f" Clientes:    {NUM_CLIENTS}")
        print(f" Rounds:      {NUM_ROUNDS}")
        print(f"{'='*60}\n")

        t_start = time.time()

        for dataset in datasets:
            for model in ALL_MODELS:
                for strategy in ALL_STRATEGIES:
                    current += 1
                    print(f"\n{'='*60}")
                    print(f" [{current}/{total}] {model} + {strategy} + {dataset}")
                    print(f"{'='*60}")

                    success = run_experiment(model, strategy, dataset)
                    results.append({
                        "model": model,
                        "strategy": strategy,
                        "dataset": dataset,
                        "success": success,
                    })

                    # Pausa entre experimentos para liberar recursos
                    if current < total:
                        print(f"\n[Launcher] Pausa de 10s entre experimentos...")
                        time.sleep(10)

        # Resumo final
        elapsed = time.time() - t_start
        ok_count = sum(1 for r in results if r["success"])
        fail_count = total - ok_count

        print(f"\n\n{'='*60}")
        print(f" RESUMO — {total} EXPERIMENTOS")
        print(f" Tempo total: {elapsed/60:.1f} minutos")
        print(f" Sucesso: {ok_count} | Falhas: {fail_count}")
        print(f"{'='*60}")

        if fail_count > 0:
            print(f"\n Experimentos com falha:")
            for r in results:
                if not r["success"]:
                    print(f"   - {r['model']} + {r['strategy']} + {r['dataset']}")

        print(f"\n Resultados salvos em: output/")
        return

    # Modo individual
    if not args.model or not args.strategy:
        parser.print_help()
        print(f"\n\nExemplos:")
        print(f"  python run_all.py --model xgboost --strategy bagging")
        print(f"  python run_all.py --model xgboost --strategy sdn-bagging --dataset higgs_full")
        print(f"  python run_all.py --run-all --dataset higgs")
        print(f"  python run_all.py --run-all --all-datasets")
        print(f"  python run_all.py --test")
        return

    print(f"{'='*60}")
    print(f" FL-SDN — Federated Learning + Software-Defined Networking")
    print(f" Modelo:      {args.model} | Estrategia: {args.strategy}")
    print(f" Dataset:     {args.dataset}")
    print(f" Clientes:    {NUM_CLIENTS}")
    print(f" Rounds:      {NUM_ROUNDS}")
    print(f" Epocas loc:  {LOCAL_EPOCHS}")
    print(f" Servidor:    {CLIENT_CONNECT_ADDRESS}")
    print(f"{'='*60}\n")

    run_experiment(args.model, args.strategy, args.dataset)

    print(f"\n{'='*60}")
    print(f" CONCLUIDO!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
