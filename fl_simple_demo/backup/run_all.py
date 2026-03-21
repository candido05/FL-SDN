"""
Script auxiliar que lanca servidor + clientes em processos separados.

Uso:
    python run_all.py --model xgboost --strategy bagging
    python run_all.py --model lightgbm --strategy cycling
    python run_all.py --model catboost --strategy bagging
"""

import argparse
import subprocess
import sys
import time

from config import NUM_CLIENTS, NUM_ROUNDS, LOCAL_EPOCHS, SERVER_PORT, CLIENT_CONNECT_ADDRESS


def main():
    parser = argparse.ArgumentParser(description="Lanca servidor + clientes automaticamente")
    parser.add_argument("--model", type=str, required=True,
                        choices=["xgboost", "lightgbm", "catboost"])
    parser.add_argument("--strategy", type=str, required=True,
                        choices=["bagging", "cycling"])
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f" FL SIMPLE DEMO")
    print(f" Modelo:      {args.model} | Estrategia: {args.strategy}")
    print(f" Clientes:    {NUM_CLIENTS}")
    print(f" Rounds:      {NUM_ROUNDS}")
    print(f" Epocas loc:  {LOCAL_EPOCHS}")
    print(f" Servidor:    {CLIENT_CONNECT_ADDRESS}")
    print(f"{'='*60}\n")

    # Lancar servidor
    server_cmd = [
        sys.executable, "server.py",
        "--model", args.model,
        "--strategy", args.strategy,
    ]
    print(f"[Launcher] Iniciando servidor...")
    server_proc = subprocess.Popen(server_cmd)

    # Esperar servidor e dataset carregarem
    # O Higgs pode demorar na primeira vez (download)
    print(f"[Launcher] Aguardando servidor carregar dataset (15s)...")
    time.sleep(15)

    # Lancar clientes
    client_procs = []
    for cid in range(NUM_CLIENTS):
        client_cmd = [
            sys.executable, "client.py",
            "--client-id", str(cid),
            "--model", args.model,
        ]
        print(f"[Launcher] Iniciando cliente {cid}...")
        proc = subprocess.Popen(client_cmd)
        client_procs.append(proc)
        time.sleep(2)

    # Aguardar todos
    print(f"\n[Launcher] Todos os processos iniciados. Aguardando conclusao...\n")
    server_proc.wait()
    for proc in client_procs:
        proc.wait()

    print(f"\n{'='*60}")
    print(f" CONCLUIDO!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
