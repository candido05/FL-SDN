"""
Servidor Flower com conexao gRPC explicita.

Uso:
    python server.py --model xgboost --strategy bagging
    python server.py --model lightgbm --strategy cycling

Logging:
    EXP=com_sdn  python server.py --model xgboost --strategy bagging
    EXP=sem_sdn  python server.py --model xgboost --strategy bagging
"""

import argparse
import logging
import os
import sys
import warnings

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*DEPRECATED.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")
warnings.filterwarnings("ignore", message=".*feature names.*")

import flwr as fl

logging.getLogger("flwr").setLevel(logging.ERROR)
logging.getLogger("grpc").setLevel(logging.ERROR)

from config import SERVER_ADDRESS, NUM_CLIENTS, NUM_ROUNDS, LOCAL_EPOCHS
from core.csv_logger import CSVLogger
from core.output import create_run_dir
from datasets import DatasetRegistry
from strategies import create_strategy


def main():
    parser = argparse.ArgumentParser(description="Servidor FL com conexao gRPC explicita")
    parser.add_argument("--model", type=str, required=True,
                        choices=["xgboost", "lightgbm", "catboost"])
    parser.add_argument("--strategy", type=str, required=True,
                        choices=["bagging", "cycling", "sdn-bagging", "sdn-cycling"])
    parser.add_argument("--dataset", type=str, default="higgs",
                        choices=["higgs", "higgs_full", "epsilon", "avazu"],
                        help="Dataset a utilizar (default: higgs)")
    args = parser.parse_args()

    exp_name = os.environ.get("EXP", "experimento")
    run_dir = create_run_dir(args.model, args.strategy, exp_name)
    logger = CSVLogger(run_dir, exp_name)

    print(f"\n{'='*60}")
    print(f"  SERVIDOR FL - {args.model.upper()} + {args.strategy.upper()}")
    print(f"  Experimento: {logger.exp_name}  →  {logger.log_file}")
    print(f"  Run dir:     {run_dir}")
    print(f"{'='*60}")

    print(f"[Servidor] Carregando dataset {args.dataset}...")
    X_test, y_test = DatasetRegistry.load(args.dataset, role="server")

    strategy = create_strategy(
        name=args.strategy,
        num_clients=NUM_CLIENTS,
        X_test=X_test,
        y_test=y_test,
        logger=logger,
    )

    print(f"\n[Servidor] Configuracao:")
    print(f"    Dataset:     {args.dataset}")
    print(f"    Modelo:      {args.model}")
    print(f"    Estrategia:  {args.strategy}")
    print(f"    Rounds:      {NUM_ROUNDS}")
    print(f"    Clientes:    {NUM_CLIENTS}")
    print(f"    Epocas loc:  {LOCAL_EPOCHS}")
    print(f"    Endereco:    {SERVER_ADDRESS}")
    print(f"    Log CSV:     {logger.log_file}")
    print(f"\n[Servidor] Aguardando {NUM_CLIENTS} cliente(s) conectarem...\n")
    sys.stdout.flush()

    logger.start_timer()

    fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

    total = logger.total_elapsed()
    print(f"\n{'='*60}")
    print(f"  SERVIDOR - TREINAMENTO CONCLUIDO")
    print(f"  Tempo total: {total}s  |  Rounds: {NUM_ROUNDS}  |  Log: {logger.log_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
