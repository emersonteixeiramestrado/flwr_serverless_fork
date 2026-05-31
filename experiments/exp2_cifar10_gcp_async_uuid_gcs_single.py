import os
import logging

logging.getLogger("flwr_serverless").setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from .utils.federated_learning_runner_gcp_async_uuid_gcs_single import (
    FederatedLearningRunner,
)


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


if __name__ == "__main__":
    from argparse import ArgumentParser
    from dotenv import load_dotenv

    load_dotenv()

    parser = ArgumentParser(
        description=(
            "Run federated learning on CIFAR-10 with "
            "AsyncFederatedNode and GCS backend "
            "(UUID por processo, 1 nó por VM)."
        )
    )

    base_config = {
        "project": "cifar10_gcp_async_uuid_gcs",
        "epochs": 20,
        "batch_size": 128,
        "steps_per_epoch": 100,
        "lr": 0.0005,
        "use_async": True,
        "federated_type": "concurrent",  # mantido para compatibilidade
        "dataset": "cifar10",
        "strategy": "fedavg",
        "data_split": "random",
        "skew_factor": 0.9,
        "test_steps": None,
        "net": "resnet18",
        "random_seed": 0,
        "track": False,
        "storage_backend": "gcs",
        "gcs_bucket": "",
        "gcs_folder": "flwr-serverless/cifar10",
        "storage_retry_sleep_time": 3,
        "storage_max_retry": 300,
        "storage_check_at_init": True,
    }

    for key, value in base_config.items():
        if value is None:
            parser.add_argument(f"--{key}", default=value)
        elif isinstance(value, bool):
            parser.add_argument(f"--{key}", type=_parse_bool, default=value)
        elif isinstance(value, int):
            parser.add_argument(f"--{key}", type=int, default=value)
        elif isinstance(value, float):
            parser.add_argument(f"--{key}", type=float, default=value)
        else:
            parser.add_argument(f"--{key}", type=str, default=value)

    args = parser.parse_args()
    config = vars(args)

    if config["storage_backend"] == "gcs" and not config["gcs_bucket"]:
        raise ValueError("gcs_bucket must be informed when storage_backend='gcs'")

    print(
        "Starting async UUID single-node run -> "
        f"storage={config['storage_backend']}://"
        f"{config['gcs_bucket']}/{config['gcs_folder']}"
    )

    runner = FederatedLearningRunner(config=config)
    runner.run()