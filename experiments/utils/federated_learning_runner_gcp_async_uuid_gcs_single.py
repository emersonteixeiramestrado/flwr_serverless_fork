import os
import uuid
from typing import Any

from tensorflow.keras.utils import set_random_seed
from flwr.common import ndarrays_to_parameters
from flwr.server.strategy import (
    FedAvg,
    FedAdam,
    FedAvgM,
    FedOpt,
    FedMedian,
)

from flwr_serverless.federated_node.async_federated_node import AsyncFederatedNode
from flwr_serverless.federated_node.sync_federated_node import SyncFederatedNode
from flwr_serverless.keras.federated_learning_callback import FlwrFederatedCallback
from flwr_serverless.shared_folder.in_memory_folder import InMemoryFolder
from flwr_serverless.shared_folder.gcs_folder import GCSFolderWithPickle

from experiments.utils.base_experiment_runner import BaseExperimentRunner, Config
from experiments.utils.node_logger_callback import NodeEpochLogger
from experiments.utils.custom_wandb_callback import CustomWandbCallback


class FederatedLearningRunner(BaseExperimentRunner):
    """Runner simplificado: este processo/VM representa **um** nó FL."""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.config: Config = self.config
        self.num_rounds = self.epochs

        # UUID único por processo/execução
        self.node_uuid = str(uuid.uuid4())

        # backend de storage: GCS (prod) ou memória (teste)
        self.storage_backend: Any = self._build_storage_backend()

    # ---------------------------------------------------------------------
    # Storage backend
    # ---------------------------------------------------------------------
    def _build_storage_backend(self) -> Any:
        backend = getattr(self.config, "storage_backend", "memory")
        if backend == "gcs":
            bucket = getattr(self.config, "gcs_bucket", None)
            folder = getattr(self.config, "gcs_folder", None)
            if not bucket:
                raise ValueError(
                    "gcs_bucket must be provided when storage_backend='gcs'"
                )
            directory = f"gs://{bucket}/{folder}" if folder else f"gs://{bucket}"
            return GCSFolderWithPickle(directory=directory)
        return InMemoryFolder()

    # ---------------------------------------------------------------------
    # Execução principal
    # ---------------------------------------------------------------------
    def run(self):
        print("Effective Config in GCP runner:", self.config.__dict__)
        config: Config = self.config
        if config.random_seed is not None:
            set_random_seed(config.random_seed)

        if config.track:
            import wandb

            strategy = self.config.strategy
            data_split = self.config.data_split
            sync_or_async: str = "async" if self.config.use_async else "sync"
            name = (
                f"{sync_or_async}_{strategy}_single_node_"
                f"{data_split}_uuid_{self.node_uuid[:8]}"
            )
            if data_split == "skewed":
                name += f"_{self.config.skew_factor}"
            wandb.init(
                project=self.config.project,
                entity=os.getenv("WANDB_ENTITY", "example_entity"),
                name=name,
                config=config.__dict__,
            )

        # apenas 1 modelo local
        self.models = self.create_models()
        assert len(self.models) == 1, "Este runner assume apenas 1 modelo local"
        self.model = self.models[0]

        self.set_strategy()

        # usa API de split baseada em índices da classe base
        (
            self.partitioned_x_train,
            self.partitioned_y_train,
            self.x_test,
            self.y_test,
        ) = self.split_data()

        print("x_test shape:", self.x_test.shape)
        print("y_test shape:", self.y_test.shape)
        print(f"Async node UUID for this process: {self.node_uuid}")

        self.train_single_node()
        self.evaluate()

        if config.track:
            import wandb

            wandb.finish()

    # ---------------------------------------------------------------------
    # Estratégia FL (apenas uma por nó)
    # ---------------------------------------------------------------------
    def set_strategy(self):
        if self.strategy_name == "fedavg":
            self.strategy = FedAvg()
        elif self.strategy_name == "fedavgm":
            self.strategy = FedAvgM()
        elif self.strategy_name == "fedadam":
            self.strategy = FedAdam(
                initial_parameters=ndarrays_to_parameters(self.model.get_weights())
            )
        elif self.strategy_name == "fedopt":
            self.strategy = FedOpt(
                initial_parameters=ndarrays_to_parameters(self.model.get_weights())
            )
        elif self.strategy_name == "fedmedian":
            self.strategy = FedMedian()
        else:
            raise ValueError(f"Strategy not supported: {self.strategy_name}")

    # ---------------------------------------------------------------------
    # Split de dados usando BaseExperimentRunner
    # ---------------------------------------------------------------------
    def split_data(self):
        """Encaminha para os modos de split previstos na base."""
        config: Config = self.config

        if self.data_split == "random":
            # alias na classe base: usa create_partitioned_datasets()
            return self.random_split()
        elif self.data_split == "partitioned":
            # se você quiser um modo "partitioned", implemente na base
            return self.create_partitioned_datasets()
        elif self.data_split == "skewed":
            # alias na classe base: usa create_partitioned_datasets()
            return self.create_skewed_partition_split(skew_factor=config.skew_factor)
        else:
            raise ValueError("Data split not supported")

    # ---------------------------------------------------------------------
    # Criação de nó FL
    # ---------------------------------------------------------------------
    def create_node(self):
        """Cria um único nó FL (assíncrono ou síncrono) para este processo."""
        if self.use_async:
            return AsyncFederatedNode(
                shared_folder=self.storage_backend,
                strategy=self.strategy,
                node_id=self.node_uuid,
            )
        return SyncFederatedNode(
            shared_folder=self.storage_backend,
            strategy=self.strategy,
            num_nodes=1,
        )

    # ---------------------------------------------------------------------
    # Treino local de um único nó
    # ---------------------------------------------------------------------
    def train_single_node(self):
        node = self.create_node()

        # Usa partição 0 deste nó único
        x_train = self.partitioned_x_train[0]
        y_train = self.partitioned_y_train[0]

        # lê faixa de delay configurada a partir do speed_group
        delay_s_min = getattr(self.config, "delay_s_min", 0)
        delay_s_max = getattr(self.config, "delay_s_max", 2)
        
        

        for round_idx in range(self.num_rounds):
            callbacks = [
                FlwrFederatedCallback(
                    node,
                    num_examples_per_epoch=(self.steps_per_epoch * self.batch_size),
                    global_epoch=round_idx,
                ),
                NodeEpochLogger(
                    node_id=self.node_uuid,
                    min_delay_s=delay_s_min,
                    max_delay_s=delay_s_max,
                ),
            ]
            if self.config.track:
                callbacks.append(CustomWandbCallback(0))

            self.model.fit(
                x=x_train,
                y=y_train,
                epochs=1,
                steps_per_epoch=self.steps_per_epoch,
                callbacks=callbacks,
                verbose=0,
            )
            print(f"[node={self.node_uuid[:8]}] finished local epoch {round_idx}")

    # ---------------------------------------------------------------------
    # Avaliação
    # ---------------------------------------------------------------------
    def evaluate(self):
        loss, acc = self.model.evaluate(
            self.x_test,
            self.y_test,
            batch_size=self.batch_size,
            steps=self.test_steps,
            verbose=0,
        )
        print(
            f"[node {self.node_uuid[:8]}] "
            f"test_loss: {loss:.4f} | test_accuracy: {acc:.4f}"
        )