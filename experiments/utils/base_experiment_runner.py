import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any

from experiments.model.simple_mnist_model import SimpleMnistModel
from experiments.model.keras_models import ResNetModelBuilder


@dataclass
class Config:
    # ------------------------
    # Parâmetros não-compartilhados
    # ------------------------
    num_nodes: int = 1
    strategy: str = "fedavg"
    project: str = "experiments"
    track: bool = False
    random_seed: int = 0

    # ------------------------
    # Parâmetros compartilhados de FL
    # ------------------------
    use_async: bool = True
    federated_type: str = "concurrent"
    dataset: str = "mnist"
    epochs: int = 100
    batch_size: int = 32
    steps_per_epoch: int = 64
    lr: float = 0.001
    test_steps: Optional[int] = None
    net: str = "simple"
    data_split: str = "skewed"
    skew_factor: float = 0.9

    # ------------------------
    # Backend de storage
    # ------------------------
    storage_backend: str = "memory"
    gcs_bucket: str = ""
    gcs_folder: str = ""
    storage_retry_sleep_time: int = 3
    storage_max_retry: int = 300
    storage_check_at_init: bool = True

    # ------------------------
    # Perfil de velocidade / delay
    # ------------------------
    speed_group: str = "fast"
    delay_s_min: float = 0.0
    delay_s_max: float = 2.0

    # ------------------------
    # Diversos
    # ------------------------
    use_default_configs: bool = False


class BaseExperimentRunner:
    def __init__(self, config, tracking=False):
        if isinstance(config, dict):
            config = Config(**config)
        assert isinstance(
            config, Config
        ), f"config must be of type Config, got {type(config)}"

        self.config = config
        self.num_nodes = config.num_nodes
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.steps_per_epoch = config.steps_per_epoch
        self.lr = config.lr
        self.test_steps = config.test_steps
        self.use_async = config.use_async
        self.federated_type = config.federated_type
        self.strategy_name = config.strategy
        self.data_split = config.data_split
        self.dataset = config.dataset
        self.net = config.net
        self.tracking = tracking

        # dados e índices
        self.x_train: np.ndarray = None
        self.y_train: np.ndarray = None
        self.x_test: np.ndarray = None
        self.y_test: np.ndarray = None

        # índices por partição (sem duplicar imagens)
        self.partition_indices: List[np.ndarray] = []

        self.get_original_data()

    # ---------------------------------------------------------------------
    # MODELOS
    # ---------------------------------------------------------------------
    # ***currently works only for mnist***
    def create_models(self):
        if self.dataset == "mnist":
            assert self.net == "simple", f"Net not supported: {self.net} for mnist"
        if self.net == "simple":
            return [SimpleMnistModel(lr=self.lr).run() for _ in range(self.num_nodes)]
        elif self.net == "resnet50":
            return [
                ResNetModelBuilder(lr=self.lr, net="ResNet50", weights="imagenet").run()
                for _ in range(self.num_nodes)
            ]
        elif self.net == "resnet18":
            return [
                ResNetModelBuilder(lr=self.lr, net="ResNet18").run()
                for _ in range(self.num_nodes)
            ]
        else:
            raise ValueError(f"Unsupported net: {self.net}")

    # ---------------------------------------------------------------------
    # CARREGAMENTO E NORMALIZAÇÃO (apenas UMA cópia float32)
    # ---------------------------------------------------------------------
    def get_original_data(self):
        """Carrega o dataset e já normaliza para float32 in-place, guardando
        apenas uma cópia em memória."""
        if self.dataset == "mnist":
            from tensorflow.keras.datasets import mnist

            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            # MNIST original: (N, 28, 28), uint8
            x_train = x_train.astype(np.float32) / 255.0
            x_test = x_test.astype(np.float32) / 255.0

            # adiciona canal 1
            x_train = np.expand_dims(x_train, -1)  # (N, 28, 28, 1)
            x_test = np.expand_dims(x_test, -1)

            self.x_train, self.x_test = x_train, x_test
            self.y_train, self.y_test = y_train.astype(np.int64), y_test.astype(
                np.int64
            )

        elif self.dataset == "cifar10":
            from tensorflow.keras.datasets import cifar10

            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            # CIFAR10 original: (N, 32, 32, 3), uint8
            x_train = x_train.astype(np.float32) / 255.0
            x_test = x_test.astype(np.float32) / 255.0

            self.x_train, self.x_test = x_train, x_test  # (N, 32, 32, 3)
            self.y_train = np.squeeze(y_train, -1).astype(np.int64)
            self.y_test = np.squeeze(y_test, -1).astype(np.int64)
        else:
            raise ValueError(f"Dataset not supported: {self.dataset}")

        assert len(self.y_train.shape) == 1, f"y_train shape: {self.y_train.shape}"
        assert len(self.y_test.shape) == 1, f"y_test shape: {self.y_test.shape}"

    # ---------------------------------------------------------------------
    # UTILIDADES DE SPLIT BASEADAS EM ÍNDICES
    # ---------------------------------------------------------------------
    def _random_indices_split(self) -> List[np.ndarray]:
        """Gera índices aleatórios divididos em num_nodes partições."""
        num_samples = self.x_train.shape[0]
        indices = np.random.permutation(num_samples)
        return np.array_split(indices, self.num_nodes)

    def _skewed_indices_split(
        self, skew_factor: float = 0.8, num_classes: int = 10
    ) -> List[np.ndarray]:
        """Gera partições 'skewed' mas sempre guardando só índices.

        0.8 => ~80% dos exemplos de classes de um grupo vão para uma partição
        preferencial, 20% vão aleatoriamente para outra.
        """
        # índices por classe
        indices_by_label: List[List[int]] = [[] for _ in range(num_classes)]
        for idx, label in enumerate(self.y_train):
            indices_by_label[int(label)].append(idx)

        # classes por partição
        splitted_classes = np.array_split(np.arange(num_classes), self.num_nodes)
        print("splitted_classes", splitted_classes)

        def class_partition(class_idx: int) -> int:
            for i, part in enumerate(splitted_classes):
                if class_idx in part:
                    return i
            raise RuntimeError("Class not found in any partition")

        part_indices: List[List[int]] = [[] for _ in range(self.num_nodes)]

        rng = np.random.default_rng(self.config.random_seed)

        for c in range(num_classes):
            class_idxs = indices_by_label[c]
            for idx in class_idxs:
                target_part = class_partition(c)
                if rng.random() < skew_factor:
                    part_indices[target_part].append(idx)
                else:
                    # joga aleatoriamente em outra partição
                    random_part = int(rng.integers(0, self.num_nodes))
                    part_indices[random_part].append(idx)

        # embaralha cada partição e converte para np.array
        part_indices_np: List[np.ndarray] = []
        for i in range(self.num_nodes):
            p = np.array(part_indices[i], dtype=np.int64)
            perm = rng.permutation(p.shape[0])
            p = p[perm]
            part_indices_np.append(p)

            # debug de distribuição
            print(f"Partition {i}:")
            for c in range(num_classes):
                print(f"  Label {c}: {(self.y_train[p] == c).sum()}")

        return part_indices_np

    # ---------------------------------------------------------------------
    # API PRINCIPAL DE SPLIT
    # ---------------------------------------------------------------------
    def create_partitioned_datasets(
        self,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
        """Cria partições usando apenas índices, sem duplicar imagens.

        Retorna:
          - partitioned_x_train: lista de 'views' de x_train por partição
          - partitioned_y_train: lista de 'views' de y_train por partição
          - x_test, y_test originais (normalizados)
        """
        if self.data_split == "random":
            self.partition_indices = self._random_indices_split()
        elif self.data_split == "skewed":
            self.partition_indices = self._skewed_indices_split(
                skew_factor=self.config.skew_factor, num_classes=10
            )
        else:
            raise ValueError(f"Unsupported data_split: {self.data_split}")

        partitioned_x_train: List[np.ndarray] = []
        partitioned_y_train: List[np.ndarray] = []

        for idxs in self.partition_indices:
            partitioned_x_train.append(self.x_train[idxs])
            partitioned_y_train.append(self.y_train[idxs])

        return partitioned_x_train, partitioned_y_train, self.x_test, self.y_test

    # ---------------------------------------------------------------------
    # DATALOADER POR NÓ (usa índices)
    # ---------------------------------------------------------------------
    def get_train_dataloader_for_node(self, node_idx: int):
        """Iterador infinito de batches para um nó específico.

        Aqui usamos self.partition_indices para buscar batches em self.x_train
        e self.y_train sem criar cópias adicionais.
        """
        assert (
            self.partition_indices
        ), "create_partitioned_datasets deve ser chamado antes"

        idxs = self.partition_indices[node_idx]
        num_samples = idxs.shape[0]

        while True:
            # permuta a cada epoch local
            perm = np.random.permutation(num_samples)
            idxs_epoch = idxs[perm]

            for i in range(0, num_samples, self.batch_size):
                batch_idxs = idxs_epoch[i : i + self.batch_size]
                x_batch = self.x_train[batch_idxs]
                y_batch = self.y_train[batch_idxs]
                yield x_batch, y_batch