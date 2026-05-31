import pickle
import time
from typing import Any, Optional

from google.cloud import storage


class GCSFolderWithPickle:
    def __init__(
        self,
        bucket_name: Optional[str] = None,
        folder: Optional[str] = None,
        directory: Optional[str] = None,
        retry_sleep_time: int = 3,
        max_retry: int = 3,
        check_at_init: bool = True,
    ):
        self.bucket_name, self.prefix = self._resolve_path(
            bucket_name=bucket_name,
            folder=folder,
            directory=directory,
        )
        self.directory = directory or self._build_directory(self.bucket_name, self.prefix)
        self.suffix = ".pkl"
        self.retry_sleep_time = retry_sleep_time
        self.max_retry = max_retry

        self.client = storage.Client()
        self.bucket = self.client.bucket(self.bucket_name)

        if check_at_init:
            self._check()

    @staticmethod
    def _resolve_path(
        bucket_name: Optional[str],
        folder: Optional[str],
        directory: Optional[str],
    ):
        if directory:
            if directory.startswith("gs://"):
                directory = directory[5:]
            parts = directory.split("/", 1)
            parsed_bucket = parts[0]
            parsed_prefix = parts[1].rstrip("/") if len(parts) > 1 else None
            return parsed_bucket, parsed_prefix

        if not bucket_name:
            raise ValueError("bucket_name must be provided when directory is not informed")

        prefix = folder.rstrip("/") if folder else None
        return bucket_name, prefix

    @staticmethod
    def _build_directory(bucket_name: str, prefix: Optional[str]) -> str:
        if prefix:
            return f"gs://{bucket_name}/{prefix}"
        return f"gs://{bucket_name}"

    def _build_path(self, key: str) -> str:
        if self.prefix is None:
            return key
        return f"{self.prefix}/{key}"

    def _check(self):
        timestamp_ms = int(time.time() * 1000)
        key = f"dummy_{timestamp_ms}"
        self[key] = "dummy"
        assert self[key] == "dummy"
        del self[key]

    def _exists(self, key: str) -> bool:
        return self.bucket.blob(key).exists()

    def get(self, key, default=None):
        success_flag_file = self._get_success_flag_file(key)
        patience = self.max_retry

        while not self._exists(success_flag_file):
            print(f"\nwaiting for success flag of {key}")
            time.sleep(self.retry_sleep_time)
            patience -= 1
            if patience == 0:
                return default

        filepath = self._build_path(key + self.suffix)
        if self._exists(filepath):
            data = self.bucket.blob(filepath).download_as_bytes()
            return pickle.loads(data)
        return default

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value: Any):
        if value is None:
            raise ValueError("value must not be None")

        filepath = self._build_path(key + self.suffix)
        self._delete_success_flag(key)
        self.bucket.blob(filepath).upload_from_string(pickle.dumps(value))
        self._put_success_flag(key)

    def __delitem__(self, key):
        self.bucket.blob(self._build_path(key + self.suffix)).delete(if_generation_match=None)

    def _get_success_flag_file(self, key):
        return self._build_path(f"{key}.success")

    def _delete_success_flag(self, key):
        filepath = self._get_success_flag_file(key)
        if self._exists(filepath):
            self.bucket.blob(filepath).delete(if_generation_match=None)

    def _put_success_flag(self, key):
        self.bucket.blob(self._get_success_flag_file(key)).upload_from_string(b"")

    def __len__(self):
        return len(self._list_pickle_files())

    def _list_pickle_files(self):
        return [
            blob.name
            for blob in self.client.list_blobs(self.bucket_name, prefix=self.prefix)
            if blob.name.endswith(self.suffix)
        ]

    def items(self):
        for filepath in self._list_pickle_files():
            yield self.get_parameter(filepath)

    def get_parameter(self, filepath):
        model_key = filepath.split("/")[-1].replace(self.suffix, "")
        return model_key, self.get(model_key)
    
    def get_raw_folder(self):
        """Retorna o próprio objeto como 'raw folder' para salvar métricas."""
        return self