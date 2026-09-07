

import csv
import json
import os
import random
import time
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_run_dir(root: str, mode: str, name: str, seed: int,
                 timestamp: Optional[str] = None) -> str:
    stamp = timestamp or time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(root, mode, "%s_seed%d_%s" % (name, seed, stamp))
    os.makedirs(os.path.join(path, "checkpoints"), exist_ok=True)
    return path


def save_json(path: str, payload: Dict[str, Any]) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_default)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return str(value)


class CsvLogger:
    """Append-only CSV writer that fixes its columns from the first row."""

    def __init__(self, path: str):
        self.path = path
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._columns: Optional[List[str]] = None
        self._handle = None
        self._writer = None

    def write(self, row: Dict[str, Any]) -> None:
        if self._handle is None:
            self._columns = list(row.keys())
            self._handle = open(self.path, "w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._handle, fieldnames=self._columns)
            self._writer.writeheader()
        clean = {key: _scalar(row.get(key)) for key in self._columns}
        self._writer.writerow(clean)
        self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "CsvLogger":
        return self

    def __exit__(self, *args) -> None:
        self.close()


def _scalar(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return int(bool(value))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if value is None:
        return ""
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return ""
    return value


def save_checkpoint(path: str, payload: Dict[str, Any]) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)
