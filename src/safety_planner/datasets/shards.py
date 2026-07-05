from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..interfaces import TensorizedScene, stack_tensorized_scenes


def deterministic_subsets(
    scenario_ids: list[str],
    train_count: int = 1000,
    val_count: int = 200,
    closed_loop_count: int = 20,
    seed: int = 42,
) -> dict[str, list[str]]:
    unique_ids = sorted(set(scenario_ids))
    required = train_count + val_count + closed_loop_count
    if len(unique_ids) < required:
        raise ValueError(f"Need {required} unique scenarios, got {len(unique_ids)}.")
    ranked = sorted(
        unique_ids,
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest(),
    )
    return {
        "train": ranked[:train_count],
        "val": ranked[train_count : train_count + val_count],
        "closed_loop": ranked[train_count + val_count : required],
    }


def write_npz_shards(
    scenes: list[TensorizedScene],
    output_dir: str | Path,
    split: str,
    shard_size: int = 512,
    config_hash: str = "",
) -> list[dict]:
    if shard_size <= 0:
        raise ValueError("shard_size must be positive.")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for shard_index, offset in enumerate(range(0, len(scenes), shard_size)):
        chunk = scenes[offset : offset + shard_size]
        arrays = stack_tensorized_scenes(chunk)
        filename = f"{split}-{shard_index:05d}.npz"
        np.savez(root / filename, **arrays)
        for item_index, scene in enumerate(chunk):
            rows.append(
                {
                    "schema_version": scene.schema_version,
                    "scenario_id": scene.scenario_id,
                    "timestamp_us": scene.timestamp_us,
                    "split": split,
                    "shard": filename,
                    "index": item_index,
                    "config_hash": config_hash,
                }
            )
    (root / f"{split}-manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return rows


def write_supervised_npz_shards(
    scenes: list[TensorizedScene],
    targets: dict[str, np.ndarray],
    output_dir: str | Path,
    split: str,
    shard_size: int = 512,
    config_hash: str = "",
) -> list[dict]:
    for name, value in targets.items():
        if value.shape[0] != len(scenes):
            raise ValueError(f"Target {name!r} first dimension must match scenes.")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for shard_index, offset in enumerate(range(0, len(scenes), shard_size)):
        chunk = scenes[offset : offset + shard_size]
        arrays = stack_tensorized_scenes(chunk)
        arrays.update(
            {
                name: value[offset : offset + len(chunk)]
                for name, value in targets.items()
            }
        )
        filename = f"{split}-{shard_index:05d}.npz"
        np.savez(root / filename, **arrays)
        for item_index, scene in enumerate(chunk):
            rows.append(
                {
                    "schema_version": scene.schema_version,
                    "scenario_id": scene.scenario_id,
                    "timestamp_us": scene.timestamp_us,
                    "split": split,
                    "shard": filename,
                    "index": item_index,
                    "config_hash": config_hash,
                }
            )
    (root / f"{split}-manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return rows


def write_manifest_parquet(rows: list[dict], path: str | Path) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to write the Parquet manifest.") from exc
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), output)


class NpzShardDataset:
    """Map-style dataset that keeps only the current NPZ shard in memory."""

    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.rows = [
            json.loads(line)
            for line in self.manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self._cached_name: str | None = None
        self._cached_arrays: dict[str, np.ndarray] | None = None

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, item: int) -> dict:
        row = self.rows[item]
        shard_name = str(row["shard"])
        if shard_name != self._cached_name:
            with np.load(self.root / shard_name) as source:
                self._cached_arrays = {name: source[name].copy() for name in source.files}
            self._cached_name = shard_name
        assert self._cached_arrays is not None
        index = int(row["index"])
        result = {name: value[index] for name, value in self._cached_arrays.items()}
        result["manifest"] = row
        return result
