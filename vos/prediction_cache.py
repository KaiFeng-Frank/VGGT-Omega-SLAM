"""Lossless front-end caches for CPU-only backend ablations (never pickle)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def save_predictions(path, predictions, frame_names, submap_id):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays, values, tensor_keys, tensor_storage, nested = {}, {}, [], {}, {}

    def store_array(key, value):
        if hasattr(value, "detach"):
            tensor = value.detach().cpu().contiguous()
            if str(tensor.dtype) == "torch.bfloat16":
                import torch
                arrays[key] = tensor.view(torch.uint16).numpy()
                tensor_storage[key] = "bfloat16_bits"
            else:
                arrays[key] = tensor.numpy()
            tensor_keys.append(key)
        else:
            if value.dtype.hasobject:
                raise TypeError(f"Object array cannot be cached: {key}")
            arrays[key] = value

    def store_nested(value):
        if hasattr(value, "detach") or isinstance(value, np.ndarray):
            key = f"__nested_array_{len(arrays)}"
            while key in arrays or key in predictions:
                key += "_"
            store_array(key, value)
            return {"kind": "array", "key": key}
        if isinstance(value, (list, tuple)):
            return {"kind": "tuple" if isinstance(value, tuple) else "list",
                    "items": [store_nested(item) for item in value]}
        if isinstance(value, dict):
            if not all(isinstance(key, str) for key in value):
                raise TypeError("Nested prediction dictionary keys must be strings")
            return {"kind": "dict", "items": {key: store_nested(item) for key, item in value.items()}}
        return {"kind": "value", "value": value}

    for key, value in predictions.items():
        if key == "__metadata__":
            raise ValueError("Reserved prediction cache key: __metadata__")
        if key == "detected_loops":
            values[key] = [loop._asdict() if hasattr(loop, "_asdict") else vars(loop) for loop in value]
        elif hasattr(value, "detach") or isinstance(value, np.ndarray):
            store_array(key, value)
        elif key == "frames_lc_names":
            values[key] = [Path(name).name for name in value]
        elif isinstance(value, (list, tuple, dict)):
            nested[key] = store_nested(value)
        else:
            values[key] = value
    metadata = {"schema": "vos-prediction-cache-v1", "submap_id": int(submap_id),
                "frame_names": [Path(name).name for name in frame_names],
                "tensor_keys": tensor_keys, "values": values,
                "tensor_storage": tensor_storage, "nested_values": nested}
    def encode(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Unsupported cache metadata type: {type(value).__name__}")
    arrays["__metadata__"] = np.array(json.dumps(metadata, default=encode, allow_nan=False))
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def load_predictions(path, restore_torch=True):
    """Restore cached predictions on CPU, including VGGT's nested/bf16 outputs.

    With restore_torch=False, bf16 tensors remain uint16 bit arrays; their
    storage tags in metadata distinguish these from numeric uint16 arrays.
    Older v1 caches without these optional storage/tree fields remain readable.
    """
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["__metadata__"]))
        if metadata["schema"] != "vos-prediction-cache-v1":
            raise ValueError("Unsupported prediction cache schema")
        predictions = {key: archive[key].copy() for key in archive.files if key != "__metadata__"}
    if restore_torch and metadata["tensor_keys"]:
        import torch
        for key in metadata["tensor_keys"]:
            predictions[key] = torch.from_numpy(predictions[key])
            storage = metadata.get("tensor_storage", {}).get(key)
            if storage == "bfloat16_bits":
                predictions[key] = predictions[key].view(torch.bfloat16)
            elif storage is not None:
                raise ValueError(f"Unsupported tensor cache storage: {storage}")

    def restore_nested(node):
        kind = node["kind"]
        if kind == "array":
            return predictions.pop(node["key"])
        if kind in ("list", "tuple"):
            items = [restore_nested(item) for item in node["items"]]
            return tuple(items) if kind == "tuple" else items
        if kind == "dict":
            return {key: restore_nested(item) for key, item in node["items"].items()}
        if kind == "value":
            return node["value"]
        raise ValueError(f"Unsupported nested cache node: {kind}")

    for key, node in metadata.get("nested_values", {}).items():
        predictions[key] = restore_nested(node)
    predictions.update(metadata["values"])
    predictions["detected_loops"] = [SimpleNamespace(**entry) for entry in predictions.get("detected_loops", [])]
    return predictions, metadata
