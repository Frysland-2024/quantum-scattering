from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare(expected: Path, actual: Path) -> None:
    with Image.open(expected) as image_a, Image.open(actual) as image_b:
        a = np.asarray(image_a.convert("RGB"), dtype=np.int16)
        b = np.asarray(image_b.convert("RGB"), dtype=np.int16)
    print(f"expected={expected}")
    print(f"actual={actual}")
    print(f"sha_equal={sha256(expected) == sha256(actual)}")
    print(f"expected_shape={a.shape} actual_shape={b.shape}")
    if a.shape != b.shape:
        print("pixel_metrics=not_comparable")
        return
    delta = np.abs(a - b)
    changed = np.any(delta != 0, axis=2)
    print(f"changed_pixel_fraction={changed.mean():.9f}")
    print(f"mean_abs_channel_delta={delta.mean():.9f}")
    print(f"max_abs_channel_delta={delta.max()}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: compare_part4_images.py EXPECTED ACTUAL")
    compare(Path(sys.argv[1]), Path(sys.argv[2]))
