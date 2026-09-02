from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageSequence


def inspect_image(path: Path) -> dict[str, object]:
    with Image.open(path) as image:
        frames = 0
        sizes: set[tuple[int, int]] = set()
        for frame in ImageSequence.Iterator(image):
            frame.load()
            frames += 1
            sizes.add(frame.size)
        return {
            "path": str(path),
            "format": image.format,
            "frames": frames,
            "sizes": sorted(sizes),
            "mode": image.mode,
        }


def inspect_json(path: Path) -> dict[str, object]:
    return {"path": str(path), "json": json.loads(path.read_text(encoding="utf-8"))}


def main(paths: list[str]) -> None:
    for raw_path in paths:
        path = Path(raw_path)
        if path.suffix.lower() == ".json":
            payload = inspect_json(path)
        else:
            payload = inspect_image(path)
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1:])
