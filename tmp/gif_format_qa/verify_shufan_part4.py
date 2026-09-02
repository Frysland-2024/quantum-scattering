from __future__ import annotations

from pathlib import Path

from PIL import Image


ROOT = Path(r"E:\量子散射\Shufan ZHANG 999018435\codes and figures\outputs\part4")
OUT = Path(r"E:\量子散射\tmp\gif_format_qa\shufan")
OUT.mkdir(parents=True, exist_ok=True)

CASES = {
    "first_resonance": 14,
    "first_resonance_local": 35,
    "second_resonance": 14,
    "second_resonance_local": 11,
}


def main() -> None:
    for stem, frame_index in CASES.items():
        path = ROOT / f"part4_{stem}.gif"
        with Image.open(path) as image:
            assert image.size == (1200, 500), (path, image.size)
            assert image.n_frames == 40, (path, image.n_frames)
            image.seek(frame_index)
            frame = image.convert("RGB")
            target = OUT / f"{stem}_frame{frame_index:02d}.png"
            frame.save(target)
            print(
                f"{path.name}: size={image.size}, frames={image.n_frames}, "
                f"duration_ms={image.info.get('duration')}, qa_frame={target}"
            )


if __name__ == "__main__":
    main()
