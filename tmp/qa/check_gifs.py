from pathlib import Path

from PIL import Image


ROOTS = (
    Path(r"E:\量子散射\Yifeng CHEN 999016959\Codes and Figures\output"),
    Path(r"E:\量子散射\Shufan ZHANG 999018435\codes and figures\outputs"),
)


def main() -> int:
    failures = []
    files = sorted(path for root in ROOTS for path in root.rglob("*.gif"))
    for path in files:
        try:
            with Image.open(path) as image:
                expected = getattr(image, "n_frames", 1)
                durations = []
                sizes = set()
                for frame in range(expected):
                    image.seek(frame)
                    image.load()
                    sizes.add(image.size)
                    durations.append(int(image.info.get("duration", 0)))
                if image.format != "GIF" or expected < 2 or len(sizes) != 1:
                    raise ValueError(
                        f"format={image.format}, frames={expected}, sizes={sorted(sizes)}"
                    )
                print(
                    f"PASS\t{expected:4d} frames\t{next(iter(sizes))[0]}x"
                    f"{next(iter(sizes))[1]}\t"
                    f"duration={sum(durations) / 1000:.3f}s\t{path}"
                )
        except Exception as exc:  # pragma: no cover - diagnostic utility
            failures.append((path, exc))
            print(f"FAIL\t{path}\t{exc}")
    print(f"SUMMARY\tchecked={len(files)}\tfailed={len(failures)}")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
