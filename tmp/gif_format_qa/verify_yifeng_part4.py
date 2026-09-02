from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(r"E:\量子散射\Yifeng CHEN 999016959\Codes and Figures")
OUTPUT = ROOT / "output" / "part4_wavepacket_scattering"
SOURCE = ROOT / "src" / "quantum_scattering" / "figures.py"


def main() -> None:
    gifs = (
        "first_resonance_full.gif",
        "first_resonance_zoom.gif",
        "second_resonance_full.gif",
        "second_resonance_zoom.gif",
    )
    for name in gifs:
        with Image.open(OUTPUT / name) as image:
            assert image.size == (1200, 500), (name, image.size)
            assert image.n_frames == 40, (name, image.n_frames)
            for frame in range(image.n_frames):
                image.seek(frame)
                image.load()
        print(f"decoded {name}: 1200x500, 40 frames")

    first = json.loads((OUTPUT / "first_resonance_diagnostics.json").read_text(encoding="utf-8"))
    second = json.loads((OUTPUT / "second_resonance_diagnostics.json").read_text(encoding="utf-8"))
    assert abs(first["overall_transmission"] - 0.9983639116896925) < 1.0e-12
    assert first["basis_center_transmission"] > 0.99
    assert first["overall_unitarity_error"] < 1.0e-8
    assert second["overall_transmission"] > 0.999
    assert second["overall_unitarity_error"] < 1.0e-8
    assert first["global_rendering"]["mode"] == "complex carrier"
    assert second["global_rendering"]["mode"] == "complex carrier"

    manifest = json.loads((OUTPUT / "part4_manifest.json").read_text(encoding="utf-8"))
    assert manifest["profile"] == "reference"
    assert manifest["include_gifs"] is True
    for name in gifs:
        assert name in manifest["artifacts"]

    source_text = SOURCE.read_text(encoding="utf-8")
    assert "magnitude envelope" not in source_text
    assert "samples/wavelength" not in source_text
    print("numerical diagnostics, manifest, and source-format checks passed")


if __name__ == "__main__":
    main()
