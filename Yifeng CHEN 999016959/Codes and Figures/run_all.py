"""Generate every project part through the package command-line interface."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantum_scattering.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["all", *sys.argv[1:]]))
