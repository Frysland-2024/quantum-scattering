import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description="Run Parts 1-6")
parser.add_argument("--quick", action="store_true")
parser.add_argument("--skip-gifs", action="store_true", help="skip the Part 4 animations")
args = parser.parse_args()

for script in ("part1_free_gaussian.py", "part2_regularized_delta.py", "part3_stationary_scattering.py"):
    command = [sys.executable, str(ROOT / script)]
    if args.quick and script == "part1_free_gaussian.py":
        command.append("--quick")
    subprocess.run(command, check=True, cwd=ROOT)

part4 = [sys.executable, str(ROOT / "part4_wavepacket_scattering.py"), "--scenario", "all"]
if args.quick:
    part4.append("--quick")
if args.skip_gifs:
    part4.append("--skip-gifs")
subprocess.run(part4, check=True, cwd=ROOT)

part5 = [sys.executable, str(ROOT / "part5_box_basis.py")]
if args.quick:
    part5.append("--quick")
subprocess.run(part5, check=True, cwd=ROOT)

part6 = [sys.executable, str(ROOT / "part6_complex_scaling.py")]
if args.quick:
    part6.append("--quick")
subprocess.run(part6, check=True, cwd=ROOT)

part6_wavefunctions = [sys.executable, str(ROOT / "part6_resonance_wavefunctions.py")]
if args.quick:
    part6_wavefunctions.append("--quick")
subprocess.run(part6_wavefunctions, check=True, cwd=ROOT)
