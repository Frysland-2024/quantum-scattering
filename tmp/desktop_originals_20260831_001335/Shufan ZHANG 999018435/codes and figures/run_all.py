import argparse,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser(); p.add_argument('--quick',action='store_true'); a=p.parse_args()
for script in ['part1_free_gaussian.py','part2_regularized_delta.py','part3_stationary_scattering.py']:
    cmd=[sys.executable,str(ROOT/script)] + (['--quick'] if a.quick and script=='part1_free_gaussian.py' else [])
    subprocess.run(cmd,check=True,cwd=ROOT)
cmd=[sys.executable,str(ROOT/'part4_wavepacket_scattering.py'),'--scenario','all'] + (['--quick'] if a.quick else [])
subprocess.run(cmd,check=True,cwd=ROOT)
