# Numerical Quantum Scattering in One Dimension

A compact course project on one-dimensional quantum scattering, transmission resonances, and non-Hermitian resonance analysis. The project numerically solves the Schrödinger equation for a short-ranged double-barrier potential and connects three complementary pictures:

- stationary scattering and the transmission probability $|T(E)|^2$;
- time-dependent Gaussian wave-packet scattering;
- resonance states exposed by complex scaling / Siegert-state ideas.

The central goal is to recover the same resonance structure from conventional scattering calculations and from a non-Hermitian resonance description, then interpret the sharp transmission peaks in terms of complex resonance energies and Breit-Wigner line shapes.

## Representative results

### Transmission resonances

![Transmission profile](figures/transmission_profile.png)

The numerical scan of the model potential

$$
V(x) = (0.5x^2-0.8)e^{-0.1x^2}
$$

shows two prominent low-energy transmission resonances near $E\approx0.621$ and $E\approx1.33$.

### Wave-packet scattering

<p align="center">
  <img src="figures/first_resonance_wavepacket.gif" width="48%" />
  <img src="figures/second_resonance_wavepacket.gif" width="48%" />
</p>

Narrow-band Gaussian packets centered on the resonance energies are propagated through the interaction region to visualize resonant tunnelling in the time domain.

### Finite-box spectrum

![Finite-box states](figures/box_basis_states.png)

A particle-in-a-box basis converts the Hamiltonian into a finite matrix eigenvalue problem. Negative-energy bound states remain localized, while positive-energy box states discretize the continuum and include states close to the scattering resonances.

### Complex scaling and resonance poles

<p align="center">
  <img src="figures/complex_eigenvalues.png" width="48%" />
  <img src="figures/breit_wigner_zoom.png" width="48%" />
</p>

Under complex scaling, the discretized continuum rotates into the lower half of the complex-energy plane while isolated resonance eigenvalues remain approximately stationary. Their real parts locate resonance energies and their imaginary parts determine resonance widths. The first resonance is very narrow and is well described by an isolated Breit-Wigner profile; the second is broader and shows a stronger background contribution.

![Complex-scaled resonance wavefunction](figures/resonance_wavefunction.png)

The Siegert resonance wavefunction grows asymptotically on the real axis because of the outgoing boundary condition. After complex scaling it becomes numerically localized and can be treated with a square-integrable basis.

## Numerical workflow

1. **Free Gaussian packet** — Fourier representation and free propagation.
2. **Regularized Dirac delta** — finite-$L$ approximation to the delta distribution.
3. **Stationary scattering** — finite-difference solution of the 1D scattering boundary-value problem and numerical evaluation of $T(E)$.
4. **Wave-packet scattering** — superposition of stationary scattering states over a Gaussian momentum distribution.
5. **Finite-box diagonalization** — matrix representation of the Hamiltonian in a particle-in-a-box basis.
6. **Complex scaling** — complex eigenvalue spectra, resonance widths, Breit-Wigner comparison, and resonance wavefunctions.

## Repository structure

```text
.
├── README.md
├── code/
│   ├── part1_free_gaussian.py
│   ├── part2_regularized_delta.py
│   ├── part3_stationary_scattering.py
│   ├── part4_wavepacket_scattering.py
│   ├── part5_box_basis.py
│   ├── part6_complex_scaling.py
│   ├── part6_resonance_wavefunctions.py
│   ├── run_all.py
│   ├── requirements.txt
│   └── quantum_scattering/
├── figures/                 # selected portfolio figures
└── report/
    └── quantum_scattering_report.docx
```

Generated numerical outputs are written to `code/outputs/` and are intentionally excluded from version control. The `figures/` directory contains only a small set of representative results used in this README.

## Reproducing the calculations

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r code/requirements.txt
python code/run_all.py --quick --skip-gifs
```

For the full calculations, remove `--quick`. To regenerate the wave-packet animations, also remove `--skip-gifs`.

## Main dependencies

- NumPy
- SciPy
- Matplotlib
- Pillow

## Course context

This repository is the cleaned portfolio version of my individual **Quantum Scattering Theory** course project. The original assignment focuses on comparing the conventional Hermitian scattering treatment with a non-Hermitian Siegert-state description of transmission resonances.

Primary theoretical reference:

- M. Sindelka, *An Introduction to Scattering Theory*, arXiv:2204.03651.

## Author

**Shufan Zhang**

This portfolio branch intentionally excludes lecture handouts, instructor whiteboard scans, temporary numerical diagnostics, and other students' files. It keeps only the code, selected results, and the submitted project report needed to understand and reproduce the work.
