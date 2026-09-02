"""Cross-check the two Part 5 implementations against an independent oracle."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

QA_DIR = Path(__file__).resolve().parent
YIFENG_PATH = Path(
    r"E:\量子散射\Yifeng CHEN 999016959\Codes and Figures\src\quantum_scattering\part5.py"
)
SHUFAN_PATH = Path(
    r"E:\量子散射\Shufan ZHANG 999018435\codes and figures\quantum_scattering\box_basis.py"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


oracle = load_module("qa_part5_reference", QA_DIR / "part5_reference.py")
yifeng = load_module("qa_yifeng_part5", YIFENG_PATH)
shufan = load_module("qa_shufan_box_basis", SHUFAN_PATH)


def yifeng_full_hamiltonian(potential, length: float, size: int, points: int) -> np.ndarray:
    moments = yifeng._cosine_moments(
        potential,
        length=length,
        grid_points=points,
        maximum_index=2 * size,
    )
    indices = np.arange(1, size + 1, dtype=np.int64)
    differences = np.abs(indices[:, None] - indices[None, :])
    sums = indices[:, None] + indices[None, :]
    hamiltonian = (moments[differences] - moments[sums]) / length
    hamiltonian[np.diag_indices(size)] += oracle.kinetic_diagonal(length, size)
    return hamiltonian


def direct_hamiltonian(potential, length: float, size: int, order: int = 1024) -> np.ndarray:
    result = oracle.quadrature_potential_matrix(potential, length, size, order=order)
    result[np.diag_indices(size)] += oracle.kinetic_diagonal(length, size)
    return result


def orthogonality_error(coefficients: np.ndarray) -> float:
    count = coefficients.shape[1]
    return float(np.max(np.abs(coefficients.T @ coefficients - np.eye(count))))


def localized_summary(energies: np.ndarray, coefficients: np.ndarray, length: float) -> list[dict]:
    bound = {
        "index_zero_based": 0,
        "energy": float(energies[0]),
        "central_probability": oracle.reconstruct_probability(
            coefficients[:, 0], length, points=2001
        )[0],
        "parity_expectation": oracle.reconstruct_probability(
            coefficients[:, 0], length, points=2001
        )[2],
    }
    first = oracle.most_localized_in_window(
        energies, coefficients, length, 0.45, 0.85, points=2001
    )
    second = oracle.most_localized_in_window(
        energies, coefficients, length, 1.05, 1.60, points=2001
    )
    return [bound, first, second]


def core_comparison() -> dict:
    length_small = 30.0
    size_small = 16
    points_small = 769
    matrix_rows = {}
    for name, potential in (
        ("harmonic", oracle_harmonic),
        ("double_barrier", oracle.double_barrier_potential),
    ):
        reference = direct_hamiltonian(potential, length_small, size_small)
        hy = yifeng_full_hamiltonian(potential, length_small, size_small, points_small)
        hs = shufan.build_hamiltonian(
            potential, size_small, length_small, points_small
        )
        matrix_rows[name] = {
            "yifeng_max_abs_from_direct_gauss": float(np.max(np.abs(hy - reference))),
            "shufan_max_abs_from_direct_gauss": float(np.max(np.abs(hs - reference))),
            "between_implementations_max_abs": float(np.max(np.abs(hy - hs))),
            "yifeng_symmetry_error": float(np.max(np.abs(hy - hy.T))),
            "shufan_symmetry_error": float(np.max(np.abs(hs - hs.T))),
            "off_parity_yifeng_max_abs": float(
                np.max(np.abs(hy[(np.indices(hy.shape).sum(axis=0) % 2) == 1]))
            ),
            "off_parity_shufan_max_abs": float(
                np.max(np.abs(hs[(np.indices(hs.shape).sum(axis=0) % 2) == 1]))
            ),
        }

    ho_config = {"length": 80.0, "size": 240, "points": 1201, "states": 10}
    y_ho = yifeng.solve_box_basis(
        yifeng.harmonic_oscillator_potential,
        length=ho_config["length"],
        basis_size=ho_config["size"],
        grid_points=ho_config["points"],
        state_count=ho_config["states"],
    )
    s_ho = shufan.solve_box_basis(
        oracle_harmonic,
        ho_config["size"],
        ho_config["length"],
        ho_config["points"],
        n_states=ho_config["states"],
    )
    exact = np.arange(10, dtype=float) - 0.3

    db_config = {"length": 200.0, "size": 400, "points": 2401, "states": 125}
    y_db = yifeng.solve_box_basis(
        yifeng.short_range_potential,
        length=db_config["length"],
        basis_size=db_config["size"],
        grid_points=db_config["points"],
        state_count=db_config["states"],
    )
    s_db = shufan.solve_box_basis(
        oracle.double_barrier_potential,
        db_config["size"],
        db_config["length"],
        db_config["points"],
        n_states=db_config["states"],
    )
    h_oracle = oracle.double_barrier_hamiltonian(
        db_config["length"], db_config["size"]
    )
    e_oracle, c_oracle = np.linalg.eigh(h_oracle)
    e_oracle = e_oracle[: db_config["states"]]
    c_oracle = c_oracle[:, : db_config["states"]]

    y_local = localized_summary(y_db.energies, y_db.coefficients, db_config["length"])
    s_local = localized_summary(s_db.energies, s_db.coefficients, db_config["length"])
    o_local = localized_summary(e_oracle, c_oracle, db_config["length"])
    selected = [0, y_local[1]["index_zero_based"], y_local[2]["index_zero_based"]]
    overlaps_y_s = [
        float(abs(np.vdot(y_db.coefficients[:, index], s_db.coefficients[:, index])))
        for index in selected
    ]
    overlaps_y_o = [
        float(abs(np.vdot(y_db.coefficients[:, index], c_oracle[:, index])))
        for index in selected
    ]

    return {
        "matrix_checks": matrix_rows,
        "common_harmonic": {
            "config": ho_config,
            "yifeng_max_abs_error": float(np.max(np.abs(y_ho.energies - exact))),
            "shufan_max_abs_error": float(np.max(np.abs(s_ho.energies - exact))),
            "between_implementations_max_abs_energy": float(
                np.max(np.abs(y_ho.energies - s_ho.energies))
            ),
            "yifeng_orthogonality_error": orthogonality_error(y_ho.coefficients),
            "shufan_orthogonality_error": orthogonality_error(s_ho.coefficients),
        },
        "common_double_barrier": {
            "config": db_config,
            "negative_counts": {
                "yifeng": int(np.sum(y_db.energies < 0.0)),
                "shufan": int(np.sum(s_db.energies < 0.0)),
                "oracle": int(np.sum(e_oracle < 0.0)),
            },
            "max_abs_energy_yifeng_vs_shufan": float(
                np.max(np.abs(y_db.energies - s_db.energies))
            ),
            "max_abs_energy_yifeng_vs_oracle": float(
                np.max(np.abs(y_db.energies - e_oracle))
            ),
            "max_abs_energy_shufan_vs_oracle": float(
                np.max(np.abs(s_db.energies - e_oracle))
            ),
            "yifeng_localized": y_local,
            "shufan_localized": s_local,
            "oracle_localized": o_local,
            "selected_state_abs_overlap_yifeng_shufan": overlaps_y_s,
            "selected_state_abs_overlap_yifeng_oracle": overlaps_y_o,
            "yifeng_orthogonality_error": orthogonality_error(y_db.coefficients),
            "shufan_orthogonality_error": orthogonality_error(s_db.coefficients),
        },
    }


def oracle_harmonic(x: np.ndarray) -> np.ndarray:
    return 0.5 * np.asarray(x, dtype=float) ** 2 - 0.8


def measured_call(function, *args, **kwargs):
    gc.collect()
    tracemalloc.start()
    start = time.perf_counter()
    value = function(*args, **kwargs)
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return value, {"seconds": elapsed, "tracemalloc_peak_mib": peak / 2**20}


def formal_scale_benchmark(run_eigensolvers: bool) -> dict:
    length, size, points = 200.0, 2000, 10000
    payload = {"config": {"length": length, "size": size, "points": points}}

    def yifeng_blocks():
        moments = yifeng._cosine_moments(
            yifeng.short_range_potential,
            length=length,
            grid_points=points,
            maximum_index=2 * size,
        )
        indices = np.arange(1, size + 1, dtype=np.int64)
        maxima = []
        for parity in (1, 0):
            block_indices = indices[indices % 2 == parity]
            block = yifeng._hamiltonian_block(block_indices, moments, length=length)
            maxima.append(float(np.max(np.abs(block))))
            del block
        return maxima

    _, payload["yifeng_dct_plus_two_parity_block_build"] = measured_call(yifeng_blocks)
    h_shufan, payload["shufan_full_hamiltonian_build"] = measured_call(
        shufan.build_hamiltonian,
        oracle.double_barrier_potential,
        size,
        length,
        points,
    )
    payload["shufan_full_hamiltonian_mib"] = h_shufan.nbytes / 2**20
    del h_shufan

    if run_eigensolvers:
        y_result, payload["yifeng_low_130_eigensolve_total"] = measured_call(
            yifeng.solve_box_basis,
            yifeng.short_range_potential,
            length=length,
            basis_size=size,
            grid_points=points,
            state_count=130,
        )
        payload["yifeng_key_energies"] = [
            float(y_result.energies[index]) for index in (0, 67, 100)
        ]
        del y_result
        s_result, payload["shufan_low_125_eigensolve_total"] = measured_call(
            shufan.solve_box_basis,
            oracle.double_barrier_potential,
            size,
            length,
            points,
            n_states=125,
        )
        payload["shufan_key_energies"] = [
            float(s_result.energies[index]) for index in (0, 67, 100)
        ]
        del s_result
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal", action="store_true", help="also time J=2000,N=10000 eigensolvers")
    parser.add_argument("--build-risk", action="store_true", help="time only J=2000,N=10000 matrix construction")
    args = parser.parse_args()
    payload = {"core": core_comparison()}
    if args.formal or args.build_risk:
        payload["formal_scale"] = formal_scale_benchmark(args.formal)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
