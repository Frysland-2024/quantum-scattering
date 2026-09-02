from __future__ import annotations
from dataclasses import dataclass
import numpy as np

HBAR = 1.0
MASS = 1.0
A_RANGE = 12.0


def model_potential(x: np.ndarray | float) -> np.ndarray:
    """Lecture Eq. (116): V(x)=(0.5 x^2-0.8) exp(-0.1 x^2), m=hbar=1."""
    x = np.asarray(x, dtype=float)
    return (0.5 * x**2 - 0.8) * np.exp(-0.1 * x**2)


def finite_range_potential(x: np.ndarray | float, a: float = A_RANGE) -> np.ndarray:
    """Finite-range numerical realization for Appendix V; tails are negligible at |x|=12."""
    x = np.asarray(x, dtype=float)
    return np.where(np.abs(x) < a, model_potential(x), 0.0)


@dataclass
class ScanResult:
    energies: np.ndarray
    T: np.ndarray
    R: np.ndarray
    probability_error: np.ndarray


def appendix_v_scan(energies: np.ndarray, *, dx: float = 0.0025, a: float = A_RANGE,
                    margin: float = 0.05, mass: float = MASS, hbar: float = HBAR) -> ScanResult:
    """Appendix V / Neuhauser finite-difference calculation (Eqs. 312-320, 315)."""
    E = np.asarray(energies, dtype=float)
    if E.ndim != 1 or np.any(E <= 0):
        raise ValueError("energies must be a 1D array with E>0")
    k = np.sqrt(2.0 * mass * E) / hbar
    x_left, x_right = -a - margin, a + margin
    n_steps = int(round((x_right - x_left) / dx))
    x = x_left + dx * np.arange(n_steps + 1)
    V = finite_range_potential(x, a)

    # Eq. (313): modified state has unit transmitted amplitude on the right.
    psi_np1 = np.exp(1j * k * x[-1])
    psi_n = np.exp(1j * k * x[-2])
    c = 2.0 * mass * dx**2 / hbar**2
    for n in range(n_steps - 1, 0, -1):
        psi_nm1 = (2.0 + c * (V[n] - E)) * psi_n - psi_np1
        psi_np1, psi_n = psi_n, psi_nm1

    # Left free region: psi=A exp(ikx)+B exp(-ikx).
    psi0, psi1 = psi_n, psi_np1
    x0, x1 = x[0], x[1]
    e0p, e0m = np.exp(1j*k*x0), np.exp(-1j*k*x0)
    e1p, e1m = np.exp(1j*k*x1), np.exp(-1j*k*x1)
    det = e0p*e1m - e0m*e1p
    A = (psi0*e1m - psi1*e0m) / det
    B = (e0p*psi1 - e1p*psi0) / det
    T, R = 1.0/A, B/A
    err = np.abs(np.abs(T)**2 + np.abs(R)**2 - 1.0)
    return ScanResult(E, T, R, err)


@dataclass
class StateResult:
    x: np.ndarray
    psi_modified: np.ndarray
    psi_physical: np.ndarray
    T: complex
    R: complex
    A: complex
    B: complex


def appendix_v_state(energy: float, x_eval: np.ndarray, *, dx: float = 0.0025,
                     a: float = A_RANGE, margin: float = 0.05,
                     mass: float = MASS, hbar: float = HBAR) -> StateResult:
    E = float(energy)
    k = np.sqrt(2.0 * mass * E) / hbar
    x_left, x_right = -a - margin, a + margin
    n_steps = int(round((x_right - x_left) / dx))
    xg = x_left + dx*np.arange(n_steps+1)
    Vg = finite_range_potential(xg,a)
    bar = np.empty(xg.size,dtype=complex)
    bar[-1]=np.exp(1j*k*xg[-1]); bar[-2]=np.exp(1j*k*xg[-2])
    c=2.0*mass*dx**2/hbar**2
    for n in range(n_steps-1,0,-1):
        bar[n-1]=(2.0+c*(Vg[n]-E))*bar[n]-bar[n+1]
    x0,x1=xg[0],xg[1]; psi0,psi1=bar[0],bar[1]
    e0p,e0m=np.exp(1j*k*x0),np.exp(-1j*k*x0)
    e1p,e1m=np.exp(1j*k*x1),np.exp(-1j*k*x1)
    det=e0p*e1m-e0m*e1p
    A=(psi0*e1m-psi1*e0m)/det
    B=(e0p*psi1-e1p*psi0)/det
    T,R=1/A,B/A
    xe=np.asarray(x_eval,dtype=float)
    mod=np.empty(xe.size,dtype=complex)
    left=xe < -a; right=xe > a; inside=~(left|right)
    mod[left]=A*np.exp(1j*k*xe[left])+B*np.exp(-1j*k*xe[left])
    mod[right]=np.exp(1j*k*xe[right])
    mod[inside]=np.interp(xe[inside],xg,bar.real)+1j*np.interp(xe[inside],xg,bar.imag)
    return StateResult(xe,mod,mod/A,T,R,A,B)


def left_incident_state_matrix(p_values: np.ndarray, x_eval: np.ndarray, *, dx: float = 0.01,
                               a: float = A_RANGE, margin: float = 0.05,
                               mass: float = MASS, hbar: float = HBAR):
    """Momentum-normalized physical left-incident stationary states used in Part 4."""
    p=np.asarray(p_values,dtype=float); E=p**2/(2*mass); k=p/hbar
    x_left,x_right=-a-margin,a+margin
    n_steps=int(round((x_right-x_left)/dx)); xg=x_left+dx*np.arange(n_steps+1)
    Vg=finite_range_potential(xg,a)
    bar=np.empty((p.size,xg.size),dtype=complex)
    bar[:,-1]=np.exp(1j*k*xg[-1]); bar[:,-2]=np.exp(1j*k*xg[-2])
    c=2*mass*dx**2/hbar**2
    for n in range(n_steps-1,0,-1):
        bar[:,n-1]=(2+c*(Vg[n]-E))*bar[:,n]-bar[:,n+1]
    x0,x1=xg[0],xg[1]; psi0,psi1=bar[:,0],bar[:,1]
    e0p,e0m=np.exp(1j*k*x0),np.exp(-1j*k*x0)
    e1p,e1m=np.exp(1j*k*x1),np.exp(-1j*k*x1)
    det=e0p*e1m-e0m*e1p
    A=(psi0*e1m-psi1*e0m)/det; B=(e0p*psi1-e1p*psi0)/det
    T,R=1/A,B/A
    xe=np.asarray(x_eval,dtype=float); states=np.empty((p.size,xe.size),dtype=complex)
    left=xe < -a; right=xe > a; inside=~(left|right)
    if np.any(left):
        states[:,left]=np.exp(1j*k[:,None]*xe[left])+R[:,None]*np.exp(-1j*k[:,None]*xe[left])
    if np.any(right):
        states[:,right]=T[:,None]*np.exp(1j*k[:,None]*xe[right])
    if np.any(inside):
        for j in range(p.size):
            y=bar[j]/A[j]
            states[j,inside]=np.interp(xe[inside],xg,y.real)+1j*np.interp(xe[inside],xg,y.imag)
    states /= np.sqrt(2*np.pi*hbar)
    return states,T,R
