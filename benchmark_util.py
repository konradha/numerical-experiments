from dataclasses import dataclass
import numpy as np

@dataclass
class MethodStatistics:
    method_name: str
    CFL_valid: bool

    num_trials: int
    samples_per_trial: int

    ti: float
    tf: float
    nt: int
    dt: float

    Lx_min: float
    Lx_max: float
    Ly_min: float
    Ly_max: float
    dx: float
    dy: float
    nx: int
    ny: int

    solution_u: np.ndarray # shape: (nt, nx, ny,)
    solution_v: np.ndarray # shape: (nt, nx, ny,)

    energies: np.ndarray      # shape: (nt,)

    RSME_errors: np.ndarray   # shape: (nt,) 
    L2_errors: np.ndarray     # shape: (nt,)
    Linfty_errors: np.ndarray # shape: (nt,)

    walltimes: np.ndarray # shape: (num_trials,)

    device: str
    nthreads: int

