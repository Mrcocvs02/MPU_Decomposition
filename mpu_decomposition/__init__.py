"""
MPU_Decomposition: A theoretical and numerical framework for
Matrix-Product Unitaries and their quantum circuit synthesis.
"""

from . import utils
from . import checks
from . import MPU

from .MPU import UniformMPU, CircuitDecomposition
from .checks import (
    check_mpo_unitarity,
    check_assumption_1,
    verify_lcu,
    verify_merging_unitary,
)
from .utils import (
    get_mpo_site_tensors,
    optimize_q_unif,
    matrix_sqrt_hermitian,
    get_merging_operator,
)

__all__ = [
    "UniformMPU",
    "CircuitDecomposition",
    "check_mpo_unitarity",
    "get_mpo_site_tensors",
    "optimize_q_unif",
    "check_assumption_1",
    "utils",
    "checks",
    "matrix_sqrt_hermitian",
    "verify_lcu",
    "get_merging_operator",
    "verify_merging_unitary",
    "MPU",
]
