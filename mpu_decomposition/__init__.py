"""
MPU_Decomposition: A theoretical and numerical framework for
Matrix-Product Unitaries and their quantum circuit synthesis.
"""

# 1. Importa i sottomoduli per renderli accessibili come attributi (es. mpu_decomposition.utils)
from . import utils
from . import checks
from . import MPU  # Assicurati che il file sia MPU.py (maiuscolo)

# 2. Esponi le classi e funzioni principali direttamente al livello root
from .MPU import UniformMPU
from .checks import check_mpo_unitarity, check_assumption_1
from .utils import get_mpo_site_tensors, optimize_q_unif, matrix_sqrt_hermitian

# 3. Aggiorna __all__ per includere la classe principale della tua tesi
__all__ = [
    "UniformMPU",
    "check_mpo_unitarity",
    "get_mpo_site_tensors",
    "optimize_q_unif",
    "check_assumption_1",
    "utils",
    "checks",
    "matrix_sqrt_hermitian",
    "verify_factored_decomposition",
    "MPU",
]
