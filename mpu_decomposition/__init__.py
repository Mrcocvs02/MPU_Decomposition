"""
MPU_Decomposition: A theoretical and numerical framework for 
Matrix-Product Unitaries and their quantum circuit synthesis.
"""

from .checks import check_unitary, check_assumption_1

# The __all__ list strictly defines the public API.
# Only these functions will be imported when a user calls:
# from mpu_decomposition import *
__all__ = [
    "check_unitary", 
    "check_assumption_1"
]