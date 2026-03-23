import numpy as np # pyright: ignore[reportMissingImports]
from typing import Tuple

def check_unitary(A: np.ndarray, tol: float = 1e-12) -> bool:
    """
    Verifies the local isometry condition (Left Canonical Form).
    Necessary condition for the bulk tensor to generate a valid MPU.
    
    Equation: sum_{i, a} A_{i,j,a,b} * A^*_{i,k,a,c} = delta_{jk} delta_{bc}
    
    Args:
        A: Array (d, d, D, D) with axes (phys_in, phys_out, bond_L, bond_R)
        tol: Numerical tolerance for equality with the identity matrix
        
    Returns:
        bool: True if the tensor is a local isometry, False otherwise
    """
    d = A.shape[0]
    D = A.shape[2]
    
    # Contraction over phys_in (axis 0) and bond_L (axis 2)
    # A_dag_A[j, b, k, c] = sum_{i, a} conj(A[i, j, a, b]) * A[i, k, a, c]
    A_dag_A = np.tensordot(A.conj(), A, axes=([0, 2], [0, 2]))
    
    # Reshape the indices to obtain a (d*D) x (d*D) matrix
    # Current axes: (phys_out_1, bond_R_1, phys_out_2, bond_R_2) -> (j, b, k, c)
    # Permute into (j, b, k, c) -> not necessary, it is already aligned.
    matrix_repr = A_dag_A.reshape(d * D, d * D)
    
    # Create the identity matrix of dimension d*D
    identity = np.eye(d * D)
    
    # Measure the distance from the identity using the Frobenius norm
    diff = np.linalg.norm(matrix_repr - identity)
    return diff < tol

def check_assumption_1(
    A: np.ndarray, 
    l: np.ndarray, 
    r: np.ndarray, 
    tol: float = 1e-12
) -> Tuple[bool, np.ndarray, np.ndarray]:
    """
    Verifies Assumption 1 for a uniform MPU.
    Checks that the map from the bond space to the physical space is injective
    at the boundaries of the chain.
    
    Args:
        A: Array (d, d, D, D) - Bulk tensor (phys_in, phys_out, bond_L, bond_R)
        l: Array (D,) - Left boundary vector
        r: Array (D,) - Right boundary vector
        tol: Numerical threshold for singular values
        
    Returns:
        Tuple[bool, np.ndarray, np.ndarray]:
            - Verification result (True if rank == D on both sides)
            - Singular values of the left contraction
            - Singular values of the right contraction
    """
    D = A.shape[2]
    
    # 1. Left Boundary: contraction of l over the bond_L index (axis 2)
    left_map = np.tensordot(l, A, axes=(0, 2))  # Result: (d, d, D)
    left_matrix = left_map.reshape(-1, D)
    s_left = np.linalg.svd(left_matrix, compute_uv=False)
    is_left_ok = np.sum(s_left > tol) == D
    
    # 2. Right Boundary: contraction of r over the bond_R index (axis 3)
    right_map = np.tensordot(A, r, axes=(3, 0)) # Result: (d, d, D)
    right_matrix = right_map.reshape(-1, D)
    s_right = np.linalg.svd(right_matrix, compute_uv=False)
    is_right_ok = np.sum(s_right > tol) == D
    
    return is_left_ok and is_right_ok, s_left, s_right