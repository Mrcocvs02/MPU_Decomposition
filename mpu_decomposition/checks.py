import numpy as np # pyright: ignore[reportMissingImports]
from typing import Tuple
from quimb import tensor as qtn

from .utils import get_mpo_site_tensors

def check_mpo_unitarity(N_max, A_np, l_in, r_in, tol=1e-6, verbose=False, early_stop=True):
    """
    Iteratively verifies the unitarity of an MPO chain by checking Tr(rho)/Tr(rho^2).
    In a perfect unitary evolution, this ratio (scaled) should be 1.0.
    """
    # Defensive copies to prevent global state leakage
    l = qtn.Tensor(l_in.copy(), inds=["bond_0"], tags=["l"])
    r = qtn.Tensor(r_in.copy(), inds=["bond_0"], tags=["r"])

    # Adjoint boundaries
    l_dag = l.H.reindex({"bond_0": "bond_0_dag"}).retag({'l': 'l*'})
    r_dag = r.H.reindex({"bond_0": "bond_0_dag"}).retag({'r': 'r*'})
    
    store = l & l_dag 
    failed_orders = []
    for N in range(1, N_max + 1):
        # Local right boundary for current length N
        r_loc = r.reindex({"bond_0": f"bond_{N}"})
        r_dag_loc = r_dag.reindex({"bond_0_dag": f"bond_{N}_dag"})
        
        A_k, A_k_dag = get_mpo_site_tensors(N, A_np)
        store @= A_k @ A_k_dag
        
        # UU_dag represents the full chain closed by boundaries
        UU_dag = store @ r_loc @ r_dag_loc
        
        # 1. Numerator: (Tr[U U^dagger])^2
        # We close the physical indices p_in and p_out_f of the current site
        trace_net = UU_dag.reindex({f'p_in_{N}': f'p_out_f_{N}'})
        val_tr = trace_net.contract( )
        numerator = val_tr ** 2
        
        # 2. Denominator: 2 * Tr[(U U^dagger)^2]
        # Safe swap via dummy index to connect two copies of the chain
        UU_dag_2 = (
            UU_dag.reindex({f'p_in_{N}': f'dummy_{N}'})
                  .reindex({f'p_out_f_{N}': f'p_in_{N}'})
                  .reindex({f'dummy_{N}': f'p_out_f_{N}'})
        )
        denominator = 2 * (UU_dag & UU_dag_2).contract( )
        
        # Stabilization and Evaluation
        try:
            check_val = (numerator / denominator).real
        except (ZeroDivisionError, RuntimeWarning):
            check_val = np.nan
            
        error_k = abs(1.0 - float(check_val))
        
        if verbose:
            print(f"Site {N:03d} | Unitarity Check: {check_val:.10f} | Error: {error_k:.2e}")

            
        if error_k > tol or np.isnan(error_k):
            failed_orders.append((N, error_k))
            if early_stop and error_k > 1e-1: # Stop if unitarity is completely lost
                if verbose: print(f"CRITICAL: Unitarity lost at N={N}. Stopping.")
                break

        # Move to next site: contract physical indices and normalize Transfer Matrix
        store = store.reindex({f"p_out_f_{N}": f"p_in_{N}"}).contract()
        store /= store.norm()


    return failed_orders==[]

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