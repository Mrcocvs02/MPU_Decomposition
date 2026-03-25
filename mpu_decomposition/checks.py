import numpy as np  # pyright: ignore[reportMissingImports]
from typing import Tuple
from quimb import tensor as qtn  # type: ignore

from .utils import get_mpo_site_tensors


def check_mpo_unitarity(
    N_max, A_np, l_in, r_in, tol=1e-6, verbose=False, early_stop=True
):
    """
    Iteratively verifies the unitarity of an MPO chain by checking the ratio
    between Tr(rho)^2 and Tr(rho^2). For a unitary evolution, this ratio
    (after dimension scaling) must be 1.0.
    """

    # --- 1. Boundary Initialization ---
    # Defensive copies to prevent global state leakage
    l_tensor = qtn.Tensor(l_in.copy(), inds=["bond_0"], tags=["l"])
    r_tensor = qtn.Tensor(r_in.copy(), inds=["bond_0"], tags=["r"])

    # Adjoint boundaries for the double-layer contraction
    l_dag = l_tensor.H.reindex({"bond_0": "bond_0_dag"}).retag({"l": "l*"})
    r_dag = r_tensor.H.reindex({"bond_0": "bond_0_dag"}).retag({"r": "r*"})

    # The 'store' represents the accumulated Transfer Matrix environment
    store = l_tensor & l_dag
    failed_orders = []

    # --- 2. Iterative Site Evaluation ---
    for N in range(1, N_max + 1):
        # Update local right boundaries for current chain length N
        r_loc = r_tensor.reindex({"bond_0": f"bond_{N}"})
        r_dag_loc = r_dag.reindex({"bond_0_dag": f"bond_{N}_dag"})

        # Get MPO site and its conjugate
        A_k, A_k_dag = get_mpo_site_tensors(N, A_np)
        store @= A_k @ A_k_dag

        # UU_dag represents the full MPO chain closed by the boundary vectors
        UU_dag = store @ r_loc @ r_dag_loc

        # --- A. Compute Numerator: (Tr[U U^dagger])^2 ---
        # Close the physical indices to compute the trace
        trace_net = UU_dag.reindex({f"p_in_{N}": f"p_out_f_{N}"})
        val_tr = trace_net.contract()
        numerator = val_tr**2

        # --- B. Compute Denominator: d * Tr[(U U^dagger)^2] ---
        # Swap indices to connect two copies of the chain for the Tr(rho^2) term
        UU_dag_swapped = (
            UU_dag.reindex({f"p_in_{N}": f"dummy_{N}"})
            .reindex({f"p_out_f_{N}": f"p_in_{N}"})
            .reindex({f"dummy_{N}": f"p_out_f_{N}"})
        )
        # Scale by physical dimension d = A_np.shape[0]
        denominator = A_np.shape[0] * (UU_dag & UU_dag_swapped).contract()

        # --- C. Numerical Evaluation ---
        try:
            check_val = (numerator / denominator).real
        except (ZeroDivisionError, RuntimeWarning):
            check_val = np.nan

        error_k = abs(1.0 - float(check_val))

        if verbose:
            print(
                f"Site {N:03d} | Unitarity Check: {check_val:.10f} | Error: {error_k:.2e}"
            )

        # Validation Logic
        if error_k > tol or np.isnan(error_k):
            failed_orders.append((N, error_k))
            if early_stop and error_k > 1e-1:
                if verbose:
                    print(f"CRITICAL: Unitarity lost at N={N}. Stopping.")
                break

        # --- 3. Environment Update ---
        # Move to next site: contract physical indices and normalize to prevent overflow
        store = store.reindex({f"p_out_f_{N}": f"p_in_{N}"}).contract()
        store /= store.norm()

    return failed_orders == []


def check_assumption_1(
    A: np.ndarray, l_vec: np.ndarray, r_vec: np.ndarray, tol: float = 1e-12
) -> Tuple[bool, np.ndarray, np.ndarray]:
    """
    Verifies Assumption 1 for a uniform MPU.
    Checks that the map from the bond space to the physical space is injective
    at the boundaries of the chain.

    Args:
        A: Array (d, d, D, D) - Bulk tensor (phys_in, phys_out, bond_L, bond_R)
        l_vec: Array (D,) - Left boundary vector
        r_vec: Array (D,) - Right boundary vector
        tol: Numerical threshold for singular values

    Returns:
        Tuple[bool, np.ndarray, np.ndarray]:
            - Verification result (True if rank == D on both sides)
            - Singular values of the left contraction
            - Singular values of the right contraction
    """
    D = A.shape[2]

    # 1. Left Boundary: contraction of l_vec over the bond_L index (axis 2)
    left_map = np.tensordot(l_vec, A, axes=(0, 2))  # Result: (d, d, D)
    left_matrix = left_map.reshape(-1, D)
    s_left = np.linalg.svd(left_matrix, compute_uv=False)
    is_left_ok = np.sum(s_left > tol) == D

    # 2. Right Boundary: contraction of r_vec over the bond_R index (axis 3)
    right_map = np.tensordot(A, r_vec, axes=(3, 0))  # Result: (d, d, D)
    right_matrix = right_map.reshape(-1, D)
    s_right = np.linalg.svd(right_matrix, compute_uv=False)
    is_right_ok = np.sum(s_right > tol) == D

    return is_left_ok and is_right_ok, s_left, s_right
