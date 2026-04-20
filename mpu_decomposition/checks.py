import numpy as np  # pyright: ignore[reportMissingImports]
from typing import Tuple
from quimb import tensor as qtn  # type: ignore
import logging
from .utils import get_mpo_site_tensors

logger = logging.getLogger(__name__)


def check_mpo_unitarity(
    N_max: int, A_np: np.ndarray, l_in: np.ndarray, r_in: np.ndarray, tol: float = 1e-6
) -> None:
    """
    Iteratively verifies the unitarity of an MPO chain by checking the ratio
    between Tr(rho)^2 and Tr(rho^2). For a unitary evolution, this ratio must be 1.0.
    Raises ValueError if unitarity is violated beyond the tolerance.
    """
    l_vec = qtn.Tensor(l_in.copy(), inds=["bond_0"], tags=["l"])
    r_vec = r_in.copy()

    l_vec_conj = l_vec.H.reindex({"bond_0": "bond_0_dag"}).retag({"l": "l*"})

    single_store = l_vec & l_vec_conj
    double_store = (
        l_vec
        & l_vec_conj
        & l_vec.reindex({"bond_0": f"bond_{N_max+1}"})
        & l_vec_conj.reindex({"bond_0_dag": f"bond_{N_max+1}_dag"})
    )

    r_vec_tensor = qtn.Tensor(r_vec, inds=["bond_0"], tags=["r"])
    r_vec_conj = r_vec_tensor.H.reindex({"bond_0": "bond_0_dag"}).retag({"r": "r*"})

    for N in range(1, N_max + 1):
        r_loc = r_vec_tensor.reindex({"bond_0": f"bond_{N}"})
        r_loc_dag = r_vec_conj.reindex({"bond_0_dag": f"bond_{N}_dag"})

        A_k, A_k_dag = get_mpo_site_tensors(N, A_np)
        single_store &= A_k & A_k_dag.reindex({f"p_out_f_{N}": f"p_in_{N}"})

        A_k_double, A_k_dag_double = get_mpo_site_tensors(N + N_max + 1, A_np)
        double_store &= (
            A_k
            & A_k_dag.reindex({f"p_out_f_{N}": f"p_middle_{N}"})
            & A_k_double.reindex({f"p_in_{N+N_max+1}": f"p_middle_{N}"})
            & A_k_dag_double.reindex({f"p_out_f_{N+N_max+1}": f"p_in_{N}"})
        )

        UU_dag = single_store.copy() & r_loc & r_loc_dag
        numerator = UU_dag.contract() ** 2

        UU_double = (
            double_store
            & r_loc
            & r_loc_dag
            & r_vec_tensor.reindex({"bond_0": f"bond_{N+N_max+1}"})
            & r_vec_conj.reindex({"bond_0_dag": f"bond_{N+N_max+1}_dag"})
        )

        denominator = (A_np.shape[0] ** N) * UU_double.contract()

        try:
            check_val = (numerator / denominator).real
        except (ZeroDivisionError, RuntimeWarning):
            check_val = np.nan

        error_k = abs(1.0 - float(check_val))
        logger.debug(
            f"Site {N:03d} | Unitarity Check: {check_val:.10f} | Error: {error_k:.2e}"
        )

        if error_k > tol or np.isnan(error_k):
            raise ValueError(
                f"Unitarity strictly lost at N={N}. Error {error_k:.2e} exceeds tolerance {tol}."
            )

    logger.debug(f"MPO Unitarity verified up to N_max={N_max}.")


def check_assumption_1(
    A: np.ndarray, l_vec: np.ndarray, r_vec: np.ndarray, tol: float = 1e-12
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Verifies Assumption 1 for a uniform MPU.
    Checks that the map from the bond space to the physical space is injective.
    Raises ValueError if the rank is strictly less than the bond dimension D.
    Returns singular values for diagnostics.
    """
    D = A.shape[2]

    left_map = np.tensordot(l_vec, A, axes=(0, 2))
    left_matrix = left_map.reshape(-1, D)
    s_left = np.linalg.svd(left_matrix, compute_uv=False)

    right_map = np.tensordot(A, r_vec, axes=(3, 0))
    right_matrix = right_map.reshape(-1, D)
    s_right = np.linalg.svd(right_matrix, compute_uv=False)

    rank_left = np.sum(s_left > tol)
    rank_right = np.sum(s_right > tol)

    if rank_left < D:
        raise ValueError(
            f"Assumption 1 failed: Left boundary map is not injective. Rank {rank_left} < D ({D})."
        )
    if rank_right < D:
        raise ValueError(
            f"Assumption 1 failed: Right boundary map is not injective. Rank {rank_right} < D ({D})."
        )

    logger.debug("Assumption 1 (Injectivity) verified on both boundaries.")
    return s_left, s_right


def verify_lcu(
    M_original: np.ndarray, coefficients: list, unitaries: list, tol: float = 1e-10
) -> None:
    """
    Verifies the validity of an LCU decomposition.
    Raises ValueError if any mathematical constraint is violated.
    """
    M_original = np.asarray(M_original, dtype=complex)
    M_reconstructed = np.zeros_like(M_original, dtype=complex)
    I_exact = np.eye(M_original.shape[0], dtype=complex)

    for idx, (c, W) in enumerate(zip(coefficients, unitaries)):
        if not (np.isreal(c) and c > 0):
            raise ValueError(f"Coefficient at index {idx} is not real positive: {c}")

        W = np.asarray(W, dtype=complex)
        if not np.allclose(W @ W.conj().T, I_exact, atol=tol):
            raise ValueError(f"Unitary at index {idx} violates W @ W† = I.")

        M_reconstructed += c * W

    if not np.allclose(M_reconstructed, M_original, atol=tol):
        error_norm = np.linalg.norm(M_reconstructed - M_original, ord="fro")
        raise ValueError(
            f"LCU sum does not reconstruct M. Frobenius error: {error_norm:.4e}"
        )

    logger.debug(
        "LCU verification passed: real/positive coefficients, valid unitaries, exact reconstruction."
    )


def verify_merging_unitary(
    B: np.ndarray,
    W_ctrl: np.ndarray,
    M_operator: np.ndarray,
    C: float,
    dim_system: int,
    dim_ancilla: int,
    tol: float = 1e-10,
) -> None:
    """
    Verifies that the block encoding correctly embeds M / C in the |0>_A subspace.
    Assumes tensor product structure: System ⊗ Ancilla.
    """
    B_full = np.kron(np.eye(dim_system, dtype=complex), B)
    B_full_dag = B_full.conj().T

    # L'unitario totale del protocollo LCU richiede B^dagger alla fine
    U_total = B_full_dag @ W_ctrl @ B_full

    proj_0_ancilla = np.zeros((dim_ancilla, dim_ancilla), dtype=complex)
    proj_0_ancilla[0, 0] = 1.0
    P_0 = np.kron(np.eye(dim_system, dtype=complex), proj_0_ancilla)

    # Post-selezione
    U_post_selected = P_0 @ U_total @ P_0
    M_target_embedded = np.kron(M_operator / C, proj_0_ancilla)

    if not np.allclose(U_post_selected, M_target_embedded, atol=tol):
        error = np.linalg.norm(U_post_selected - M_target_embedded, ord="fro")
        raise ValueError(
            f"Merging unitary post-selection failed. Operator does not yield M/C. Frobenius error: {error:.4e}"
        )

    logger.debug(
        "Merging unitary verification passed: post-selection perfectly extracts M / C."
    )
