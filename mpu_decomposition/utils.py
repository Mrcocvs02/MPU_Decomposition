# mpu_decomposition/utils.py
import numpy as np  # type: ignore
import quimb.tensor as qtn  # type: ignore
import scipy.linalg as la  # type: ignore
import jax
import jax.numpy as jnp
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)


def optimize_q_unif(T_blocked, l_vec, r_vec, eps_reg=1e-8):
    dim = T_blocked.shape[0]
    n_params = 2 * dim**2

    def vec_to_rho(p):
        X = p[: dim**2].reshape((dim, dim)) + 1j * p[dim**2 :].reshape((dim, dim))
        rho = X @ X.conj().T
        return rho / jnp.trace(rho).real

    @jax.jit
    def loss_and_grad(params, T, L_in, R_in):
        def loss_fn(p):
            sigma = vec_to_rho(p[:n_params])
            tau = vec_to_rho(p[n_params:])

            L2 = jnp.einsum("ij, oipq, pm, ojmn -> qn", sigma, T, L_in, T.conj())
            R2 = jnp.einsum("ij, oipq, qn, ojmn -> pm", tau, T, R_in, T.conj())

            eigvals_L, eigvecs_L = jnp.linalg.eigh(L2)
            eigvals_R, eigvecs_R = jnp.linalg.eigh(R2)

            inv_L = (
                eigvecs_L
                @ (jnp.diag(1.0 / (jnp.maximum(eigvals_L, 0.0) + eps_reg)))
                @ eigvecs_L.conj().T
            )
            inv_R = (
                eigvecs_R
                @ (jnp.diag(1.0 / (jnp.maximum(eigvals_R, 0.0) + eps_reg)))
                @ eigvecs_R.conj().T
            )

            return jnp.sqrt(jnp.abs(jnp.real(jnp.trace(inv_R @ inv_L.T))))

        return jax.value_and_grad(loss_fn)(params)

    # Pre-calcolo contorni e conversioni JAX
    T_j = jnp.asarray(T_blocked)
    L_in_j = jnp.outer(l_vec, l_vec.conj())
    R_in_j = jnp.outer(r_vec, r_vec.conj())

    def scipy_wrapper(p):
        val, grad = loss_and_grad(p, T_j, L_in_j, R_in_j)
        return np.asarray(val, dtype=np.float64), np.asarray(grad, dtype=np.float64)

    # Inizializzazione parametri
    x0 = np.random.randn(2 * n_params) * 0.001
    eye_flat = np.eye(dim).flatten() / np.sqrt(dim)
    x0[: dim**2] += eye_flat
    x0[n_params : n_params + dim**2] += eye_flat

    res = minimize(
        scipy_wrapper,
        x0,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 10000, "ftol": 1e-12, "gtol": 1e-8},
    )

    return vec_to_rho(res.x[:n_params]), vec_to_rho(res.x[n_params:]), res.fun


def get_mpo_site_tensors(k, A_np):
    """
    Generates the ket and bra (dag) tensors for a specific site k.
    Naming convention follows the MPU synthesis requirements.
    """
    # A[phys_in, phys_out, bond_L, bond_R]
    A_k = qtn.Tensor(
        A_np,
        inds=[f"p_in_{k}", f"p_out_{k}", f"bond_{k-1}", f"bond_{k}"],
        tags=[f"A_{k}"],
    )

    # Complex conjugation is mandatory for the adjoint tensor.
    A_k_dag = qtn.Tensor(
        np.conj(A_np),
        inds=[f"p_out_f_{k}", f"p_out_{k}", f"bond_{k-1}_dag", f"bond_{k}_dag"],
        tags=[f"A_{k}_dag"],
    )

    return A_k, A_k_dag


def matrix_sqrt_hermitian(M: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Compute M^{1/2} for a Hermitian positive-semidefinite matrix."""
    eigvals, eigvecs = np.linalg.eigh(M)
    if np.any(eigvals < -tol):
        raise ValueError(
            f"Matrix is not positive-semidefinite: min eigenvalue = {eigvals.min():.2e}"
        )
    eigvals = np.maximum(
        eigvals, 0.0
    )  # to erase small negative eigenvalues due to numerical errors
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.conj().T


def get_merging_operator(L_inv: np.ndarray, R_inv: np.ndarray) -> np.ndarray:
    """
    Computes the merging operator M from the boundary operators L2 and R2.
    This operator is crucial for the synthesis of the MPU into a quantum circuit.
    The merging operator encapsulates the entanglement structure at the boundaries
    and is derived from the inverses of L2 and R2.
    """
    # Validate input dimensions
    if L_inv.shape[0] != L_inv.shape[1] or R_inv.shape[0] != R_inv.shape[1]:
        raise ValueError("L2 and R2 must be square matrices.")

    if L_inv.shape != R_inv.shape:
        raise ValueError("L2 and R2 must be of the same dimension.")

    # Compute the merging operator M = |0,0⟩⟨I|(R⁻¹ ⊗ L⁻¹)
    try:
        D = L_inv.shape[0]

        M = np.zeros((D, D, D, D), dtype=complex)

        kernel = R_inv.T @ L_inv  # (D, D)
        M[0, 0, :, :] = kernel

        return M.reshape(D**2, D**2)
    except la.LinAlgError as e:
        raise ValueError(f"Failed to compute merging operator: {e}")


def dilate_isometry_to_unitary(V_mat: np.ndarray) -> np.ndarray:
    """
    Dilates an isometry V (shape: m x n, m >= n) to a unitary (shape: m x m)
    such that the first n columns equal V exactly.

    The complementary columns are an orthonormal basis for the null space of V†,
    computed via SVD (scipy.linalg.null_space).

    Parameters
    ----------
    V_mat : np.ndarray, shape (m, n), m >= n
        An isometry satisfying V†V = I_n.

    Returns
    -------
    V_tilde : np.ndarray, shape (m, m)
        Unitary with V_tilde[:, :n] == V_mat (up to floating point).

    Raises
    ------
    ValueError
        If m < n or V†V ≠ I_n.
    """
    m, n = V_mat.shape

    if m < n:
        raise ValueError(
            f"V_mat must have at least as many rows as columns. Got ({m}, {n})."
        )

    if not np.allclose(V_mat.conj().T @ V_mat, np.eye(n), atol=1e-10):
        raise ValueError("V_mat is not a valid isometry: V†V ≠ I.")

    if m == n:
        return V_mat.copy()

    # Orthogonal complement = null space of V†, computed via SVD
    tail = la.null_space(V_mat.conj().T)  # shape (m, m-n)
    V_tilde = np.hstack((V_mat, tail))  # shape (m, m)

    return V_tilde


def compute_lcu_pad_indices(
    D_bond: int, D_bond_pad: int, dim_rot: int = 2
) -> np.ndarray:
    """
    Computes the index map from the raw (unpadded) bond+rot space to the
    padded bond+rot space used by Qiskit's little-endian qubit ordering.

    The raw space has axes (ext_R, ext_L, rot) with sizes
    (D_bond, D_bond, dim_rot) in C-order (ext_R slowest, rot fastest).

    The padded space has the same axis order but D_bond is replaced by
    D_bond_pad, so the flat index of (i_R, i_L, rot) in the padded space is:
        idx_pad = i_L + i_R * D_bond_pad + rot * D_bond_pad**2

    This reflects Qiskit's convention:
        - ext_L  -> LSB  (stride 1)
        - ext_R  -> mid  (stride D_bond_pad)
        - rot    -> MSB  (stride D_bond_pad**2)

    Parameters
    ----------
    D_bond : int
        Physical (unpadded) bond dimension.
    D_bond_pad : int
        Padded bond dimension (must be >= D_bond, typically next power of 2).
    dim_rot : int, optional
        Dimension of the rotation register. Default is 2 (for a single qubit rotation).
        Rarely 1 as would imply an integer ell parameter by chance.

    Returns
    -------
    pad_indices : np.ndarray, shape (D_bond * D_bond * dim_rot,), dtype=np.intp
        pad_indices[k] is the flat index in the padded space corresponding
        to the k-th basis vector in the raw space (C-order enumeration).

    Raises
    ------
    ValueError
        If D_bond_pad < D_bond.
    """
    if D_bond_pad < D_bond:
        raise ValueError(f"D_bond_pad ({D_bond_pad}) must be >= D_bond ({D_bond}).")

    # Broadcasting: avoids the wasteful memory allocation of meshgrid
    i_R = np.arange(D_bond)[:, None, None]
    i_L = np.arange(D_bond)[None, :, None]
    rot = np.arange(dim_rot)[None, None, :]

    # The operation is resolved in a single vectorised pass
    pad_indices = (i_L + i_R * D_bond_pad + rot * (D_bond_pad**2)).ravel()

    return pad_indices.astype(np.intp, copy=False)


def pad_block_diagonal_operator(
    W_ctrl: np.ndarray,
    D_bond: int,
    D_bond_pad: int,
    dim_rot: int = 2,
) -> np.ndarray:
    """
    Embeds the raw 2D block-diagonal W_ctrl into the padded Hilbert space
    required by Qiskit's qubit register layout.

    Parameters
    ----------
    W_ctrl : np.ndarray, shape (dim_lcu_A * dim_W_raw, dim_lcu_A * dim_W_raw)
        The raw controlled-unitary built in the unpadded space.
    D_bond : int
        Physical (unpadded) bond dimension.
    D_bond_pad : int
        Padded bond dimension (next power of 2 >= D_bond).
    dim_rot : int, optional
        Dimension of the rotation ancilla register. Default is 2.

    Returns
    -------
    W_ctrl_full : np.ndarray, shape (dim_lcu_A * dim_W_pad, dim_lcu_A * dim_W_pad)
        Padded controlled-unitary ready for UnitaryGate instantiation.

    Raises
    ------
    ValueError
        If W_ctrl shape is inconsistent with the given dimensions.
    """
    dim_W_raw = (D_bond**2) * dim_rot
    dim_W_pad = (D_bond_pad**2) * dim_rot

    if W_ctrl.shape[0] % dim_W_raw != 0:
        raise ValueError(
            f"W_ctrl rows ({W_ctrl.shape[0]}) not divisible by dim_W_raw ({dim_W_raw}). "
            f"Check D_bond={D_bond}, dim_rot={dim_rot}."
        )

    dim_lcu_A = W_ctrl.shape[0] // dim_W_raw
    pad_indices = compute_lcu_pad_indices(D_bond, D_bond_pad, dim_rot)

    W_ctrl_full = np.eye(dim_lcu_A * dim_W_pad, dtype=complex)

    for k in range(dim_lcu_A):
        offset_raw = k * dim_W_raw
        offset_pad = k * dim_W_pad

        raw_block = W_ctrl[
            offset_raw : offset_raw + dim_W_raw,
            offset_raw : offset_raw + dim_W_raw,
        ]

        target_indices = offset_pad + pad_indices
        W_ctrl_full[np.ix_(target_indices, target_indices)] = raw_block

    return W_ctrl_full
