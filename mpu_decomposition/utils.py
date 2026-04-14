# mpu_decomposition/utils.py
import numpy as np  # type: ignore
import quimb.tensor as qtn  # type: ignore
import scipy.linalg as la  # type: ignore
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
from itertools import product as iterproduct

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


def factored_lcu_decomposition(M_factors, basis_R, basis_L):
    """
    Decompose a factored operator M = Σ_m F_R[m] ⊗ F_L[m] into a
    given unitary basis on each side independently.

    Each factor is decomposed independently:
        F_R[m] = Σ_i h_R[m,i] * W_R[i]
        F_L[m] = Σ_j h_L[m,j] * W_L[j]

    The full operator is:
        M = Σ_m Σ_{i,j} h_R[m,i] * h_L[m,j] * (W_R[i] ⊗ W_L[j])

    The bond index m is NOT summed over in the returned data.
    This preserves the three-register structure needed for the circuit.

    Parameters
    ----------
    M_factors : list of tuples [(F_R_0, F_L_0), (F_R_1, F_L_1), ...]
    basis_R   : list of (dim_R, dim_R) unitary matrices
    basis_L   : list of (dim_L, dim_L) unitary matrices

    Returns
    -------
    h_R : (n_bonds, n_R) complex array
    h_L : (n_bonds, n_L) complex array
    """
    n_bonds = len(M_factors)
    n_R = len(basis_R)
    n_L = len(basis_L)
    dim_R = basis_R[0].shape[0]
    dim_L = basis_L[0].shape[0]

    h_R = np.zeros((n_bonds, n_R), dtype=complex)
    h_L = np.zeros((n_bonds, n_L), dtype=complex)

    for m in range(n_bonds):
        F_R_m, F_L_m = M_factors[m]
        for i, W_R in enumerate(basis_R):
            h_R[m, i] = np.trace(W_R.conj().T @ F_R_m) / dim_R
        for j, W_L in enumerate(basis_L):
            h_L[m, j] = np.trace(W_L.conj().T @ F_L_m) / dim_L

    return h_R, h_L


def build_merge_factors(R_inv, L_inv):
    """
    Build the factored form of M = Σ_m (|0><m| R_inv) ⊗ (|0><m| L_inv).
    Padded to dim = 2^n.
    """
    D = R_inv.shape[0]
    n_anc = int(np.ceil(np.log2(max(D, 2))))
    dim = 2**n_anc

    e0 = np.zeros(D, dtype=complex)
    e0[0] = 1.0

    factors = []
    for m in range(D):
        F_R = np.zeros((dim, dim), dtype=complex)
        F_R[:D, :D] = np.outer(e0, R_inv[m, :])

        F_L = np.zeros((dim, dim), dtype=complex)
        F_L[:D, :D] = np.outer(e0, L_inv[m, :])

        factors.append((F_R, F_L))

    return factors, n_anc


def pauli_basis(n):
    Id = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    paulis = [Id, X, Y, Z]
    names = ["I", "X", "Y", "Z"]
    basis = []
    labels = []
    for indices in iterproduct(range(4), repeat=n):
        mat = np.array([[1.0]], dtype=complex)
        label = ""
        for idx in indices:
            mat = np.kron(mat, paulis[idx])
            label += names[idx]
        basis.append(mat)
        labels.append(label)
    return basis, labels


def get_merging_operator(L_inv: np.ndarray, R_inv: np.ndarray) -> np.ndarray:
    """
    Computes the merging operator M from the boundary operators L2 and R2.
    This operator is crucial for the synthesis of the MPU into a quantum circuit.
    The merging operator encapsulates the entanglement structure at the boundaries
    and is derived from the inverses of L2 and R2.
    """
    # Validate input dimensions
    if L_inv.shape != R_inv.shape or L_inv.shape[0] != L_inv.shape[1]:
        raise ValueError("L2 and R2 must be square matrices of the same dimension.")

    # Compute the merging operator M = |0,0⟩⟨I|(R⁻¹ ⊗ L⁻¹)
    try:
        D = L_inv.shape[0]

        M = np.zeros((D, D, D, D), dtype=complex)

        kernel = R_inv.T @ L_inv  # (D, D)
        M[0, 0, :, :] = kernel

        return M
    except la.LinAlgError as e:
        raise ValueError(f"Failed to compute merging operator: {e}")
