# mpu_decomposition/utils.py
import numpy as np  # type: ignore
import quimb.tensor as qtn  # type: ignore
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
