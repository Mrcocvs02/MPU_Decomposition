import numpy as np  # pyright: ignore[reportMissingImports]
import quimb.tensor as qtn  # type: ignore
from abc import ABC, abstractmethod
from typing import List
import scipy.linalg as la

# Assuming your pure mathematical validation functions are in checks.py
from .checks import check_assumption_1, check_mpo_unitarity
from .utils import (
    get_mpo_site_tensors,
    get_merging_operator,
)


class CircuitDecomposition(ABC):
    """
    Abstract Base Class for Matrix-Product Unitaries (MPUs).
    Enforces a strict API for compiling the MPU into a quantum circuit.
    """

    def __init__(self, N: int, d: int, l_vec: np.ndarray, r_vec: np.ndarray):
        self._N = N
        self._d = d
        self.l_vec = qtn.Tensor(l_vec, inds=["bond_0"], tags=["l"])
        self.r_vec = qtn.Tensor(r_vec, inds=[f"bond_{N}"], tags=["r"])

    @staticmethod
    def _build_lcu_data(M: np.ndarray) -> tuple[list, list, float]:
        """
        Computes the optimal LCU decomposition of a merging operator M using SVD-based
        root-of-unity phase factors, returning all LCU terms as precomputed lists.

        The decomposition is:
            M = Σ_{k,m} c_{k,m} W_{k,m}
        where:
            - W_{k,m} = U @ diag(ω^{m*j}) @ Vh, with ω = exp(2πi / D²)
            - c_{k,m} = S_k * ω^{-m*k} / D²
        and S_k are the non-zero singular values of M.

        Parameters
        ----------
        M : np.ndarray
            The merging operator (D² x D²) to decompose.

        Returns
        -------
        coefficients : list of complex
            List of LCU coefficients c_{k,m}
        unitaries : list of np.ndarray
            List of unitary matrices W_{k,m} (each D² x D²)
        C : float
            The 1-norm of M, i.e., sum of singular values (||M||_1).
        """
        D_squared = M.shape[0] ** 2

        M = M.reshape(D_squared, D_squared)
        assert M.shape == (D_squared, D_squared), f"M must be D² x D², got {M.shape}"

        # SVD on the full D² x D² matrix
        U, S, Vh = np.linalg.svd(M, full_matrices=True)

        # Ensure S is 1D
        S = np.asarray(S).ravel()

        # Compute theoretical 1-norm (nuclear norm)
        C = np.sum(S[S >= 1e-14])

        # Root of unity: D²-th root, not D-th
        omega = np.exp(2j * np.pi / D_squared)

        coefficients = []
        unitaries = []

        for k in range(len(S)):
            if S[k] < 1e-14:
                continue
            for m in range(D_squared):
                diag_vals = np.array([omega ** (m * j) for j in range(D_squared)])
                W = U @ np.diag(diag_vals) @ Vh
                c = S[k] * (omega ** (-m * k)) / D_squared
                coefficients.append(c)
                unitaries.append(W)

        return coefficients, unitaries, C

    @abstractmethod
    def synthesize(self) -> qtn.TensorNetwork:
        pass


class GeneralMPU(CircuitDecomposition):
    """
    Encapsulates a heterogeneous (site-dependent) Matrix-Product Unitary.
    Defined by a strictly ordered sequence of local bulk tensors.
    """

    def __init__(
        self, tensors_A: List[qtn.Tensor], l_vec: np.ndarray, r_vec: np.ndarray
    ):
        """
        Args:
            tensors_A: List of N site-dependent bulk tensors [A_1, A_2, ..., A_N].
            l_vec: Left boundary vector.
            r_vec: Right boundary vector.
        """
        # 1. Theoretical Validation
        # Enforce Assumption 1 iteratively across all bipartitions
        for k, A_k in enumerate(tensors_A):
            try:
                check_assumption_1(A_k)
            except ValueError as e:
                raise ValueError(f"Tensor A_{k} violates minimal bond dimension: {e}")

        # 2. State Encapsulation
        self.tensors = tensors_A
        self.l_vec = l_vec
        self.r_vec = r_vec
        self._N = len(tensors_A)

    def synthesize(self) -> qtn.TensorNetwork:
        # Choi-state mapping and synthesis logic goes here
        pass


class UniformMPU(CircuitDecomposition):
    """
    Encapsulates a translationally invariant Matrix-Product Unitary.
    Defined by a single repeating bulk tensor A.
    """

    def __init__(
        self,
        A: np.ndarray,
        l_vec: np.ndarray,
        r_vec: np.ndarray,
        N: int,
        tol: float = 1e-12,
    ):
        """
        Args:
            A: The bulk numpy array (d, d, D, D).
            l_vec: Left boundary vector.
            r_vec: Right boundary vector.
            N: Total number of physical sites.
            tol: Numerical tolerance for structural checks.
        """
        # --- 1. Theoretical Validations (Fail-Fast Logic) ---

        # Check global unitary property via local left-canonical isometry condition
        if not check_mpo_unitarity(N, A, l_in=l_vec, r_in=r_vec, tol=tol):
            raise ValueError(
                "Instantiation aborted: The bulk tensor 'A' does not satisfy "
                "the isometry condition."
            )

        # Check injectivity / minimal bond dimension at the boundaries
        is_injective, s_left, s_right = check_assumption_1(A, l_vec, r_vec, tol=tol)

        if not is_injective:
            raise ValueError(
                f"Instantiation aborted: Boundary injectivity failed (Assumption 1).\n"
                f"Left Singular Values: {s_left}\n"
                f"Right Singular Values: {s_right}"
            )

        # --- 2. State Encapsulation ---

        # Metadata and Dimensions
        self.tol = tol
        self._D = A.shape[2]  # Bond dimension
        physical_dim = A.shape[0]

        # Convert raw numpy arrays into quimb Tensors for the graph
        self.A = qtn.Tensor(
            data=A, inds=("p_out", "p_in", "bond_0", "bond_N"), tags=["A"]
        )

        super().__init__(N=N, d=physical_dim, l_vec=l_vec, r_vec=r_vec)

        # --- 3. Derived Quantum Properties ---
        self.L, self.R = self._compute_boundary_operators()
        self.L_inv = np.linalg.inv(self.L)
        self.R_inv = np.linalg.inv(self.R)

        self._lcu_cache = None
        self.q_unif = self._compute_q_unif()

    def _compute_boundary_operators(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Extracts the L2 and R2 boundary operators as matrices via TN contractions.
        L2 acts on the first bond index (bond_0), R2 on the last (bond_{N}).
        The output are numpy array since we need to apply them to many different
        sites at once and it's easier with the raw matrix entries.
        """

        # --- 1. Left Boundary Contraction (L2) ---
        # Prepare the conjugated left boundary vector
        l_vec_conj = self.l_vec.H.reindex({"bond_0": "bond_0_dag"}).retag({"l": "l*"})

        # Retrieve site tensors for the first MPO position
        A_1, A_1_adj = get_mpo_site_tensors(1, self.A.data)

        # Build the L2 network: A_1, l, l*, A_1* and the identity on the physical leg
        L2 = A_1 & self.l_vec & l_vec_conj & A_1_adj

        L2 &= qtn.Tensor(
            np.eye(self._d) / self._d, inds=["p_in_1", "p_out_f_1"], tags=["σ"]
        )

        # --- 2. Right Boundary Contraction (R2) ---
        # Prepare the conjugated right boundary vector
        r_vec_conj = self.r_vec.H.reindex(
            {f"bond_{self._N}": f"bond_{self._N}_dag"}
        ).retag({"r": "r*"})

        # Retrieve site tensors for the last MPO position
        A_N, A_N_adj = get_mpo_site_tensors(self._N, self.A.data)

        # Build the R2 network: A_N, r, r*, A_N* and the identity on the physical leg
        R2 = A_N & self.r_vec & r_vec_conj & A_N_adj

        R2 &= qtn.Tensor(
            np.eye(self._d) / self._d,
            inds=[f"p_in_{self._N}", f"p_out_f_{self._N}"],
            tags=["τ"],
        )

        # --- 3. Matrix Extraction & Dimensionality Checks ---
        # Contract the networks to obtain matrices in Bond Space (D x D)
        L2_matrix = (L2 ^ ...).data
        R2_matrix = (R2 ^ ...).data

        # Verify physical validity (positive trace requirement for isometry/unitarity)
        trace_L2 = np.real(np.trace(L2_matrix))
        trace_R2 = np.real(np.trace(R2_matrix))

        if trace_L2 < 1e-15 or trace_R2 < 1e-15:
            raise ValueError(
                f"Unphysical non-positive trace detected (L:{trace_L2:.2e}, R:{trace_R2:.2e}). "
                "Check MPO tensors and boundary vectors."
            )

        # Structural validation of the resulting matrices
        validation = [
            ("L2 Rank", L2_matrix.ndim == 2),
            ("R2 Rank", R2_matrix.ndim == 2),
            ("L2 Shape", L2_matrix.shape == (self._D, self._D)),
            ("R2 Shape", R2_matrix.shape == (self._D, self._D)),
        ]

        failed_checks = [label for label, passed in validation if not passed]
        if failed_checks:
            raise ValueError(
                f"Boundary operator contraction failed: {', '.join(failed_checks)}. "
                f"Verify index naming logic in get_mpo_site_tensors."
            )

        eig_L, vec_L = np.linalg.eig(L2_matrix)
        eig_R, vec_R = np.linalg.eig(R2_matrix)

        sqrt_L = np.sqrt(eig_L)
        sqrt_R = np.sqrt(eig_R)

        L = vec_L @ np.diag(sqrt_L) @ la.inv(vec_L)
        R = vec_R @ np.diag(sqrt_R) @ la.inv(vec_R)

        return L, R

    def _compute_q_unif(self) -> float:
        r"""
        Calculates the entangling power $q_{unif}$ for a Translationally Invariant (TI) MPU.
        Formula: $q_{unif} = \sqrt{\text{Tr}(L_2^{-1}) \text{Tr}(R_2^{-1})}$
        Reference: Eq. 54/127, Styliaris (2025).
        """

        L2_inv = self.L_inv @ self.L_inv
        R2_inv = self.R_inv @ self.R_inv
        if L2_inv.shape[0] != L2_inv.shape[1] or R2_inv.shape[0] != R2_inv.shape[1]:
            raise ValueError("Boundary operators L2 and R2 must be square matrices.")

        if L2_inv.shape != R2_inv.shape:
            raise ValueError(
                f"Dimension mismatch: L2 {L2_inv.shape} vs R2 {R2_inv.shape}"
            )

        if np.real(np.trace(self.L)) < 0 or np.real(np.trace(self.R)) < 0:
            raise ValueError(
                "Unphysical negative trace detected in boundary operators."
            )

        q_unif = np.sqrt(np.trace(R2_inv @ L2_inv.T))

        return float(np.real(q_unif))

    def get_merging_operator(self) -> np.ndarray:
        # Call the parent method using the instance attributes
        return get_merging_operator(self.L_inv, self.R_inv)

    def _build_lcu_data(self):
        """
        Uniform case: the merge operator is the same at every bond.
        Computed once, cached, and reused for every merge in the tree.
        Uses the optimal SVD-based LCU decomposition with root-of-unity phases.
        """
        if self._lcu_cache is None:
            M = self.get_merging_operator()
            coeffs, units, C = CircuitDecomposition._build_lcu_data(M)
            self._lcu_cache = (coeffs, units, C)
        return self._lcu_cache

    def synthesize(self) -> qtn.TensorNetwork:
        """
        Placeholder per la scomposizione in circuiti (Styliaris 2025).
        """
        raise NotImplementedError("Implementazione di synthesize in corso...")
