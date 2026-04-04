import numpy as np  # pyright: ignore[reportMissingImports]
import quimb.tensor as qtn  # type: ignore
from abc import ABC, abstractmethod
from typing import List

# Assuming your pure mathematical validation functions are in checks.py
from .checks import check_assumption_1, check_mpo_unitarity
from .utils import get_mpo_site_tensors


class AbstractMPU(ABC):
    """
    Abstract Base Class for Matrix-Product Unitaries (MPUs).
    Enforces a strict API for compiling the MPU into a quantum circuit.
    """

    def __init__(self, N: int, d: int, l_vec: np.ndarray, r_vec: np.ndarray):
        self._N = N
        self._d = d
        self.l_vec = qtn.Tensor(l_vec, inds=["bond_0"], tags=["l"])
        self.r_vec = qtn.Tensor(r_vec, inds=[f"bond_{N}"], tags=["r"])

    @abstractmethod
    def synthesize(self) -> qtn.TensorNetwork:
        pass


class GeneralMPU(AbstractMPU):
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


class UniformMPU(AbstractMPU):
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
        self.L2, self.R2 = self._compute_boundary_operators()
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
        L_2 = A_1 & self.l_vec & l_vec_conj & A_1_adj

        L_2 &= qtn.Tensor(
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
        R_2 = A_N & self.r_vec & r_vec_conj & A_N_adj

        R_2 &= qtn.Tensor(
            np.eye(self._d) / self._d,
            inds=[f"p_in_{self._N}", f"p_out_f_{self._N}"],
            tags=["τ"],
        )

        # --- 3. Matrix Extraction & Dimensionality Checks ---
        # Contract the networks to obtain matrices in Bond Space (D x D)
        L_2_matrix = (L_2 ^ ...).data
        R_2_matrix = (R_2 ^ ...).data

        # Verify physical validity (positive trace requirement for isometry/unitarity)
        trace_L_2 = np.real(np.trace(L_2_matrix))
        trace_R_2 = np.real(np.trace(R_2_matrix))

        if trace_L_2 < 1e-15 or trace_R_2 < 1e-15:
            raise ValueError(
                f"Unphysical non-positive trace detected (L:{trace_L_2:.2e}, R:{trace_R_2:.2e}). "
                "Check MPO tensors and boundary vectors."
            )

        # Structural validation of the resulting matrices
        validation = [
            ("L2 Rank", L_2_matrix.ndim == 2),
            ("R2 Rank", R_2_matrix.ndim == 2),
            ("L2 Shape", L_2_matrix.shape == (self._D, self._D)),
            ("R2 Shape", R_2_matrix.shape == (self._D, self._D)),
        ]

        failed_checks = [label for label, passed in validation if not passed]
        if failed_checks:
            raise ValueError(
                f"Boundary operator contraction failed: {', '.join(failed_checks)}. "
                f"Verify index naming logic in get_mpo_site_tensors."
            )

        return L_2_matrix, R_2_matrix

    def _compute_q_unif(self) -> float:
        r"""
        Calculates the entangling power $q_{unif}$ for a Translationally Invariant (TI) MPU.
        Formula: $q_{unif} = \sqrt{\text{Tr}(L_2^{-1}) \text{Tr}(R_2^{-1})}$
        Reference: Eq. 54/127, Styliaris (2025).
        """

        # --- 1. Validation of Boundary Operators ---
        L_2_matrix = self.L2.data
        R_2_matrix = self.R2.data

        if (
            L_2_matrix.shape[0] != L_2_matrix.shape[1]
            or R_2_matrix.shape[0] != R_2_matrix.shape[1]
        ):
            raise ValueError("Boundary operators L2 and R2 must be square matrices.")

        if L_2_matrix.shape != R_2_matrix.shape:
            raise ValueError(
                f"Dimension mismatch: L2 {L_2_matrix.shape} vs R2 {R_2_matrix.shape}"
            )

        if np.real(np.trace(self.L2)) < 0 or np.real(np.trace(self.R2)) < 0:
            raise ValueError(
                "Unphysical negative trace detected in boundary operators."
            )

        # --- 4. Stable Pseudo-Inversion ---
        # Using pinv with fixed rcond to handle potential ill-conditioning in bond space
        L_2_inv = np.linalg.pinv(L_2_matrix, rcond=1e-13)
        R_2_inv = np.linalg.pinv(R_2_matrix, rcond=1e-13)

        # --- 5. Final q_unif Computation ---
        # For identity channels (D=1), the result scales with the local dimension d
        q_unif = np.sqrt(np.trace(R_2_inv @ L_2_inv.T))

        return float(np.real(q_unif))

    def synthesize(self) -> qtn.TensorNetwork:
        """
        Placeholder per la scomposizione in circuiti (Styliaris 2025).
        """
        raise NotImplementedError("Implementazione di synthesize in corso...")
