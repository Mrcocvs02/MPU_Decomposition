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

    def __init__(self, N: int, d: int, l_vec: qtn.Tensor, r_vec: qtn.Tensor):
        self._N = N
        self._d = d
        self.l_vec = l_vec
        self.r_vec = r_vec

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

        super().__init__(
            N=N,
            d=physical_dim,
            l_vec=qtn.Tensor(l_vec, inds=["bond_0"], tags=["l"]),
            r_vec=qtn.Tensor(r_vec, inds=[f"bond_{N}"], tags=["r"]),
        )

        # --- 3. Derived Quantum Properties ---
        self.L2, self.R2 = self._compute_boundary_operators()
        self.q_unif = self._compute_q_unif()

    def _compute_boundary_operators(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Extracts the L2 and R2 boundary operators as matrices via TN contractions.
        L2 acts on the first bond index (bond_1), R2 on the last (bond_{N-1}).
        """

        # --- 1. Left Boundary Contraction (L2) ---
        # Prepare the conjugated left boundary vector
        left_boundary_conj = self.l_vec.H.reindex({"bond_0": "bond_0_dag"}).retag(
            {"l": "l*"}
        )

        # Retrieve site tensors for the first MPO position
        a1_tensor, a1_tensor_conj = get_mpo_site_tensors(1, self.A.data)

        # Build the L2 network: A_1, l, l*, A_1* and the identity on the physical leg
        l2_network = qtn.TensorNetwork(
            [a1_tensor, self.l_vec, left_boundary_conj, a1_tensor_conj]
        )
        l2_network &= qtn.Tensor(
            np.eye(self._d), inds=["p_in_1", "p_out_f_1"], tags=["σ"]
        )

        # --- 2. Right Boundary Contraction (R2) ---
        # Prepare the conjugated right boundary vector
        right_boundary_conj = self.r_vec.H.reindex(
            {f"bond_{self._N}": f"bond_{self._N}_dag"}
        ).retag({"r": "r*"})

        # Retrieve site tensors for the last MPO position
        an_tensor, an_tensor_conj = get_mpo_site_tensors(self._N, self.A.data)

        # Build the R2 network: A_N, r, r*, A_N* and the identity on the physical leg
        r2_network = qtn.TensorNetwork(
            [an_tensor, self.r_vec, right_boundary_conj, an_tensor_conj]
        )
        r2_network &= qtn.Tensor(
            np.eye(self._d), inds=[f"p_in_{self._N}", f"p_out_f_{self._N}"], tags=["τ"]
        )

        # --- 3. Matrix Extraction & Dimensionality Checks ---
        # Contract the networks to obtain matrices in Bond Space (D x D)
        l2_matrix = (l2_network ^ ...).data
        r2_matrix = (r2_network ^ ...).data

        # Verify physical validity (positive trace requirement for isometry/unitarity)
        trace_l2 = np.real(np.trace(l2_matrix))
        trace_r2 = np.real(np.trace(r2_matrix))

        if trace_l2 < 1e-15 or trace_r2 < 1e-15:
            raise ValueError(
                f"Unphysical non-positive trace detected (L:{trace_l2:.2e}, R:{trace_r2:.2e}). "
                "Check MPO tensors and boundary vectors."
            )

        # Structural validation of the resulting matrices
        validation_suite = [
            ("L2 Rank", l2_matrix.ndim == 2),
            ("R2 Rank", r2_matrix.ndim == 2),
            ("L2 Shape", l2_matrix.shape == (self._D, self._D)),
            ("R2 Shape", r2_matrix.shape == (self._D, self._D)),
        ]

        failed_checks = [label for label, passed in validation_suite if not passed]
        if failed_checks:
            raise ValueError(
                f"Boundary operator contraction failed: {', '.join(failed_checks)}. "
                f"Verify index naming logic in get_mpo_site_tensors."
            )

        return l2_matrix, r2_matrix

    def _compute_q_unif(self) -> float:
        r"""
        Calculates the entangling power $q_{unif}$ for a Translationally Invariant (TI) MPU.
        Formula: $q_{unif} = \sqrt{d} \sqrt{\text{Tr}(\sigma^{-1}) \text{Tr}(\tau^{-1})}$
        Reference: Eq. 54/127, Styliaris (2025).
        """

        # --- 1. Validation of Boundary Operators ---
        l2_matrix = self.L2.data
        r2_matrix = self.R2.data

        if (
            l2_matrix.shape[0] != l2_matrix.shape[1]
            or r2_matrix.shape[0] != r2_matrix.shape[1]
        ):
            raise ValueError("Boundary operators L2 and R2 must be square matrices.")

        if l2_matrix.shape != r2_matrix.shape:
            raise ValueError(
                f"Dimension mismatch: L2 {l2_matrix.shape} vs R2 {r2_matrix.shape}"
            )

        # --- 2. Trace Analysis & Physicality Check ---
        trace_l2 = np.real(np.trace(l2_matrix))
        trace_r2 = np.real(np.trace(r2_matrix))

        # Thresholding to detect violation of physical unitarity/isometry conditions
        if trace_l2 < 1e-15 or trace_r2 < 1e-15:
            raise ValueError(
                f"Unphysical negative trace (L:{trace_l2:.2e}, R:{trace_r2:.2e}) encountered."
            )

        # --- 3. Normalization to Density Operators ---
        # Rescale boundary operators to unit trace: sigma, tau \in Bond Space (D x D)
        sigma = l2_matrix / trace_l2
        tau = r2_matrix / trace_r2

        # --- 4. Stable Pseudo-Inversion ---
        # Using pinv with fixed rcond to handle potential ill-conditioning in bond space
        sigma_inv = np.linalg.pinv(sigma, rcond=1e-13)
        tau_inv = np.linalg.pinv(tau, rcond=1e-13)

        # --- 5. Final q_unif Computation ---
        # For identity channels (D=1), the result scales with the local dimension d
        q_value = np.sqrt(self._d) * np.sqrt(np.trace(sigma_inv) * np.trace(tau_inv))

        return float(np.real(q_value))

    def synthesize(self) -> qtn.TensorNetwork:
        """
        Placeholder per la scomposizione in circuiti (Styliaris 2025).
        """
        raise NotImplementedError("Implementazione di synthesize in corso...")
