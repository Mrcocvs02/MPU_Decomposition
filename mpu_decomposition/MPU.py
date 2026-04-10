import numpy as np  # pyright: ignore[reportMissingImports]
import quimb.tensor as qtn  # type: ignore
from abc import ABC, abstractmethod
from typing import List
import scipy.linalg as la

# Assuming your pure mathematical validation functions are in checks.py
from .checks import check_assumption_1, check_mpo_unitarity
from .utils import get_mpo_site_tensors


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

    @staticmethod
    def _build_lcu_data(D, R_inv: np.ndarray, L_inv: np.ndarray):
        """
        Decompose the merge operator into factored LCU data.

        The merge operator is:
            M = Σ_m (|0><m| R⁻¹) ⊗ (|0><m| L⁻¹)

        Each factor |0><m| R⁻¹ is a D×D rank-1 matrix. By Lemma 2,
        any rank-1 matrix s₀|u₀><v₀| can be written as a sum of D
        unitaries using the D-th roots of unity:

            s₀|u₀><v₀| = Σ_k (s₀/D) U Z_k V†

        where Z_k = diag(1, ω^k, ω^{2k}, ..., ω^{(D-1)k}) and ω = e^{2πi/D}.
        This works because (1/D) Σ_k ω^{nk} = δ_{n,0}, so the sum over k
        projects out the first singular vector.

        The factored form means we decompose the right and left factors
        independently, then pair them. This gives D³ total terms:
        D bond indices × D right unitaries × D left unitaries.

        Parameters
        ----------
        D     : bond dimension
        R_inv : D×D matrix, inverse of right boundary operator R
        L_inv : D×D matrix, inverse of left boundary operator L

        Returns
        -------
        c_k      : 1D real positive array of LCU coefficients, length d_anc
        U_R_list : list of D×D unitary matrices for the right ancilla register
        U_L_list : list of D×D unitary matrices for the left ancilla register
        C        : normalization constant Σ c_k = ||M||₁
        target   : normalized state preparation vector √c_k / ||√c_k||
        """

        # ---- Setup ----
        # |0> in the D-dimensional bond space
        e0 = np.zeros(D, dtype=complex)
        e0[0] = 1.0

        # Primitive D-th root of unity: ω = e^{2πi/D}
        omega = np.exp(2j * np.pi / D)

        # Exponent array [0, 1, 2, ..., D-1] used to build Z_k diagonals
        powers = np.arange(D)

        all_coeffs = []
        all_U_R = []
        all_U_L = []

        # ---- Main loop over bond index m ----
        for m in range(D):
            # Build the rank-1 factors for this bond index:
            #   F_R[m] = |0><m| R⁻¹   (D×D, rank 1)
            #   F_L[m] = |0><m| L⁻¹   (D×D, rank 1)
            F_R_m = np.outer(e0, R_inv[m, :])
            F_L_m = np.outer(e0, L_inv[m, :])

            # SVD of each rank-1 factor:
            #   F = U S V†  with only s[0] nonzero
            u_r, s_r, vh_r = np.linalg.svd(F_R_m, full_matrices=True)
            u_l, s_l, vh_l = np.linalg.svd(F_L_m, full_matrices=True)

            # Each factor gets coefficient s₀/D from the roots-of-unity identity.
            # The paired coefficient is the product of the two.
            c_r = s_r[0] / D
            c_l = s_l[0] / D
            c_rl = c_r * c_l

            # ---- Roots-of-unity decomposition of each factor ----
            # For each k in [0, D):
            #   Z_k = diag(ω^0, ω^k, ω^{2k}, ..., ω^{(D-1)k})
            #   W_k = U Z_k V†
            #
            # Then (1/D) Σ_k W_k = U diag(1,0,...,0) V† = |u₀><v₀|
            # because Σ_k ω^{nk} = D·δ_{n,0}

            for k in range(D):
                # Diagonal of Z_k for the right factor: [ω^0, ω^k, ω^{2k}, ...]
                zk = np.power(omega, powers * k)
                # W_R_k = U_R · Z_k · V_R†
                # The multiplication u_r * zk scales each column of U by the
                # corresponding diagonal entry of Z_k
                Wr_k = (u_r * zk) @ vh_r

                for j in range(D):
                    # Same construction for the left factor with index j
                    zj = np.power(omega, powers * j)
                    Wl_j = (u_l * zj) @ vh_l

                    # This (k,j) pair contributes one term to the LCU:
                    #   c_rl · (Wr_k ⊗ Wl_j)
                    # The kron is NOT computed here — we store the two
                    # unitaries separately for parallel circuit execution
                    all_coeffs.append(c_rl)
                    all_U_R.append(Wr_k)
                    all_U_L.append(Wl_j)

        # ---- Post-processing ----
        # All coefficients are real positive by construction:
        # each is a product of two singular values divided by D²
        all_coeffs = np.array(all_coeffs, dtype=float)
        C = np.sum(all_coeffs)

        # The LCU ancilla register needs enough states to index every term.
        # Pad to the next power of two for qubit-based implementation.
        N_terms = len(all_coeffs)
        n_anc = int(np.ceil(np.log2(max(N_terms, 2))))
        d_anc = 2**n_anc
        pad = d_anc - N_terms

        # Padded entries get identity unitaries and zero coefficients.
        # They contribute nothing to M but fill the ancilla Hilbert space.
        I_D = np.eye(D, dtype=complex)
        all_U_R.extend([I_D] * pad)
        all_U_L.extend([I_D] * pad)
        all_coeffs = np.append(all_coeffs, np.zeros(pad))

        # State preparation vector for B|0> = (1/||√c||) Σ_k √c_k |k>
        # After applying B, postselecting ancilla on |0> and undoing B
        # gives the correct LCU weighting.
        target = np.sqrt(all_coeffs)
        target = target / np.linalg.norm(target)

        return all_coeffs, all_U_R, all_U_L, C, target

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
        return CircuitDecomposition.get_merging_operator(self.L_inv, self.R_inv)

    def _build_lcu_data(self):
        """
        Uniform case: the merge operator is the same at every bond.
        Computed once, cached, and reused for every merge in the tree.
        """
        if self._lcu_cache is None:
            self._lcu_cache = CircuitDecomposition._build_lcu_data(
                self._D,
                self.R_inv,
                self.L_inv,  # CORRETTO
            )
        return self._lcu_cache

    def synthesize(self) -> qtn.TensorNetwork:
        """
        Placeholder per la scomposizione in circuiti (Styliaris 2025).
        """
        raise NotImplementedError("Implementazione di synthesize in corso...")
