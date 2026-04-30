import numpy as np  # pyright: ignore[reportMissingImports]
import quimb.tensor as qtn  # type: ignore
from abc import ABC, abstractmethod
from typing import List
import scipy.linalg as la
from scipy.linalg import qr

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
    def _compute_lcu(M: np.ndarray) -> tuple[list, list, float]:
        """
        Computes the LCU decomposition M = sum_m c_m W_m using SVD and
        root-of-unity phases (inverse DFT over singular values).
        """
        D_sq = M.shape[0]

        # SVD
        U, S, Vh = np.linalg.svd(M, full_matrices=True)
        S = np.real(np.asarray(S).ravel())
        K = len(S)

        omega = np.exp(2j * np.pi / D_sq)

        coefficients = []
        unitaries = []

        for m in range(D_sq):
            # 1. Calcolo del coefficiente complesso esatto (nessun np.real)
            c_m_complex = sum(S[k] * omega ** (-m * k) for k in range(K)) / D_sq

            # 2. Il peso LCU deve essere la magnitudo (reale, >= 0)
            magnitude = np.abs(c_m_complex)

            if magnitude < 1e-14:
                continue

            # 3. Estrazione della fase
            phase = c_m_complex / magnitude

            # 4. Costruzione dell'unitaria di base
            diag_vals = np.array([omega ** (m * j) for j in range(D_sq)], dtype=complex)
            W_m = U @ np.diag(diag_vals) @ Vh

            # 5. Assorbimento della fase nell'unitaria
            W_m_absorbed = phase * W_m

            coefficients.append(float(magnitude))
            unitaries.append(W_m_absorbed)

        return coefficients, unitaries, np.sum(coefficients)

    @staticmethod
    def _compute_merging_unitary(coefficients, unitaries, C):
        """
        Builds the two components of the merging unitary U = B† W_ctrl B.

        Given an LCU decomposition of the merging operator M = sum_i c_i W_i,
        post-selecting the ancilla on |0> after applying U implements M on any
        state in the image of the isometry V.

        Parameters
        ----------
        coefficients : array-like, shape (K,)
            Positive real coefficients c_i in M = sum_i c_i W_i.
        unitaries : list of np.ndarray, each shape (dim_system, dim_system)
            Unitary matrices W_i in the LCU decomposition.
        C: float, sum of all coefficients

        Returns
        -------
        B : np.ndarray, shape (dim_ancilla, dim_ancilla)
            Ancilla state-preparation unitary satisfying B|0> = (1/C) sum_i sqrt(c_i) |i>.
        W_ctrl : np.ndarray, shape (dim_system * dim_ancilla, dim_system * dim_ancilla)
            Controlled unitary W_ctrl = sum_i (W_i)_S ⊗ |i><i|_A.
        """
        K = len(coefficients)
        dim_ancilla = 2 ** int(np.ceil(np.log2(K)))
        dim_system = int(unitaries[0].shape[0])

        # ------------------------------------------------------------------
        # Build B: unitary that prepares the ancilla state (Eq. A33)
        #   B|0>_A = (1/C) * sum_i sqrt(c_i) |i>_A
        # ------------------------------------------------------------------
        first_col = np.zeros(dim_ancilla, dtype=complex)
        first_col[:K] = np.sqrt(coefficients) / np.sqrt(C)

        M = np.eye(dim_ancilla, dtype=complex)
        M[:, 0] = first_col
        Q, R = qr(M)

        diag_R = np.diag(R)
        phases = np.where(np.abs(diag_R) > 1e-14, diag_R / np.abs(diag_R), 1.0)
        B = Q @ np.diag(phases)

        # ------------------------------------------------------------------
        # Build W_ctrl: controlled unitary (Eq. A34)
        #   W_ctrl = sum_i (W_i)_S ⊗ |i><i|_A
        # For i >= K (unused ancilla states), identity is applied on the system.
        # ------------------------------------------------------------------
        dim_total = dim_system * dim_ancilla
        W_ctrl = np.zeros((dim_total, dim_total), dtype=complex)
        for i in range(dim_ancilla):
            proj = np.zeros((dim_ancilla, dim_ancilla), dtype=complex)
            proj[i, i] = 1.0
            W_i = (
                np.array(unitaries[i], dtype=complex)
                if i < K
                else np.eye(dim_system, dtype=complex)
            )
            W_ctrl += np.kron(proj, W_i)

        return B, W_ctrl

    @staticmethod
    def _compute_rotation_params(C: float) -> tuple[int, float]:
        """
        Compute ell and phi such that (2*ell+1)*theta = pi/2 exactly (Eq. A47).

        Parameters
        ----------
        C : float
            LCU norm of the merging operator.

        Returns
        -------
        ell : int
        phi : float
        """
        arg1 = np.clip(1.0 / C, -1.0, 1.0)
        ell = int(np.ceil(np.pi / (4 * np.arcsin(arg1)) - 0.5))
        C_prime = 1.0 / np.sin(np.pi / (4 * ell + 2))

        arg2 = np.clip(C_prime / (C * np.sqrt(2)), -1.0, 1.0)
        phi = np.arcsin(arg2) - np.pi / 4
        return ell, phi

    @staticmethod
    def _build_rotated_lcu(
        coef: list, unit: list, C: float, phi: float
    ) -> tuple[list, list, float]:
        """
        Extend M to M' = M ⊗ (cos(phi)*I + i*sin(phi)*Z).
        Makes the Grover exponent ell an integer.
        """
        Id = np.eye(2, dtype=complex)
        Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
        phase_unitaries = [Id, 1j * Z]
        phase_coeffs = [np.cos(phi), np.sin(phi)]

        coef_p, unit_p = [], []
        for c_i, W_i in zip(coef, unit):
            for p_j, V_j in zip(phase_coeffs, phase_unitaries):
                raw = c_i * p_j
                mag = np.abs(raw)
                if mag < 1e-14:
                    continue
                coef_p.append(float(mag))
                unit_p.append(np.kron(W_i, (raw / mag) * V_j))

        return coef_p, unit_p, sum(coef_p)

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
        debug: bool = False,
    ):
        """
        Args:
            A: The bulk numpy array (d, d, D, D).
            l_vec: Left boundary vector.
            r_vec: Right boundary vector.
            N: Total number of physical sites.
            tol: Numerical tolerance for structural checks.
            debug: if True gives info and sanity checks.
        """
        # --- 1. Theoretical Validations (Fail-Fast Logic) ---

        # Check global unitary property via local left-canonical isometry condition
        check_mpo_unitarity(N, A, l_in=l_vec, r_in=r_vec, tol=tol)

        # Check injectivity / minimal bond dimension at the boundaries
        s_left, s_right = check_assumption_1(A, l_vec, r_vec, tol=tol)

        # --- 2. State Encapsulation ---

        # Metadata and Dimensions
        self.tol = tol
        self._D = A.shape[2]  # Bond dimension
        self._d = A.shape[0]

        # Convert raw numpy arrays into quimb Tensors for the graph
        self.A = qtn.Tensor(
            data=A, inds=("p_out", "p_in", "bond_0", "bond_N"), tags=["A"]
        )

        super().__init__(N=N, d=self._d, l_vec=l_vec, r_vec=r_vec)

        # --- 3. Derived Quantum Properties ---
        self.L, self.R = self._compute_boundary_operators()
        self.L_inv = np.linalg.inv(self.L)
        self.R_inv = np.linalg.inv(self.R)

        self.MergingOperator = self._get_merging_operator()
        self.q_unif = self._compute_q_unif()

        self._lcu_cache = None
        self._merging_unitary_cache = None
        self.ell = None
        self.C = None

        if debug:
            self._verify_lcu()
            self._verify_merging_unitary()

    def create_local_isometries(self) -> tuple[qtn.Tensor, qtn.Tensor, qtn.Tensor]:
        """
        Constructs the local macro-isometries (V_l, V_b, V_r) for a 2-site block
        by contracting the site tensors with boundary vectors and unitary matrices.

        Returns
        -------
        tuple[qtn.Tensor, qtn.Tensor, qtn.Tensor]
            The left, bulk, and right isometric tensor networks.
        """
        A_data = self.A.data

        # Unpack MPO shape. Assuming (p_out, p_in, bond_L, bond_R) convention.
        # D_bond_pad computation removed as it is dead code in this function.
        D_sys, _, D_bond, _ = A_data.shape

        # V_joint = V1 ⊗ V2 implies a 2-macro-site logic
        N_sites_eff = self._N

        # Extract site tensors (returns qtn.Tensor objects)
        A1_tn, _ = get_mpo_site_tensors(1, A_data)
        A2_tn, _ = get_mpo_site_tensors(2, A_data)

        # Construct boundary tensors for the internal virtual bond ("bond_1")
        L_tn = qtn.Tensor(self.L, inds=["ext_L", "bond_1"])
        R_tn = qtn.Tensor(self.R, inds=["ext_R", "bond_1"])

        # --- Tensor Contractions ---
        # The @ operator in quimb automatically contracts shared indices.

        # Left boundary isometry
        V_l = self.l_vec @ A1_tn @ R_tn

        # Right boundary isometry
        # A2_tn indices: [p_out, p_in, bond_1, bond_2].
        # Reindex the right boundary vector to contract with "bond_2".
        V_r = L_tn @ A2_tn @ self.r_vec.reindex({f"bond_{N_sites_eff}": "bond_2"})

        # Bulk isometry
        # Reindex R_tn so it caps the right physical bond ("bond_2") of A2_tn.
        V_b = L_tn @ A2_tn @ R_tn.reindex({"bond_1": "bond_2"})

        return V_l, V_b, V_r

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
        """
        if self.L_inv.shape != self.R_inv.shape:
            raise ValueError(
                f"Dimension mismatch: L2 {self.L_inv.shape} vs R2 {self.R_inv.shape}"
            )

        if (
            np.real(np.trace(self.L @ self.L)) < 0
            or np.real(np.trace(self.R @ self.R)) < 0
        ):
            raise ValueError(
                "Unphysical negative trace detected in boundary operators."
            )

        L2_inv = self.L_inv @ self.L_inv
        R2_inv = self.R_inv @ self.R_inv

        q_unif = np.sqrt(np.trace(R2_inv @ L2_inv.T))

        return float(np.real(q_unif))

    def _get_merging_operator(self) -> np.ndarray:
        # Call the parent method using the instance attributes
        return get_merging_operator(self.L_inv, self.R_inv)

    def _get_rotation_params(self) -> tuple[int, float]:
        """
        Compute ell and phi for the uniform merging operator.
        Wrapper around the abstract _compute_rotation_angle using cached C.
        """
        _, _, C = self._build_lcu_data()
        return CircuitDecomposition._compute_rotation_params(C)

    def _build_lcu_data(self):
        """
        Uniform case: the merge operator is the same at every bond.
        Computed once, cached, and reused for every merge in the tree.
        Uses the optimal SVD-based LCU decomposition with root-of-unity phases.
        """
        if self._lcu_cache is None:
            coeffs, units, C = CircuitDecomposition._compute_lcu(self.MergingOperator)
            self._lcu_cache = (coeffs, units, C)
        return self._lcu_cache

    def _build_merging_unitary(self):
        """
        Uniform case: the merge operator is the same at every bond.
        Computed once, cached, and reused for every merge in the tree.
        Gives the components W_ctrl and B to implement the LCU.
        """
        if self._merging_unitary_cache is None:
            coeffs, units, C = self._build_lcu_data()
            B, W_ctrl = CircuitDecomposition._compute_merging_unitary(coeffs, units, C)
            self._merging_unitary_cache = (B, W_ctrl)
        return self._merging_unitary_cache

    def synthesize(self) -> qtn.TensorNetwork:
        """
        Placeholder per la scomposizione in circuiti (Styliaris 2025).
        """
        raise NotImplementedError("Implementazione di synthesize in corso...")
