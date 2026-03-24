import numpy as np
import quimb.tensor as qtn
from abc import ABC, abstractmethod
from typing import List, Union

# Assuming your pure mathematical validation functions are in checks.py
from .checks import check_assumption_1, check_mpo_unitarity
from .utils import get_mpo_site_tensors



class AbstractMPU(ABC):
    """
    Abstract Base Class for Matrix-Product Unitaries (MPUs).
    Enforces a strict API for compiling the MPU into a quantum circuit.
    """
    def __init__(self, N: int, d: int, l: qtn.Tensor, r: qtn.Tensor):
        self._N = N
        self._d = d
        self.l = l
        self.r = r

    @abstractmethod

    def synthesize(self) -> qtn.TensorNetwork:
            pass


class GeneralMPU(AbstractMPU):
    """
    Encapsulates a heterogeneous (site-dependent) Matrix-Product Unitary.
    Defined by a strictly ordered sequence of local bulk tensors.
    """
    
    def __init__(self, tensors_A: List[qtn.Tensor], l: np.ndarray, r: np.ndarray):
        """
        Args:
            tensors_A: List of N site-dependent bulk tensors [A_1, A_2, ..., A_N].
            l: Left boundary vector.
            r: Right boundary vector.
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
        self.l = l
        self.r = r
        self._N = len(tensors_A)

    def synthesize(self) -> qtn.TensorNetwork:
        # Choi-state mapping and synthesis logic goes here
        pass



class UniformMPU(AbstractMPU):
    """
    Encapsulates a translationally invariant Matrix-Product Unitary.
    Defined by a single repeating bulk tensor A.
    """
    def __init__(self, A: np.ndarray, l: np.ndarray, r: np.ndarray, N: int, tol: float = 1e-12):    
        """
        Args:
            A: The bulk numpy array (d, d, D, D).
            l: Left boundary vector.
            r: Right boundary vector.
            N: Total number of physical sites.
            tol: Numerical tolerance for structural checks.
        """
        # 1. Theoretical Validations (Strict Fail-Fast Logic)
        
        # Check global unitary property via local left-canonical isometry condition
        if not check_mpo_unitarity(N, A, l_in=l, r_in=r, tol=tol):
            raise ValueError(
                "Instantiation aborted: The bulk tensor 'A' does not satisfy "
                "the isometry condition."
            )

        # Check injectivity / minimal bond dimension at the boundaries
        is_injective, s_left, s_right = check_assumption_1(A, l, r, tol=tol)

        if not is_injective:
            raise ValueError(
                f"Instantiation aborted: Boundary injectivity failed (Assumption 1).\n"
                f"Left Singular Values: {s_left}\n"
                f"Right Singular Values: {s_right}"
            )
            
        # 2. State Encapsulation (Executes ONLY if all checks pass)
        # Convert the validated raw numpy array into a quimb tensor for the graph
        
        self.A = qtn.Tensor(data=A, inds=('p_out', 'p_in', 'bond_0', 'bond_N'), tags=['A'])
        self._D = A.shape[2]  # Bond dimension

        super().__init__(N=N,
                        d=A.shape[0],
                        l=qtn.Tensor(l,inds=["bond_0"],tags=["l"]),
                        r=qtn.Tensor(r,inds=[f"bond_{N}"],tags=["r"])
                        )

        self.L2, self.R2 = self._compute_boundary_operators()
        self.q_unif = self._compute_q_unif()
        self.tol = tol
    
    
    def _compute_boundary_operators(self, verbose: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Performs TN contractions to extract L2 and R2 as matrices.
        L2 lives on bond_1, R2 lives on bond_{N-1}.
        """
    
        l_s = self.l.H.reindex({"bond_0": "bond_0_dag"}).retag({"l": "l*"})
        A_1, A_1_dag = get_mpo_site_tensors(1, self. A.data)
        
        L2_tn = qtn.TensorNetwork([A_1, self.l, l_s, A_1_dag])
        L2_tn &= qtn.Tensor(np.eye( self._d), inds=["p_in_1", "p_out_f_1"], tags=["σ"])    #for the identity this is simply dividing by d but in general is not so we leave it for now

        r_s = self.r.H.reindex({f"bond_{ self._N}": f"bond_{ self._N}_dag"}).retag({"r": "r*"})
        A_N, A_N_dag = get_mpo_site_tensors( self._N, self. A.data)

        R2_tn = qtn.TensorNetwork([A_N, self.r, r_s, A_N_dag])
        R2_tn &= qtn.Tensor(np.eye( self._d), inds=[f"p_in_{ self._N}", f"p_out_f_{ self._N}"], tags=["τ"])

        L2_mat = L2_tn ^ ...
        R2_mat = R2_tn ^ ...

        checks = [
            ("L2 Rank", L2_mat.ndim == 2),
            ("R2 Rank", R2_mat.ndim == 2),
            ("L2 Shape", L2_mat.shape == ( self._d,  self._d)),
            ("R2 Shape", R2_mat.shape == ( self._d,  self._d)),
        ]

        failed = [name for name, passed in checks if not passed]
        if failed:
            raise ValueError(f"Boundary operator contraction failed: {', '.join(failed)}. "
                             f"Check index naming in get_mpo_site_tensors.")

        if verbose:
            print("--- Boundary Operators Report ---")
            print(f"  - L2: Shape={L2_mat.shape}, Norm={np.linalg.norm(L2_mat):.4e}")
            print(f"  - R2: Shape={R2_mat.shape}, Norm={np.linalg.norm(R2_mat):.4e}")
            print(f"  - L2 Eigenvalues: {np.linalg.eigvals(L2_mat)}")
            print(f"  - R2 Eigenvalues: {np.linalg.eigvals(R2_mat)}")
            print("---------------------------------")
        return L2_mat, R2_mat
    
    def _compute_q_unif(self, verbose: bool = False) -> float:
        
        r"Calculates $q_{unif} = \sqrt{\text{Tr}[R_2^{-1} (L_2^{-1})^T]}$"
        
        # 1. Structural Sanity Checks
        if self.L2.data.shape[0] != self.L2.data.shape[1] or self.R2.data.shape[0] != self.R2.data.shape[1]:
            raise ValueError("Boundary operators L2 and R2 must be square matrices.")
        
        if self.L2.data.shape != self.R2.data.shape:
            raise ValueError(f"Dimension mismatch: L2 {self.L2.data.shape} vs R2 {self.R2.data.shape}")

        # 2. Numerical Conditioning Check
        # High condition numbers suggest L2 or R2 are near-singular (Assumption 1 failure)
        cond_L = np.linalg.cond(self.L2.data)
        cond_R = np.linalg.cond(self.R2.data)
        
        # 3. Stable Inversion
        # Using pinv with the internal tolerance to handle potentially ill-conditioned boundaries
        L2_inv = np.linalg.pinv(self.L2.data, rcond=getattr(self, 'tol', 1e-13))
        R2_inv = np.linalg.pinv(self.R2.data, rcond=getattr(self, 'tol', 1e-13))
        
        M = R2_inv @ L2_inv.T
        tr_M = np.trace(M)
        
        # 4. Result Validation
        # Significant imaginary components or negative real parts indicate 
        # a breakdown of the MPO's spectral properties.
        imag_part = np.imag(tr_M)
        real_tr = np.real(tr_M)

        if abs(imag_part) > getattr(self, 'tol', 1e-12) and verbose:
            print(f"Warning: Non-negligible imaginary trace component: {imag_part:.2e}")
        
        if real_tr < -getattr(self, 'tol', 1e-12):
            raise ValueError(f"Unphysical negative trace ({real_tr:.4f}) encountered. Verify A and boundary vectors.")

        # Clip small negative values due to precision before sqrt
        q_val = float(np.sqrt(max(0.0, real_tr)))

        if verbose:
            print(f"--- q_unif Sanity Report ---")
            print(f"  - Matrix Shape:       {self.L2.shape}")
            print(f"  - L2 Condition No.:   {cond_L:.2e}")
            print(f"  - R2 Condition No.:   {cond_R:.2e}")
            print(f"  - Trace(M) Real:      {real_tr:.6f}")
            print(f"  - Trace(M) Imag:      {imag_part:.2e}")
            print(f"  - Final q_unif:       {q_val:.6f}")
            print(f"----------------------------")

        return q_val

    def synthesize(self) -> qtn.TensorNetwork:
        """
        Placeholder per la scomposizione in circuiti (Styliaris 2025).
        """
        raise NotImplementedError("Implementazione di synthesize in corso...")