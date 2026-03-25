# mpu_decomposition/utils.py
import numpy as np  # type: ignore
import quimb.tensor as qtn  # type: ignore


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
