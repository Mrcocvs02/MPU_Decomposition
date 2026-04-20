import pytest
import numpy as np
import quimb.tensor as qtn
from unittest.mock import patch
from mpu_decomposition.MPU import UniformMPU


ALL_LCU_DATA = ["identity_lcu_data", "cz_lcu_data", "semisimple_lcu_data"]


# =====================================================================
# Group 2: UniformMPU Constructor and Validation Logic
# =====================================================================
@patch("mpu_decomposition.MPU.check_mpo_unitarity")
def test_uniformmpu_fails_unitarity(mock_unitarity, identity_mpu):
    # Simula il fallimento istantaneo
    mock_unitarity.side_effect = ValueError("Unitarity strictly lost")
    _, _, A_bulk, l_in, r_in = identity_mpu

    with pytest.raises(ValueError, match="Unitarity strictly lost"):
        UniformMPU(A=A_bulk, l_vec=l_in, r_vec=r_in, N=4)


@patch("mpu_decomposition.MPU.check_assumption_1")
@patch("mpu_decomposition.MPU.check_mpo_unitarity")
def test_uniformmpu_fails_injectivity(
    mock_unitarity, mock_injectivity, cz_interaction_mpu
):
    mock_unitarity.return_value = None
    # Simula il fallimento di iniettività
    mock_injectivity.side_effect = ValueError("Assumption 1 failed")

    _, _, A_bulk, l_in, r_in = cz_interaction_mpu

    with pytest.raises(ValueError, match="Assumption 1 failed"):
        UniformMPU(A=A_bulk, l_vec=l_in, r_vec=r_in, N=4)


@patch("mpu_decomposition.MPU.UniformMPU._compute_q_unif")
@patch("mpu_decomposition.MPU.UniformMPU._compute_boundary_operators")
@patch("mpu_decomposition.MPU.check_assumption_1")
@patch("mpu_decomposition.MPU.check_mpo_unitarity")
def test_uniformmpu_quimb_conversion(
    mock_unitarity,
    mock_injectivity,
    mock_compute_boundaries,
    mock_compute_q,
    identity_mpu,
):
    """
    Verifies that after passing validation, the raw numpy array is correctly
    encapsulated into a quimb.Tensor with the expected tags and indices.
    """
    # 1. Setup mocks to bypass internal validation and derived property logic
    mock_unitarity.return_value = None
    mock_injectivity.return_value = ([1.0], [1.0])

    # Isolate initialization from subsequent tensor contractions
    dummy_dim = 2
    mock_compute_boundaries.return_value = (np.eye(dummy_dim), np.eye(dummy_dim))
    mock_compute_q.return_value = 1.0

    # 2. Extract fixture data
    _, bond_dim, A_bulk, l_in, r_in = identity_mpu
    num_sites = 4

    # 3. Instantiate and Verify
    mpu = UniformMPU(A=A_bulk, l_vec=l_in, r_vec=r_in, N=num_sites)

    # Bulk Tensor assertions
    assert isinstance(mpu.A, qtn.Tensor)
    assert "A" in mpu.A.tags
    assert mpu.A.inds == ("p_out", "p_in", "bond_0", "bond_N")

    # Metadata assertions
    assert mpu._D == bond_dim
    assert mpu._N == num_sites

    # Boundary vector tagging and indexing assertions
    assert "l" in mpu.l_vec.tags
    assert mpu.l_vec.inds == ("bond_0",)
    assert "r" in mpu.r_vec.tags
    assert mpu.r_vec.inds == (f"bond_{num_sites}",)


@pytest.mark.parametrize(
    "mpu_fixture_name",
    [
        "identity_mpu",
        "local_unitary_mpu",
        "cz_interaction_mpu",
        "semisimple_v_mpu",  # Critical case D=5
    ],
)
def test_uniformmpu_initialization_success(mpu_fixture_name, request):
    """
    Verifies that all valid MPU fixtures pass internal checks (Unitarity/Injectivity).
    """
    # 1. Setup from Fixture
    d_phys, D_bond, A_bulk, l_in, r_in = request.getfixturevalue(mpu_fixture_name)
    num_sites = 4

    # 2. Initialization must complete without raising ValueError
    mpu = UniformMPU(A=A_bulk, l_vec=l_in, r_vec=r_in, N=num_sites)

    # 3. Structural and Physical Assertions
    assert mpu._D == D_bond
    assert mpu._d == d_phys
    assert mpu.q_unif >= 1.0 - 1e-12


@pytest.mark.parametrize("lcu_fixture", ALL_LCU_DATA)
def test_get_merging_operator_shape(lcu_fixture, request):
    """get_merging_operator() must return (D, D, D, D)."""
    D, _, _, _, mpu = request.getfixturevalue(lcu_fixture)
    M = mpu._get_merging_operator()
    assert M.shape == (D, D, D, D)


def test_uniformmpu_random_fails_as_expected_deterministic():
    """
    Deterministic test: Forces a non-unitary bulk tensor to verify fail-fast behavior.
    """
    # --- Setup: Define a known non-unitary bulk tensor ---
    D = 2  # Bond dimension
    d = 2  # Physical dimension

    # Create a valid bulk tensor (D x D x d x d) — but make it non-unitary
    A = np.zeros((D, D, d, d), dtype=complex)

    # Fill with identity-like structure, but perturb one entry to break isometry
    A[0, 0, 0, 0] = 1.0
    A[0, 0, 1, 1] = 1.0
    A[1, 1, 0, 0] = 1.0
    A[1, 1, 1, 1] = 1.0

    # Now **break unitarity**: modify one entry to make the isometry fail
    A[0, 0, 0, 0] += 1.0  # This breaks the isometry condition

    # Boundary vectors (valid, but irrelevant for this test)
    l_in = np.array([1.0, 0.0])
    r_in = np.array([1.0, 0.0])

    with pytest.raises(ValueError, match="Unitarity strictly lost"):
        UniformMPU(A=A, l_vec=l_in, r_vec=r_in, N=4)


# =====================================================================
# Group 3: UniformMPU Internal Contractions & Boundary Operators
# =====================================================================
@pytest.mark.parametrize(
    "mpu_fixture_name", ["identity_mpu", "cz_interaction_mpu", "semisimple_v_mpu"]
)
def test_compute_boundary_operators_real_contraction(mpu_fixture_name, request):
    """
    Verifies that TN contractions yield valid (D, D) boundary matrices for each model.
    """
    # 1. Extract fixture data
    _, bond_dim, bulk_data, l_in, r_in = request.getfixturevalue(mpu_fixture_name)
    num_sites = 4

    # 2. Initialize MPU and execute contraction
    mpu = UniformMPU(bulk_data, l_in, r_in, N=num_sites)
    L2_mat, R2_mat = mpu._compute_boundary_operators()

    # 3. Shape Assertions
    # Boundary operators must act on the bond space (D x D)
    assert L2_mat.shape == (bond_dim, bond_dim)
    assert R2_mat.shape == (bond_dim, bond_dim)

    # 4. Physicality Assertions
    # Ensure matrices are non-null and have non-zero norm
    assert np.linalg.norm(L2_mat) > 0
    assert np.linalg.norm(R2_mat) > 0


# =====================================================================
# Group 4: UniformMPU q_unif Calculation & Numerical Stability
# =====================================================================


def test_q_unif_dimension_mismatch(identity_mpu):
    """
    Verifies that a ValueError is raised if L2 and R2 have incompatible dimensions.
    """
    d_phys, _, bulk_tensor, l_in, r_in = identity_mpu

    # Bypass initial validations to focus on the dimension check
    with patch("mpu_decomposition.MPU.check_mpo_unitarity", return_value=True), patch(
        "mpu_decomposition.MPU.check_assumption_1", return_value=(None, None)
    ), patch(
        "mpu_decomposition.MPU.UniformMPU._compute_boundary_operators",
        return_value=(np.eye(2), np.eye(2)),
    ):
        mpu = UniformMPU(bulk_tensor, l_in, r_in, N=2)

    # Inject mismatched boundary operators
    mpu.L_inv = np.eye(2)
    mpu.R_inv = np.eye(3)

    with pytest.raises(ValueError, match="Dimension mismatch"):
        mpu._compute_q_unif()


def test_q_unif_identity_case(identity_mpu):
    """
    Verifies that for the identity channel, q_unif evaluates exactly to sqrt(d).
    """
    d_phys, bond_dim, bulk_tensor, l_in, r_in = identity_mpu

    # Isolated instantiation
    with patch("mpu_decomposition.MPU.check_mpo_unitarity", return_value=True), patch(
        "mpu_decomposition.MPU.check_assumption_1", return_value=(None, None)
    ), patch(
        "mpu_decomposition.MPU.UniformMPU._compute_boundary_operators",
        return_value=(np.eye(bond_dim) / bond_dim, np.eye(bond_dim) / bond_dim),
    ):
        mpu = UniformMPU(bulk_tensor, l_in, r_in, N=4)

    # Set canonical identity boundary operators (D=1)
    mpu.L = np.array([[1.0]])
    mpu.R = np.array([[1.0]])

    expected_q = 1.0
    assert mpu._compute_q_unif() == pytest.approx(expected_q)


# =====================================================================
# Group 5: LCU Decomposition
# =====================================================================


@pytest.mark.parametrize("lcu_fixture", ALL_LCU_DATA)
def test_lcu_C_positive(lcu_fixture, request):
    """The LCU 1-norm must be strictly positive."""
    D, coeffs, units, C, mpu = request.getfixturevalue(lcu_fixture)
    assert C > 0


@pytest.mark.parametrize("lcu_fixture", ALL_LCU_DATA)
def test_lcu_unitaries_are_unitary(lcu_fixture, request):
    """Each unitary in the LCU list must be unitary."""
    D, coeffs, units, C, mpu = request.getfixturevalue(lcu_fixture)

    for i, W in enumerate(units):
        err = np.linalg.norm(W @ W.conj().T - np.eye(W.shape[0]))
        assert err < 1e-10, f"Non-unitary W[{i}] detected: {err:.2e}"


@pytest.mark.parametrize("lcu_fixture", ALL_LCU_DATA)
def test_lcu_terms_are_unique(lcu_fixture, request):
    """Ensure no duplicate unitaries in LCU list."""
    D, coeffs, units, C, mpu = request.getfixturevalue(lcu_fixture)

    # Use a tolerance for floating-point comparison
    seen = set()
    for i, W in enumerate(units):
        key = tuple(W.ravel().round(10))  # Hashable representation
        assert key not in seen, f"Duplicate unitary found at index {i}"
        seen.add(key)


@pytest.mark.parametrize("lcu_fixture", ALL_LCU_DATA)
def test_lcu_reconstruction_from_terms(lcu_fixture, request):
    """Reconstruct merging operator from LCU terms and verify correctness."""
    D, _, _, _, mpu = request.getfixturevalue(lcu_fixture)
    coeffs, units, C = mpu._build_lcu_data()

    M_recon = np.zeros((D**2, D**2), dtype=complex)
    for c, W in zip(coeffs, units):
        M_recon += c * W

    M_ref = mpu._get_merging_operator().reshape(D**2, D**2)
    assert np.allclose(
        M_recon, M_ref, atol=1e-8
    ), f"Reconstruction error: {np.max(np.abs(M_recon - M_ref)):.2e}"


def test_uniformmpu_lcu_caching(cz_interaction_mpu):
    """Multiple calls to _build_lcu_data return the same cached result."""
    d, D, A, l_in, r_in = cz_interaction_mpu
    mpu = UniformMPU(A=A, l_vec=l_in, r_vec=r_in, N=4)

    res1 = mpu._build_lcu_data()
    res2 = mpu._build_lcu_data()

    assert res1 is res2

    # Compare coefficients and unitaries
    for c1, c2 in zip(res1[0], res2[0]):
        assert c1 == pytest.approx(c2, abs=1e-12)
    for W1, W2 in zip(res1[1], res2[1]):
        assert np.allclose(W1, W2, atol=1e-12)


# =====================================================================
# Group 5: Unitary Merging Components
# =====================================================================
def test_build_merging_unitary_B_prepares_correct_state(identity_lcu_data):
    """
    B|0> must equal (1/C) * sum_i sqrt(c_i) |i> exactly.
    """
    D, coeffs, units, C, mpu = identity_lcu_data
    K = len(coeffs)
    dim_ancilla = mpu.B.shape[0]

    state_0 = np.zeros(dim_ancilla, dtype=complex)
    state_0[0] = 1.0
    prepared = mpu.B @ state_0

    expected = np.zeros(dim_ancilla, dtype=complex)
    expected[:K] = np.sqrt(coeffs) / np.sqrt(C)

    assert np.allclose(prepared, expected, atol=1e-10)


def test_build_merging_unitary_B_is_unitary(cz_lcu_data):
    """B must be a unitary matrix."""
    D, coeffs, units, C, mpu = cz_lcu_data
    err = np.linalg.norm(mpu.B @ mpu.B.conj().T - np.eye(mpu.B.shape[0]))
    assert err < 1e-10


def test_build_merging_unitary_W_ctrl_is_unitary(cz_lcu_data):
    """W_ctrl must be a unitary matrix."""
    D, coeffs, units, C, mpu = cz_lcu_data
    err = np.linalg.norm(mpu.W_ctrl @ mpu.W_ctrl.conj().T - np.eye(mpu.W_ctrl.shape[0]))
    assert err < 1e-10


def test_build_merging_unitary_B_first_col_is_unit_vector(cz_lcu_data):
    """
    The first column of B must have unit norm for B to be a valid unitary.
    This catches the /C vs /sqrt(C) normalization bug.
    """
    D, coeffs, units, C, mpu = cz_lcu_data

    first_col = mpu.B[:, 0]
    norm = np.linalg.norm(first_col)

    assert norm == pytest.approx(1.0, abs=1e-10), (
        f"First column of B has norm {norm:.6f}, expected 1.0. "
        f"Check normalization: use sqrt(c_i)/sqrt(C), not sqrt(c_i)/C."
    )
