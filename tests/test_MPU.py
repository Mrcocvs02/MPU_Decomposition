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
    """
    Asserts fail-fast behavior if the global isometry condition is not met.
    """
    mock_unitarity.return_value = False
    _, _, A_bulk, l_in, r_in = identity_mpu

    # Check for specific error message regarding isometry
    with pytest.raises(
        ValueError, match="Instantiation aborted: The bulk tensor 'A' does not satisfy"
    ):
        UniformMPU(A=A_bulk, l_vec=l_in, r_vec=r_in, N=4)


@patch("mpu_decomposition.MPU.check_assumption_1")
@patch("mpu_decomposition.MPU.check_mpo_unitarity")
def test_uniformmpu_fails_injectivity(
    mock_unitarity, mock_injectivity, cz_interaction_mpu
):
    """
    Asserts fail-fast behavior if the injectivity/bond dimension check (Assumption 1) fails.
    """
    mock_unitarity.return_value = True

    # Simulate a failure in boundary injectivity
    s_left_mock = [1.0, 0.1]
    s_right_mock = [1.0, 0.0]
    mock_injectivity.return_value = (False, s_left_mock, s_right_mock)

    _, _, A_bulk, l_in, r_in = cz_interaction_mpu

    # Check for specific error message regarding injectivity
    with pytest.raises(
        ValueError, match="Instantiation aborted: Boundary injectivity failed"
    ):
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
    mock_unitarity.return_value = True
    mock_injectivity.return_value = (True, [1.0], [1.0])

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
    M = mpu.get_merging_operator()
    assert M.shape == (D, D, D, D)


def test_uniformmpu_random_fails_as_expected_deterministic():
    """
    Deterministic test: Forces a non-unitary bulk tensor to verify fail-fast behavior.
    """
    # --- Setup: Define a known non-unitary bulk tensor ---
    D = 2  # Bond dimension
    d = 2  # Physical dimension
    N = 4  # Number of sites

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

    # --- Test: Constructor must fail due to non-unitarity ---
    expected_error = "Instantiation aborted: The bulk tensor 'A' does not satisfy"

    with pytest.raises(ValueError, match=expected_error):
        UniformMPU(A=A, l_vec=l_in, r_vec=r_in, N=N)


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
        "mpu_decomposition.MPU.check_assumption_1", return_value=(True, None, None)
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


def test_q_unif_unphysical_negative_trace(identity_mpu):
    """
    Verifies that a negative real trace in boundary operators raises a ValueError.
    """
    _, bond_dim, bulk_tensor, l_in, r_in = identity_mpu

    # Mock internal calls to reach the q_unif calculation block
    with patch("mpu_decomposition.MPU.check_mpo_unitarity", return_value=True), patch(
        "mpu_decomposition.MPU.check_assumption_1", return_value=(True, None, None)
    ), patch(
        "mpu_decomposition.MPU.UniformMPU._compute_boundary_operators",
        return_value=(np.eye(bond_dim), np.eye(bond_dim)),
    ):
        mpu = UniformMPU(bulk_tensor, l_in, r_in, N=2)

    # Force a negative trace scenario (D=1 for Identity)
    mpu.L = np.array([[1.0]])
    mpu.R = np.array([[-1.0]])

    with pytest.raises(ValueError, match="Unphysical negative trace"):
        mpu._compute_q_unif()


def test_q_unif_identity_case(identity_mpu):
    """
    Verifies that for the identity channel, q_unif evaluates exactly to sqrt(d).
    """
    d_phys, bond_dim, bulk_tensor, l_in, r_in = identity_mpu

    # Isolated instantiation
    with patch("mpu_decomposition.MPU.check_mpo_unitarity", return_value=True), patch(
        "mpu_decomposition.MPU.check_assumption_1", return_value=(True, None, None)
    ), patch(
        "mpu_decomposition.MPU.UniformMPU._compute_boundary_operators",
        return_value=(np.eye(bond_dim), np.eye(bond_dim)),
    ):
        mpu = UniformMPU(bulk_tensor, l_in, r_in, N=4)

    # Set canonical identity boundary operators (D=1)
    mpu.L = np.array([[1.0]])
    mpu.R = np.array([[1.0]])

    expected_q = 1.0
    assert mpu._compute_q_unif() == pytest.approx(expected_q)


def test_q_unif_comparison_beyond_qca(identity_mpu, semisimple_v_mpu):
    """
    Verifies that the 'Beyond QCA' model (semisimple_v) is less well-conditioned
    than the identity (perfect QCA), resulting in a higher or equal entangling power.
    """
    # 1. Identity Case (Perfect QCA): q_unif is the lower bound, equal to 1.0 for D=1
    d_id, _, A_id, l_id, r_id = identity_mpu
    mpu_identity = UniformMPU(A_id, l_id, r_id, N=4)
    q_identity = mpu_identity.q_unif

    # 2. Semisimple V Case (Beyond QCA): q_unif must be > 1.0
    # as the boundary maps are no longer perfect unitaries (higher cost C)
    d_v, _, A_v, l_v, r_v = semisimple_v_mpu
    mpu_semisimple = UniformMPU(A_v, l_v, r_v, N=4)
    q_semisimple = mpu_semisimple.q_unif

    # 3. Assertions
    # The identity must reach the theoretical minimum for a rank-1 boundary
    assert q_identity == pytest.approx(d_id)

    assert q_semisimple > 1.0 + 1e-12

    # The entangling power must be a physical real quantity
    assert np.isreal(q_semisimple) or np.abs(np.imag(q_semisimple)) < 1e-12


# =====================================================================
# Group 5: Factored Pauli LCU Decomposition
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
    _, _, _, _, mpu = request.getfixturevalue(lcu_fixture)
    coeffs, units, C = mpu._build_lcu_data()

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

    M_ref = mpu.get_merging_operator().reshape(D**2, D**2)
    assert np.allclose(
        M_recon, M_ref, atol=1e-8
    ), f"Reconstruction error: {np.max(np.abs(M_recon - M_ref)):.2e}"


def test_uniformmpu_lcu_caching(cz_interaction_mpu):
    """Multiple calls to _build_lcu_data return the same cached result."""
    d, D, A, l_in, r_in = cz_interaction_mpu
    mpu = UniformMPU(A=A, l_vec=l_in, r_vec=r_in, N=4)

    res1 = mpu._build_lcu_data()
    res2 = mpu._build_lcu_data()

    # Compare lists (deep equality)
    assert len(res1[0]) == len(res2[0])
    assert len(res1[1]) == len(res2[1])
    assert res1[2] == res2[2]

    # Compare coefficients and unitaries
    for c1, c2 in zip(res1[0], res2[0]):
        assert c1 == pytest.approx(c2, abs=1e-12)
    for W1, W2 in zip(res1[1], res2[1]):
        assert np.allclose(W1, W2, atol=1e-12)
