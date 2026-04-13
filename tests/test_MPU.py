import pytest
import numpy as np
import quimb.tensor as qtn
from unittest.mock import patch
from mpu_decomposition.MPU import CircuitDecomposition, UniformMPU


@pytest.fixture(scope="module")
def identity_lcu_data(identity_mpu):
    d, D, A, l_in, r_in = identity_mpu
    mpu = UniformMPU(A=A, l_vec=l_in, r_vec=r_in, N=4)
    coeffs, U_R, U_L, C, target = CircuitDecomposition._build_lcu_data(
        D, mpu.R_inv, mpu.L_inv
    )
    return D, coeffs, U_R, U_L, C, target, mpu


@pytest.fixture(scope="module")
def cz_lcu_data(cz_interaction_mpu):
    d, D, A, l_in, r_in = cz_interaction_mpu
    mpu = UniformMPU(A=A, l_vec=l_in, r_vec=r_in, N=4)
    coeffs, U_R, U_L, C, target = CircuitDecomposition._build_lcu_data(
        D, mpu.R_inv, mpu.L_inv
    )
    return D, coeffs, U_R, U_L, C, target, mpu


@pytest.fixture(scope="module")
def semisimple_lcu_data(semisimple_v_mpu):
    d, D, A, l_in, r_in = semisimple_v_mpu
    mpu = UniformMPU(A=A, l_vec=l_in, r_vec=r_in, N=4)
    coeffs, U_R, U_L, C, target = CircuitDecomposition._build_lcu_data(
        D, mpu.R_inv, mpu.L_inv
    )
    return D, coeffs, U_R, U_L, C, target, mpu


@pytest.fixture(scope="module")
def identity_lcu_unitary(identity_lcu_data):
    D, coeffs, U_R, U_L, C, target, mpu = identity_lcu_data
    np.random.seed(0)
    U = CircuitDecomposition._construct_lcu_unitary(target, U_R, U_L)
    return D, coeffs, U_R, U_L, C, target, mpu, U


@pytest.fixture(scope="module")
def cz_lcu_unitary(cz_lcu_data):
    D, coeffs, U_R, U_L, C, target, mpu = cz_lcu_data
    np.random.seed(0)
    U = CircuitDecomposition._construct_lcu_unitary(target, U_R, U_L)
    return D, coeffs, U_R, U_L, C, target, mpu, U


@pytest.fixture(scope="module")
def semisimple_lcu_unitary(semisimple_lcu_data):
    D, coeffs, U_R, U_L, C, target, mpu = semisimple_lcu_data
    np.random.seed(0)
    U = CircuitDecomposition._construct_lcu_unitary(target, U_R, U_L)
    return D, coeffs, U_R, U_L, C, target, mpu, U


ALL_LCU_DATA = ["identity_lcu_data", "cz_lcu_data", "semisimple_lcu_data"]
ALL_LCU_UNITARIES = ["identity_lcu_unitary", "cz_lcu_unitary", "semisimple_lcu_unitary"]

# =====================================================================
# Group 1: Abstract Base Class Constraints
# =====================================================================


def test_abstract_instantiation_fails():
    """
    Verifies that the base CircuitDecomposition class cannot be instantiated directly.
    """
    with pytest.raises(
        TypeError, match="Can't instantiate abstract class CircuitDecomposition"
    ):
        CircuitDecomposition(N=4, d=2, l_vec=qtn.Tensor(), r_vec=qtn.Tensor())


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


def test_uniformmpu_random_fails_as_expected(random_complex_mpo):
    """
    Verifies that a generic, non-unitary random MPO is blocked by the constructor.
    """
    # 1. Setup from Random Fixture
    _, _, A_bulk, l_in, r_in = random_complex_mpo
    num_sites = 4

    # 2. Assert fail-fast behavior on unitarity violation
    # Message must match the specific string in UniformMPU.__init__
    expected_error = "Instantiation aborted: The bulk tensor 'A' does not satisfy"

    with pytest.raises(ValueError, match=expected_error):
        UniformMPU(A=A_bulk, l_vec=l_in, r_vec=r_in, N=num_sites)


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

    # Il Beyond QCA deve avere un valore superiore a 1.0
    assert q_semisimple > 1.0 + 1e-12

    # The entangling power must be a physical real quantity
    assert np.isreal(q_semisimple) or np.abs(np.imag(q_semisimple)) < 1e-12


# =====================================================================
# Group 5: Merging Operator Creation
# =====================================================================


def test_merging_operator_kernel_value(identity_mpu):
    """
    For identity MPU (D=1), M[0,0,:,:] must equal R_inv.T @ L_inv exactly.
    """
    d, D, A, l_in, r_in = identity_mpu
    mpu = UniformMPU(A=A, l_vec=l_in, r_vec=r_in, N=4)

    M = mpu.get_merging_operator()
    expected_kernel = mpu.R_inv.T @ mpu.L_inv

    assert np.allclose(M[0, 0, :, :], expected_kernel, atol=1e-12)


def test_merging_operator_only_00_nonzero(cz_interaction_mpu):
    """
    By definition, only the M[0,0,:,:] block carries the kernel.
    All other slices must be exactly zero.
    """
    d, D, A, l_in, r_in = cz_interaction_mpu
    mpu = UniformMPU(A=A, l_vec=l_in, r_vec=r_in, N=4)

    M = mpu.get_merging_operator()

    for i in range(D):
        for j in range(D):
            if i == 0 and j == 0:
                continue
            assert np.allclose(
                M[i, j, :, :], 0.0, atol=1e-15
            ), f"M[{i},{j},:,:] is nonzero"


def test_merging_operator_nonsquare_raises():
    """
    Passing rectangular matrices must raise ValueError.
    """
    L_inv = np.ones((2, 3))
    R_inv = np.ones((2, 3))

    with pytest.raises(ValueError, match="square matrices"):
        CircuitDecomposition.get_merging_operator(L_inv, R_inv)


def test_merging_operator_dimension_mismatch_raises():
    """
    Passing square matrices of different sizes must raise ValueError.
    """
    L_inv = np.eye(2)
    R_inv = np.eye(3)

    with pytest.raises(ValueError, match="same dimension"):
        CircuitDecomposition.get_merging_operator(L_inv, R_inv)


@pytest.mark.parametrize(
    "mpu_fixture",
    ["identity_mpu", "cz_interaction_mpu", "semisimple_v_mpu"],
)
def test_merging_operator_complex_dtype(mpu_fixture, request):
    """
    Output must always be complex regardless of input dtype.
    """
    d, D, A, l_in, r_in = request.getfixturevalue(mpu_fixture)
    mpu = UniformMPU(A=A, l_vec=l_in, r_vec=r_in, N=4)

    M = mpu.get_merging_operator()
    assert np.iscomplexobj(M)


# =====================================================================
# Group 6: LCU Decomposition
# =====================================================================


@pytest.mark.parametrize("lcu_fixture", ALL_LCU_DATA)
def test_lcu_coefficients_nonnegative(lcu_fixture, request):
    """
    All LCU coefficients must be non-negative reals.
    """
    D, coeffs, U_R, U_L, C, target, mpu = request.getfixturevalue(lcu_fixture)

    assert np.all(coeffs >= -1e-15), f"Negative coefficient found: {coeffs.min()}"
    assert np.all(np.isreal(coeffs))


@pytest.mark.parametrize("lcu_fixture", ALL_LCU_DATA)
def test_lcu_normalization_constant(lcu_fixture, request):
    """
    C must equal the sum of all coefficients.
    """
    D, coeffs, U_R, U_L, C, target, mpu = request.getfixturevalue(lcu_fixture)

    assert C == pytest.approx(np.sum(coeffs), abs=1e-12)


@pytest.mark.parametrize("lcu_fixture", ALL_LCU_DATA)
def test_lcu_target_normalized(lcu_fixture, request):
    """
    The state preparation vector must have unit norm.
    """
    D, coeffs, U_R, U_L, C, target, mpu = request.getfixturevalue(lcu_fixture)

    assert np.linalg.norm(target) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("lcu_fixture", ALL_LCU_DATA)
def test_lcu_padded_dimension_power_of_two(lcu_fixture, request):
    """
    The ancilla dimension d_anc must be a power of two.
    """
    D, coeffs, U_R, U_L, C, target, mpu = request.getfixturevalue(lcu_fixture)

    d_anc = len(coeffs)
    assert d_anc > 0
    assert (d_anc & (d_anc - 1)) == 0, f"d_anc={d_anc} is not a power of 2"


@pytest.mark.parametrize("lcu_fixture", ALL_LCU_DATA)
def test_lcu_nonpadded_count(lcu_fixture, request):
    """
    Exactly D^3 terms must carry nonzero coefficients (before padding).
    """
    D, coeffs, U_R, U_L, C, target, mpu = request.getfixturevalue(lcu_fixture)

    N_real = D**3
    assert np.sum(np.abs(coeffs[:N_real]) > 1e-15) == N_real


def test_lcu_d1_trivial(identity_lcu_data):
    """
    D=1: single non-padded term, scalar coefficient, 1x1 unitaries.
    """
    D, coeffs, U_R, U_L, C, target, mpu = identity_lcu_data
    assert D == 1
    assert np.abs(coeffs[0]) > 1e-15
    assert U_R[0].shape == (1, 1)
    assert U_L[0].shape == (1, 1)
    assert len(coeffs) >= 2
    assert coeffs[1] == pytest.approx(0.0, abs=1e-15)


# =====================================================================
# Group 7: UniformMPU LCU Integration & Caching
# =====================================================================


def test_uniformmpu_build_lcu_success(cz_interaction_mpu):
    """
    Verifies that _build_lcu_data correctly generates the tuple of data
    """
    d, D, A, l_in, r_in = cz_interaction_mpu
    mpu = UniformMPU(A=A, l_vec=l_in, r_vec=r_in, N=4)

    # La chiamata ora deve avere successo e restituire i dati
    res = mpu._build_lcu_data()
    assert res is not None

    coeffs, U_R, U_L, C, target = res
    assert len(coeffs) >= D**3
    assert len(U_R) == len(coeffs)
    assert len(U_L) == len(coeffs)


def test_uniformmpu_lcu_caching(cz_interaction_mpu):
    """
    Verifies that multiple calls to _build_lcu_data do not recalculate
    the SVD decomposition, but instead return the cached data.
    """
    d, D, A, l_in, r_in = cz_interaction_mpu
    mpu = UniformMPU(A=A, l_vec=l_in, r_vec=r_in, N=4)

    # First call should instantiate the cache
    res1 = mpu._build_lcu_data()
    # Second one should simply return the cache
    res2 = mpu._build_lcu_data()

    # 'is' verifies that they are the same object in memory
    assert res1 is res2


# =====================================================================
# Group 8: LCU Block Encoding and Norms
# =====================================================================


@pytest.mark.parametrize("lcu_unitary_fixture", ALL_LCU_UNITARIES)
def test_lcu_norms(lcu_unitary_fixture, request):
    r"""
    Explicitly tests Equation (12) from the paper:
    U |0>_A |ψ>_S = (1/C)|0>_A M|ψ>_S + sqrt{1 - 1/C^2}|Φ^⊥>
    """
    D, coeffs, U_R, U_L, C, target, mpu, U = request.getfixturevalue(
        lcu_unitary_fixture
    )
    d_sys = U_R[0].shape[0] * U_L[0].shape[0]
    d_anc = len(target)

    M_ref = mpu.get_merging_operator().reshape(d_sys, d_sys)

    # Random normalized system state
    np.random.seed(42)
    psi_S = np.random.randn(d_sys) + 1j * np.random.randn(d_sys)
    psi_S /= np.linalg.norm(psi_S)

    e0_A = np.zeros(d_anc)
    e0_A[0] = 1.0
    psi_in = np.kron(e0_A, psi_S)

    psi_out = U @ psi_in
    psi_out_reshaped = psi_out.reshape(d_anc, d_sys)

    # Success branch: ancilla postselected on |0>
    success_vector = psi_out_reshaped[0, :]
    expected_success = (M_ref @ psi_S) / C

    assert np.allclose(
        success_vector, expected_success, atol=1e-10
    ), f"[{lcu_unitary_fixture}] Success vector mismatch"

    # Failure branch: probability conservation
    norm_success = np.linalg.norm(success_vector)
    failure_matrix = np.copy(psi_out_reshaped)
    failure_matrix[0, :] = 0
    norm_failure = np.linalg.norm(failure_matrix)
    expected_norm_failure = np.sqrt(np.maximum(0.0, 1.0 - norm_success**2))

    assert np.isclose(
        norm_failure, expected_norm_failure, atol=1e-10
    ), f"[{lcu_unitary_fixture}] Failure norm mismatch"


@pytest.mark.parametrize("lcu_unitary_fixture", ALL_LCU_UNITARIES)
def test_lcu_unitary_block_encodes_merging_operator(lcu_unitary_fixture, request):
    """<0|_anc U |0>_anc must equal (1/C) * sum_k c_k (U_R_k ⊗ U_L_k)."""
    D, coeffs, U_R, U_L, C, target, mpu, U = request.getfixturevalue(
        lcu_unitary_fixture
    )
    d_sys = U_R[0].shape[0] * U_L[0].shape[0]

    U_block = U[:d_sys, :d_sys]

    M_lcu = np.zeros((d_sys, d_sys), dtype=complex)
    for k in range(len(coeffs)):
        if np.abs(coeffs[k]) < 1e-18:
            continue
        M_lcu += coeffs[k] * np.kron(U_R[k], U_L[k])

    assert np.allclose(
        U_block, M_lcu / C, atol=1e-8
    ), f"Block encoding error: {np.max(np.abs(U_block - M_lcu / C)):.2e}"


# =====================================================================
# Group 9: Unitary U of LCU
# =====================================================================


@pytest.mark.parametrize("lcu_fixture", ALL_LCU_DATA)
def test_lcu_unitaries_are_unitary(lcu_fixture, request):
    """Every U_R_k and U_L_k must satisfy U @ U† = I."""
    D, coeffs, U_R, U_L, C, target, mpu = request.getfixturevalue(lcu_fixture)
    I_D = np.eye(D, dtype=complex)
    for k, (Ur, Ul) in enumerate(zip(U_R, U_L)):
        assert np.allclose(Ur @ Ur.conj().T, I_D, atol=1e-10), f"U_R[{k}] not unitary"
        assert np.allclose(Ul @ Ul.conj().T, I_D, atol=1e-10), f"U_L[{k}] not unitary"


@pytest.mark.parametrize("lcu_fixture", ALL_LCU_DATA)
def test_lcu_padded_entries(lcu_fixture, request):
    """Padded entries beyond D^3 must have zero coefficients and identity unitaries."""
    D, coeffs, U_R, U_L, C, target, mpu = request.getfixturevalue(lcu_fixture)
    N_real = D**3
    I_D = np.eye(D, dtype=complex)
    d_anc = len(coeffs)
    assert (d_anc & (d_anc - 1)) == 0, f"d_anc={d_anc} not power of 2"
    for k in range(N_real, len(coeffs)):
        assert coeffs[k] == pytest.approx(0.0, abs=1e-15)
        assert np.allclose(U_R[k], I_D, atol=1e-15)
        assert np.allclose(U_L[k], I_D, atol=1e-15)


@pytest.mark.parametrize("lcu_fixture", ALL_LCU_DATA)
def test_lcu_reconstruction(lcu_fixture, request):
    """sum_k c_k * kron(U_R_k, U_L_k) must reconstruct the merging operator."""
    D, coeffs, U_R, U_L, C, target, mpu = request.getfixturevalue(lcu_fixture)

    M_reconstructed = np.zeros((D * D, D * D), dtype=complex)
    for k in range(len(coeffs)):
        if np.abs(coeffs[k]) < 1e-18:
            continue
        M_reconstructed += coeffs[k] * np.kron(U_R[k], U_L[k])

    M_ref = mpu.get_merging_operator().reshape(D * D, D * D)
    assert np.allclose(
        M_reconstructed, M_ref, atol=1e-8
    ), f"LCU reconstruction error: {np.max(np.abs(M_reconstructed - M_ref)):.2e}"


# =====================================================================
# Group 10: LCU Unitary Construction
# =====================================================================


@pytest.mark.parametrize("lcu_unitary_fixture", ALL_LCU_UNITARIES)
def test_lcu_unitary_is_unitary(lcu_unitary_fixture, request):
    """U must satisfy U @ U† = U† @ U = I."""
    D, coeffs, U_R, U_L, C, target, mpu, U = request.getfixturevalue(
        lcu_unitary_fixture
    )
    dim = U.shape[0]
    I_full = np.eye(dim, dtype=complex)
    assert np.allclose(U @ U.conj().T, I_full, atol=1e-10), "U @ U† != I"
    assert np.allclose(U.conj().T @ U, I_full, atol=1e-10), "U† @ U != I"


def test_lcu_unitary_deterministic(cz_lcu_data):
    """With the same seed, two calls must produce identical U."""
    D, coeffs, U_R, U_L, C, target, mpu = cz_lcu_data
    np.random.seed(42)
    U1 = CircuitDecomposition._construct_lcu_unitary(target, U_R, U_L)
    np.random.seed(42)
    U2 = CircuitDecomposition._construct_lcu_unitary(target, U_R, U_L)
    assert np.allclose(U1, U2, atol=1e-14)
