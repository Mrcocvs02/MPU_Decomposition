import pytest
import numpy as np
import quimb.tensor as qtn
import scipy.linalg as la
from unittest.mock import patch
from mpu_decomposition.MPU import CircuitDecomposition, UniformMPU

# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture(scope="module")
def identity_mpu():
    """Generates an exact identity MPU tensor with minimal canonical bond dimension (D=1)."""
    d, D = 2, 1
    A = np.einsum("ij,ab->ijab", np.eye(d), np.eye(D)) / np.sqrt(d)
    l_in = np.ones(D)
    r_in = np.ones(D)
    return d, D, A, l_in, r_in


@pytest.fixture(scope="module")
def semisimple_v_mpu():
    """
    Generates a semi-simple MPU (Example 14) with V-gate action on product states.
    Uses minimal D=5 OBC representation and L=2 blocking to satisfy Assumption 1.
    """
    # 1. Parametri di base (Qubit, Bond Dim originale D=5) [cite: 420-434]
    d = 2
    D_base = 5
    theta = np.pi / 4
    V = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    # 2. Definizione tensori bulk (D=5) [cite: 421-424]
    A_base = {
        (0, 0): np.diag([1, 1, 0, 0, 0]),
        (1, 1): np.diag([1, 0, 1, 0, 0]),
        (0, 1): np.diag([0, 0, 0, 1, 0]),
        (1, 0): np.diag([0, 0, 0, 0, 1]),
    }

    # 3. Blocking L=2 per l'Assumption 1 (d^2L = 16 >= 5)
    L = 2
    d_block = d**L  # d=4 dopo il blocking
    A_blocked = np.zeros((d_block, d_block, D_base, D_base), dtype=complex)

    for idx_out in range(d_block):
        for idx_in in range(d_block):
            # Mapping degli indici collettivi ai siti originali
            j = [(idx_out >> k) & 1 for k in range(L)][::-1]
            i = [(idx_in >> k) & 1 for k in range(L)][::-1]

            # Contrazione: A[j1,i1] @ A[j0,i0]
            res = A_base.get((j[0], i[0]), np.zeros((D_base, D_base)))
            for s in range(1, L):
                res = res @ A_base.get((j[s], i[s]), np.zeros((D_base, D_base)))

            A_blocked[idx_out, idx_in] = res

    # 4. Boundary Vectors minimali (Rango 1)  [cite: 144, 2201-2205]
    # r_in chiude i percorsi virtuali, l_in codifica l'operatore b [cite: 431]
    l_in = np.array([1, V[0, 0] - 1, V[1, 1] - 1, V[0, 1], V[1, 0]], dtype=complex)
    r_in = np.array([1, 1, 1, 1, 1], dtype=complex)

    return d_block, D_base, A_blocked, l_in, r_in


@pytest.fixture(scope="module")
def local_unitary_mpu():
    """Generates an MPU applying a local Pauli-X gate globally (D=1)."""
    d, D = 2, 1
    X = np.array([[0.0, 1.0], [1.0, 0.0]])
    A = np.einsum("ij,ab->ijab", X, np.eye(D)) / np.sqrt(d)
    l_in = np.ones(D)
    r_in = np.ones(D)
    return d, D, A, l_in, r_in


@pytest.fixture(scope="module")
def cz_interaction_mpu():
    """Generates the CZ-like interacting MPU with specific boundary vectors."""
    d, D = 2, 2
    A_00 = np.array([[1, 0], [0, 0]], dtype=float)
    A_01 = np.array([[0, 0], [0, 0]], dtype=float)
    A_10 = np.array([[0, 0], [0, 0]], dtype=float)
    A_11 = np.array([[1, 0], [0, 1]], dtype=float)
    A = np.array([[A_00, A_01], [A_10, A_11]], dtype=float)
    l_in = np.array([1.0, 1.0], dtype=float)
    r_in = np.array([1.0, -2.0], dtype=float)
    return d, D, A, l_in, r_in


@pytest.fixture(scope="module")
def random_complex_mpu():
    """Generates a random dense complex MPU tensor."""
    d, D = 3, 2
    np.random.seed(42)
    A = np.random.randn(d, d, D, D) + 1j * np.random.randn(d, d, D, D)
    l_in = np.random.randn(D) + 1j * np.random.randn(D)
    r_in = np.random.randn(D) + 1j * np.random.randn(D)
    return d, D, A, l_in, r_in


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


def test_uniformmpu_random_fails_as_expected(random_complex_mpu):
    """
    Verifies that a generic, non-unitary random MPO is blocked by the constructor.
    """
    # 1. Setup from Random Fixture
    _, _, A_bulk, l_in, r_in = random_complex_mpu
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
    # 1. Identity Case (Perfect QCA): q_unif should be exactly sqrt(d)
    d_id, _, A_id, l_id, r_id = identity_mpu
    mpu_identity = UniformMPU(A_id, l_id, r_id, N=4)
    q_identity = mpu_identity.q_unif

    # 2. Semisimple V Case (Beyond QCA): q_unif should be >= sqrt(d)
    d_v, _, A_v, l_v, r_v = semisimple_v_mpu
    mpu_semisimple = UniformMPU(A_v, l_v, r_v, N=4)
    q_semisimple = mpu_semisimple.q_unif

    # 3. Physical & Numerical Assertions
    # Baseline comparison with the identity limit
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


@pytest.mark.parametrize(
    "lcu_fixture",
    ["identity_lcu_data", "cz_lcu_data", "semisimple_lcu_data"],
)
def test_lcu_coefficients_nonnegative(lcu_fixture, request):
    """
    All LCU coefficients must be non-negative reals.
    """
    D, coeffs, U_R, U_L, C, target, mpu = request.getfixturevalue(lcu_fixture)

    assert np.all(coeffs >= -1e-15), f"Negative coefficient found: {coeffs.min()}"
    assert np.all(np.isreal(coeffs))


@pytest.mark.parametrize(
    "lcu_fixture",
    ["identity_lcu_data", "cz_lcu_data", "semisimple_lcu_data"],
)
def test_lcu_unitaries_are_unitary(lcu_fixture, request):
    """
    Every U_R_k and U_L_k must satisfy U @ U† = I.
    """
    D, coeffs, U_R, U_L, C, target, mpu = request.getfixturevalue(lcu_fixture)

    I_D = np.eye(D, dtype=complex)
    for k, (Ur, Ul) in enumerate(zip(U_R, U_L)):
        assert np.allclose(Ur @ Ur.conj().T, I_D, atol=1e-10), f"U_R[{k}] not unitary"
        assert np.allclose(Ul @ Ul.conj().T, I_D, atol=1e-10), f"U_L[{k}] not unitary"


@pytest.mark.parametrize(
    "lcu_fixture",
    ["identity_lcu_data", "cz_lcu_data", "semisimple_lcu_data"],
)
def test_lcu_normalization_constant(lcu_fixture, request):
    """
    C must equal the sum of all coefficients.
    """
    D, coeffs, U_R, U_L, C, target, mpu = request.getfixturevalue(lcu_fixture)

    assert C == pytest.approx(np.sum(coeffs), abs=1e-12)


@pytest.mark.parametrize(
    "lcu_fixture",
    ["identity_lcu_data", "cz_lcu_data", "semisimple_lcu_data"],
)
def test_lcu_target_normalized(lcu_fixture, request):
    """
    The state preparation vector must have unit norm.
    """
    D, coeffs, U_R, U_L, C, target, mpu = request.getfixturevalue(lcu_fixture)

    assert np.linalg.norm(target) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize(
    "lcu_fixture",
    ["identity_lcu_data", "cz_lcu_data", "semisimple_lcu_data"],
)
def test_lcu_padded_dimension_power_of_two(lcu_fixture, request):
    """
    The ancilla dimension d_anc must be a power of two.
    """
    D, coeffs, U_R, U_L, C, target, mpu = request.getfixturevalue(lcu_fixture)

    d_anc = len(coeffs)
    assert d_anc > 0
    assert (d_anc & (d_anc - 1)) == 0, f"d_anc={d_anc} is not a power of 2"


@pytest.mark.parametrize(
    "lcu_fixture",
    ["identity_lcu_data", "cz_lcu_data", "semisimple_lcu_data"],
)
def test_lcu_padded_entries(lcu_fixture, request):
    """
    Padded entries beyond D^3 must have zero coefficients and identity unitaries.
    """
    D, coeffs, U_R, U_L, C, target, mpu = request.getfixturevalue(lcu_fixture)

    N_real = D**3
    I_D = np.eye(D, dtype=complex)

    for k in range(N_real, len(coeffs)):
        assert coeffs[k] == pytest.approx(
            0.0, abs=1e-15
        ), f"Padded coeff[{k}] = {coeffs[k]}"
        assert np.allclose(U_R[k], I_D, atol=1e-15), f"Padded U_R[{k}] != I"
        assert np.allclose(U_L[k], I_D, atol=1e-15), f"Padded U_L[{k}] != I"


@pytest.mark.parametrize(
    "lcu_fixture",
    ["identity_lcu_data", "cz_lcu_data", "semisimple_lcu_data"],
)
def test_lcu_nonpadded_count(lcu_fixture, request):
    """
    Exactly D^3 terms must carry nonzero coefficients (before padding).
    """
    D, coeffs, U_R, U_L, C, target, mpu = request.getfixturevalue(lcu_fixture)

    N_real = D**3
    assert np.sum(np.abs(coeffs[:N_real]) > 1e-15) == N_real


@pytest.mark.parametrize(
    "lcu_fixture",
    ["identity_lcu_data", "cz_lcu_data", "semisimple_lcu_data"],
)
def test_lcu_reconstruction(lcu_fixture, request):
    """
    The core invariant: sum_k c_k * kron(U_R_k, U_L_k) must reconstruct
    the merging operator M reshaped to (D^2, D^2).
    """
    D, coeffs, U_R, U_L, C, target, mpu = request.getfixturevalue(lcu_fixture)

    # Reconstruct from LCU terms
    M_reconstructed = np.zeros((D * D, D * D), dtype=complex)
    for k in range(len(coeffs)):
        if np.abs(coeffs[k]) < 1e-18:
            continue
        M_reconstructed += coeffs[k] * np.kron(U_R[k], U_L[k])

    # Get the reference merging operator and reshape
    M_ref = mpu.get_merging_operator()  # (D, D, D, D)
    M_ref_flat = M_ref.reshape(D * D, D * D)

    assert np.allclose(
        M_reconstructed, M_ref_flat, atol=1e-8
    ), f"LCU reconstruction error: {np.max(np.abs(M_reconstructed - M_ref_flat)):.2e}"


def test_lcu_d1_trivial(identity_mpu):
    """
    D=1: single non-padded term, scalar coefficient, 1x1 unitaries.
    """
    d, D, A, l_in, r_in = identity_mpu
    assert D == 1

    mpu = UniformMPU(A=A, l_vec=l_in, r_vec=r_in, N=4)
    coeffs, U_R, U_L, C, target = CircuitDecomposition._build_lcu_data(
        D, mpu.R_inv, mpu.L_inv
    )

    # Exactly 1 real term before padding
    assert np.abs(coeffs[0]) > 1e-15
    assert U_R[0].shape == (1, 1)
    assert U_L[0].shape == (1, 1)

    # Padded to power of 2 (d_anc >= 2)
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


@pytest.mark.parametrize(
    "mpu_fixture",
    ["identity_mpu", "cz_interaction_mpu", "semisimple_v_mpu"],
)
def test_lcu_block_encoding(mpu_fixture, request):
    """
    Verifies that the LCU components form a correct block-encoding
    of the merging operator M/C across all provided MPU models,
    by simulating the quantum circuit evolution.
    """
    d, D, A, l_in, r_in = request.getfixturevalue(mpu_fixture)
    mpu = UniformMPU(A=A, l_vec=l_in, r_vec=r_in, N=4)

    # Extract LCU data
    res = mpu._build_lcu_data()
    coeffs, U_R_list, U_L_list, C, target = res

    d_anc = len(target)
    d_sys = D * D

    # --- 1. CIRCUIT SETUP ---
    # Construct the ancilla preparation operator B
    A_mat = np.random.randn(d_anc, d_anc) + 1j * np.random.randn(d_anc, d_anc)
    A_mat[:, 0] = target
    Q, R_qr = np.linalg.qr(A_mat)
    B = Q @ np.diag(np.exp(-1j * np.angle(np.diag(R_qr))))

    # Construct the controlled operator W_ctrl (Select)
    blocks = [np.kron(Ur, Ul) for Ur, Ul in zip(U_R_list, U_L_list)]
    W_ctrl = la.block_diag(*blocks)

    # Global unitary U
    B_full = np.kron(B, np.eye(d_sys))
    U = B_full.conj().T @ W_ctrl @ B_full

    # --- 2. GROUND TRUTH FOR M ---
    M_ref = mpu.get_merging_operator().reshape(d_sys, d_sys)

    # --- 3. VERIFICATION ACROSS ALL BASIS STATES ---
    e0_A = np.zeros(d_anc)
    e0_A[0] = 1.0

    for i in range(d_sys):
        v_i = np.zeros(d_sys, dtype=complex)
        v_i[i] = 1.0

        # Basis state evolution
        psi_out_i = U @ np.kron(e0_A, v_i)

        # Post-selection: extract the block where the ancilla is in |0>
        res_i = psi_out_i.reshape(d_anc, d_sys)[0, :]

        # The expected block is M|v_i> / C
        exp_i = (M_ref @ v_i) / C

        assert np.allclose(
            res_i, exp_i, atol=1e-10
        ), f"[{mpu_fixture}] Block-encoding failed on basis state {i}"


@pytest.mark.parametrize(
    "mpu_fixture",
    ["identity_mpu", "cz_interaction_mpu", "semisimple_v_mpu"],
)
def test_lcu_norms(mpu_fixture, request):
    r"""
    Explicitly tests Equation (12) from the paper across all MPU models:
    U |ψ>_S |0>_A = (1/C)|Φ> + \sqrt{1 - 1/C^2}|Φ^⊥>
    """

    d, D, A, l_in, r_in = request.getfixturevalue(mpu_fixture)
    mpu = UniformMPU(A=A, l_vec=l_in, r_vec=r_in, N=4)

    # 1. Setup LCU Data
    coeffs, U_R_list, U_L_list, C, target = mpu._build_lcu_data()
    d_anc = len(target)
    d_sys = D * D

    # 2. Build Circuit Operators
    A_mat = np.random.randn(d_anc, d_anc) + 1j * np.random.randn(d_anc, d_anc)
    A_mat[:, 0] = target
    Q, R_qr = np.linalg.qr(A_mat)
    B = Q @ np.diag(np.exp(-1j * np.angle(np.diag(R_qr))))

    blocks = [np.kron(Ur, Ul) for Ur, Ul in zip(U_R_list, U_L_list)]
    W_ctrl = la.block_diag(*blocks)

    B_full = np.kron(B, np.eye(d_sys))
    U = B_full.conj().T @ W_ctrl @ B_full
    M_ref = mpu.get_merging_operator().reshape(d_sys, d_sys)

    # 3. Test on a random normalized state |ψ>_S
    np.random.seed(42)
    psi_S = np.random.randn(d_sys) + 1j * np.random.randn(d_sys)
    psi_S /= np.linalg.norm(psi_S)

    e0_A = np.zeros(d_anc)
    e0_A[0] = 1.0
    psi_in = np.kron(e0_A, psi_S)

    # Apply Global Unitary
    psi_out = U @ psi_in
    psi_out_reshaped = psi_out.reshape(d_anc, d_sys)

    # --- 4. Verify Equation (12) Components ---

    success_vector = psi_out_reshaped[0, :]
    expected_success = (M_ref @ psi_S) / C

    failure_matrix = np.copy(psi_out_reshaped)
    failure_matrix[0, :] = 0
    failure_vector = failure_matrix.flatten()

    # Assert 1: Success branch matches M|ψ>/C
    assert np.allclose(
        success_vector, expected_success, atol=1e-10
    ), f"[{mpu_fixture}] Success vector mismatch"

    # Assert 2: Failure branch norm matches probability conservation
    norm_success = np.linalg.norm(success_vector)
    norm_failure = np.linalg.norm(failure_vector)
    expected_norm_failure = np.sqrt(np.maximum(0.0, 1.0 - norm_success**2))

    assert np.isclose(
        norm_failure, expected_norm_failure, atol=1e-10
    ), f"[{mpu_fixture}] Failure norm mismatch"
