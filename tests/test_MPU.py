import pytest
import numpy as np
import quimb.tensor as qtn
from unittest.mock import patch
from mpu_decomposition.MPU import AbstractMPU, UniformMPU

# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def identity_mpu():
    """Generates an exact identity MPU tensor with minimal canonical bond dimension (D=1)."""
    d, D = 2, 1
    A = np.einsum("ij,ab->ijab", np.eye(d), np.eye(D))
    l_in = np.ones(D) / np.sqrt(D)
    r_in = np.ones(D) / np.sqrt(D)
    return d, D, A, l_in, r_in


@pytest.fixture
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


@pytest.fixture
def local_unitary_mpu():
    """Generates an MPU applying a local Pauli-X gate globally (D=1)."""
    d, D = 2, 1
    X = np.array([[0.0, 1.0], [1.0, 0.0]])
    A = np.einsum("ij,ab->ijab", X, np.eye(D))
    l_in = np.ones(D)
    r_in = np.ones(D)
    return d, D, A, l_in, r_in


@pytest.fixture
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


@pytest.fixture
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
    Verifies that the base AbstractMPU class cannot be instantiated directly.
    """
    # Attempting to instantiate an abstract class should raise a TypeError
    with pytest.raises(TypeError, match="Can't instantiate abstract class AbstractMPU"):
        AbstractMPU(N=4, d=2, l_vec=qtn.Tensor(), r_vec=qtn.Tensor())


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
    # Entangling power q_unif is lower-bounded by 1.0 for unitary channels
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
    mpu.L2 = np.eye(2)
    mpu.R2 = np.eye(3)

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
    mpu.L2 = np.array([[1.0]])
    mpu.R2 = np.array([[-1.0]])

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
    mpu.L2 = np.array([[1.0]])
    mpu.R2 = np.array([[1.0]])

    expected_q = np.sqrt(d_phys)
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
    assert q_identity == pytest.approx(np.sqrt(d_id))

    # The V-model introduces complexity/deformation relative to the identity
    assert q_semisimple >= q_identity

    # The entangling power must be a physical real quantity
    assert np.isreal(q_semisimple) or np.abs(np.imag(q_semisimple)) < 1e-12


def test_q_unif_ill_conditioned_matrix(identity_mpu):
    """Verifica la resilienza a matrici quasi singolari tramite pinv."""
    d, D, A, l_in, r_in = identity_mpu
    with patch("mpu_decomposition.MPU.check_mpo_unitarity", return_value=True), patch(
        "mpu_decomposition.MPU.check_assumption_1", return_value=(True, None, None)
    ), patch(
        "mpu_decomposition.MPU.UniformMPU._compute_boundary_operators",
        return_value=(np.eye(d), np.eye(d)),
    ):
        mpu = UniformMPU(A, l_in, r_in, N=4)

    mpu.L2 = np.array([[1.0, 0.0], [0.0, 1e-20]])
    mpu.R2 = np.eye(d)
    q_val = mpu._compute_q_unif()
    assert isinstance(q_val, float)
    assert not np.isnan(q_val)
