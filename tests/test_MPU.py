import pytest
import numpy as np
import quimb.tensor as qtn
from unittest.mock import patch
from mpu_decomposition.MPU import CircuitDecomposition, UniformMPU


@pytest.fixture(scope="module")
def identity_lcu_data(identity_mpu):
    d, D, A, l_in, r_in = identity_mpu
    mpu = UniformMPU(A=A, l_vec=l_in, r_vec=r_in, N=4)
    h_R, h_L, basis_R, basis_L, labels_R, labels_L, C = mpu._build_lcu_data()
    return D, h_R, h_L, basis_R, basis_L, labels_R, labels_L, C, mpu


@pytest.fixture(scope="module")
def cz_lcu_data(cz_interaction_mpu):
    d, D, A, l_in, r_in = cz_interaction_mpu
    mpu = UniformMPU(A=A, l_vec=l_in, r_vec=r_in, N=4)
    h_R, h_L, basis_R, basis_L, labels_R, labels_L, C = mpu._build_lcu_data()
    return D, h_R, h_L, basis_R, basis_L, labels_R, labels_L, C, mpu


@pytest.fixture(scope="module")
def semisimple_lcu_data(semisimple_v_mpu):
    d, D, A, l_in, r_in = semisimple_v_mpu
    mpu = UniformMPU(A=A, l_vec=l_in, r_vec=r_in, N=4)
    h_R, h_L, basis_R, basis_L, labels_R, labels_L, C = mpu._build_lcu_data()
    return D, h_R, h_L, basis_R, basis_L, labels_R, labels_L, C, mpu


ALL_LCU_DATA = ["identity_lcu_data", "cz_lcu_data", "semisimple_lcu_data"]
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
# Group 5: Factored Pauli LCU Decomposition
# =====================================================================


@pytest.mark.parametrize("lcu_fixture", ALL_LCU_DATA)
def test_lcu_h_shapes_consistent(lcu_fixture, request):
    """h_R and h_L must have shape (D, n_paulis) where n_paulis = 4^n_anc."""
    D, h_R, h_L, basis_R, basis_L, labels_R, labels_L, C, mpu = request.getfixturevalue(
        lcu_fixture
    )
    n_bonds = h_R.shape[0]
    assert n_bonds == D, f"Expected {D} bond terms, got {n_bonds}"
    assert h_R.shape[0] == h_L.shape[0]
    assert h_R.shape[1] == len(basis_R)
    assert h_L.shape[1] == len(basis_L)


@pytest.mark.parametrize("lcu_fixture", ALL_LCU_DATA)
def test_lcu_pauli_basis_size(lcu_fixture, request):
    """Pauli basis must have 4^n elements for n = ceil(log2(D))."""
    D, h_R, h_L, basis_R, basis_L, labels_R, labels_L, C, mpu = request.getfixturevalue(
        lcu_fixture
    )
    import math

    n_anc = max(1, math.ceil(math.log2(D)))
    expected_size = 4**n_anc
    assert len(basis_R) == expected_size
    assert len(basis_L) == expected_size
    assert len(labels_R) == expected_size
    assert len(labels_L) == expected_size


@pytest.mark.parametrize("lcu_fixture", ALL_LCU_DATA)
def test_lcu_normalization_constant(lcu_fixture, request):
    """C must equal Σ_m (Σ_i |h_R[m,i]|)(Σ_j |h_L[m,j]|)."""
    D, h_R, h_L, basis_R, basis_L, labels_R, labels_L, C, mpu = request.getfixturevalue(
        lcu_fixture
    )
    C_recomputed = sum(
        np.sum(np.abs(h_R[m])) * np.sum(np.abs(h_L[m])) for m in range(h_R.shape[0])
    )
    assert C == pytest.approx(C_recomputed, abs=1e-12)


@pytest.mark.parametrize("lcu_fixture", ALL_LCU_DATA)
def test_lcu_C_positive(lcu_fixture, request):
    """The LCU 1-norm must be strictly positive."""
    D, h_R, h_L, basis_R, basis_L, labels_R, labels_L, C, mpu = request.getfixturevalue(
        lcu_fixture
    )
    assert C > 0


@pytest.mark.parametrize("lcu_fixture", ALL_LCU_DATA)
def test_lcu_reconstruction_from_paulis(lcu_fixture, request):
    """
    Σ_{m,i,j} h_R[m,i] h_L[m,j] (P_i ⊗ P_j) must reconstruct the merging operator.
    """
    D, h_R, h_L, basis_R, basis_L, labels_R, labels_L, C, mpu = request.getfixturevalue(
        lcu_fixture
    )
    import math

    n_anc = max(1, math.ceil(math.log2(D)))
    dim = 2**n_anc  # padded dimension

    M_reconstructed = np.zeros((dim * dim, dim * dim), dtype=complex)
    for m in range(h_R.shape[0]):
        for i in range(h_R.shape[1]):
            if np.abs(h_R[m, i]) < 1e-18:
                continue
            for j in range(h_L.shape[1]):
                if np.abs(h_L[m, j]) < 1e-18:
                    continue
                M_reconstructed += (
                    h_R[m, i] * h_L[m, j] * np.kron(basis_R[i], basis_L[j])
                )

    # Compare against the dense merge operator (padded to dim x dim)
    M_ref_raw = mpu.get_merging_operator()  # (D, D, D, D)
    M_ref = np.zeros((dim, dim, dim, dim), dtype=complex)
    M_ref[:D, :D, :D, :D] = M_ref_raw
    M_ref = M_ref.reshape(dim * dim, dim * dim)

    assert np.allclose(
        M_reconstructed, M_ref, atol=1e-8
    ), f"Pauli reconstruction error: {np.max(np.abs(M_reconstructed - M_ref)):.2e}"


@pytest.mark.parametrize("lcu_fixture", ALL_LCU_DATA)
def test_lcu_pauli_bases_are_hermitian_unitary(lcu_fixture, request):
    """Every Pauli basis element must be Hermitian and unitary."""
    D, h_R, h_L, basis_R, basis_L, labels_R, labels_L, C, mpu = request.getfixturevalue(
        lcu_fixture
    )
    for label, P in zip(labels_R, basis_R):
        dim = P.shape[0]
        Id = np.eye(dim)
        assert np.allclose(P, P.conj().T, atol=1e-14), f"{label} not Hermitian"
        assert np.allclose(P @ P, Id, atol=1e-14), f"{label} not involutory"


# =====================================================================
# Group 6: UniformMPU LCU Integration & Caching
# =====================================================================


def test_uniformmpu_build_lcu_success(cz_interaction_mpu):
    """_build_lcu_data returns a 7-tuple with correct types."""
    d, D, A, l_in, r_in = cz_interaction_mpu
    mpu = UniformMPU(A=A, l_vec=l_in, r_vec=r_in, N=4)

    res = mpu._build_lcu_data()
    assert res is not None

    h_R, h_L, basis_R, basis_L, labels_R, labels_L, C = res
    assert h_R.shape[0] == D
    assert h_L.shape[0] == D
    assert isinstance(C, float)
    assert C > 0


def test_uniformmpu_lcu_caching(cz_interaction_mpu):
    """Multiple calls return the same cached object."""
    d, D, A, l_in, r_in = cz_interaction_mpu
    mpu = UniformMPU(A=A, l_vec=l_in, r_vec=r_in, N=4)

    res1 = mpu._build_lcu_data()
    res2 = mpu._build_lcu_data()
    assert res1 is res2


# =====================================================================
# Group 7: Unitary U of LCU
# =====================================================================


@pytest.mark.parametrize("lcu_fixture", ALL_LCU_DATA)
def test_lcu_reconstruction(lcu_fixture, request):
    """Pauli decomposition must reconstruct the merging operator."""
    D, h_R, h_L, basis_R, basis_L, labels_R, labels_L, C, mpu = request.getfixturevalue(
        lcu_fixture
    )
    from mpu_decomposition.utils import build_merge_factors
    from mpu_decomposition.checks import verify_factored_decomposition

    factors, _ = build_merge_factors(mpu.R_inv, mpu.L_inv)
    error = verify_factored_decomposition(factors, h_R, h_L, basis_R, basis_L)

    assert error < 1e-10, f"Pauli reconstruction error: {error:.2e}"
