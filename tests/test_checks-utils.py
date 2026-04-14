import pytest  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]
from mpu_decomposition.checks import check_mpo_unitarity, check_assumption_1
from mpu_decomposition.utils import optimize_q_unif, get_merging_operator
from mpu_decomposition.MPU import UniformMPU

# ============================================================================
# ============================================================================
# ============================================================================
# UNIFORM BULK CASE TESTS
# ============================================================================
# ============================================================================
# ============================================================================


# FIXTURES
# ============================================================================


@pytest.fixture(scope="module")
def identity_q_unif_result(identity_mpu):
    """Cached optimize_q_unif result for identity MPU."""
    d, D, A, l_in, r_in = identity_mpu
    np.random.seed(0)
    sigma, tau, q_val = optimize_q_unif(A, l_in, r_in)
    return d, D, A, l_in, r_in, sigma, tau, q_val


@pytest.fixture(scope="module")
def cz_q_unif_result(cz_interaction_mpu):
    """Cached optimize_q_unif result for CZ MPU."""
    d, D, A, l_in, r_in = cz_interaction_mpu
    np.random.seed(0)
    sigma, tau, q_val = optimize_q_unif(A, l_in, r_in)
    return d, D, A, l_in, r_in, sigma, tau, q_val


@pytest.fixture(scope="module")
def semisimple_q_unif_result(semisimple_v_mpu):
    """Cached optimize_q_unif result for semisimple V MPU."""
    d, D, A, l_in, r_in = semisimple_v_mpu
    np.random.seed(0)
    sigma, tau, q_val = optimize_q_unif(A, l_in, r_in)
    return d, D, A, l_in, r_in, sigma, tau, q_val


ALL_Q_UNIF_CACHED = [
    "identity_q_unif_result",
    "cz_q_unif_result",
    "semisimple_q_unif_result",
]

# ============================================================================
# SECTION 1: MPO UNITARITY CHECKS
# ============================================================================


@pytest.mark.parametrize("N", [1, 5, 10])
@pytest.mark.parametrize(
    "mpu_fixture",
    ["identity_mpu", "local_unitary_mpu", "cz_interaction_mpu", "semisimple_v_mpu"],
)
def test_check_mpo_unitarity_valid(mpu_fixture, N, request):
    """
    Evaluates unitarity on physically sound MPU tensors.
    Locked to N_max=1 until the trace metric in check_mpo_unitarity is fixed.
    """
    _, _, A, l_in, r_in = request.getfixturevalue(mpu_fixture)

    is_unitary = check_mpo_unitarity(N_max=N, A_np=A, l_in=l_in, r_in=r_in, tol=1e-6)
    assert is_unitary


@pytest.mark.parametrize("N", [1, 5, 10])
def test_check_mpo_unitarity_failure(random_complex_mpo, N):
    """
    A strictly random MPO violates unitarity.
    """
    _, _, A, l_in, r_in = random_complex_mpo

    is_unitary = check_mpo_unitarity(
        N_max=N, A_np=A, l_in=l_in, r_in=r_in, tol=1e-6, early_stop=True
    )
    assert not is_unitary


# ============================================================================
# SECTION 2: ASSUMPTION 1 (INJECTIVITY)
# ============================================================================


@pytest.mark.parametrize(
    "mpu_fixture",
    ["identity_mpu", "local_unitary_mpu", "cz_interaction_mpu", "semisimple_v_mpu"],
)
def test_check_assumption_1_valid(mpu_fixture, request):
    """
    Valid MPUs map the bond space injectively to the physical space.
    """
    _, D, A, l_in, r_in = request.getfixturevalue(mpu_fixture)

    success, s_left, s_right = check_assumption_1(A, l_in, r_in)

    assert success
    assert np.sum(s_left > 1e-12) == D
    assert np.sum(s_right > 1e-12) == D


def test_check_assumption_1_deficient_rank():
    """
    Forces Assumption 1 to fail (d^2 < D).
    """
    d, D = 2, 5
    np.random.seed(42)
    A = np.random.randn(d, d, D, D)
    l_in = np.random.randn(D)
    r_in = np.random.randn(D)

    success, s_left, s_right = check_assumption_1(A, l_in, r_in)

    assert not success
    assert np.sum(s_left > 1e-12) <= d**2
    assert np.sum(s_right > 1e-12) <= d**2


def test_check_assumption_1_singular():
    """
    A degenerate tensor (zeros) must fail injectivity.
    """
    d, D = 2, 2
    A = np.zeros((d, d, D, D))
    l_in = np.ones(D)
    r_in = np.ones(D)

    success, s_left, s_right = check_assumption_1(A, l_in, r_in)

    assert not success
    assert np.all(s_left < 1e-12)
    assert np.all(s_right < 1e-12)


# ============================================================================
# SECTION 3: q_unif optimization
# ============================================================================
@pytest.mark.parametrize("cached_fixture", ALL_Q_UNIF_CACHED)
def test_optimize_q_unif_density_matrices_valid(cached_fixture, request):
    d, D, A, l_in, r_in, sigma, tau, q_val = request.getfixturevalue(cached_fixture)

    for label, rho in [("sigma", sigma), ("tau", tau)]:
        assert np.allclose(rho, rho.conj().T, atol=1e-8), f"{label} not Hermitian"
        assert np.real(np.trace(rho)) == pytest.approx(1.0, abs=1e-6)
        eigvals = np.linalg.eigvalsh(rho)
        assert np.all(eigvals > -1e-8), f"{label} negative eigenvalue: {eigvals}"


# ---------------------------------------------------------------------
# 5.2: Returned density matrices have correct shape
# ---------------------------------------------------------------------
@pytest.mark.parametrize(
    "cached_fixture, expected_D",
    [
        ("identity_q_unif_result", 2),
        ("cz_q_unif_result", 2),
        ("semisimple_q_unif_result", 4),
    ],
)
def test_optimize_q_unif_output_shapes(cached_fixture, expected_D, request):
    """
    Verifies that sigma and tau have shape (d, d) matching the bond dimension.
    """
    d, D, A, l_in, r_in, sigma, tau, q_val = request.getfixturevalue(cached_fixture)
    assert sigma.shape == (expected_D, expected_D)
    assert tau.shape == (expected_D, expected_D)


# ---------------------------------------------------------------------
# 5.3: Optimized q_val is a finite positive real number
# ---------------------------------------------------------------------
@pytest.mark.parametrize("cached_fixture", ALL_Q_UNIF_CACHED)
def test_optimize_q_unif_returns_finite_positive(cached_fixture, request):
    """
    Verifies that the optimized q value is real, finite, and positive.
    """
    d, D, A, l_in, r_in, sigma, tau, q_val = request.getfixturevalue(cached_fixture)
    assert isinstance(q_val, (float, np.floating))
    assert not np.isnan(q_val)
    assert not np.isinf(q_val)
    assert q_val > 0.0


# ---------------------------------------------------------------------
# 5.4: Identity MPU gives known analytical q_unif
# ---------------------------------------------------------------------
def test_optimize_q_unif_identity_analytical(identity_q_unif_result):
    """
    For the identity MPU (D=1), the optimal q_unif has a known value.
    The optimizer must recover it.
    """
    d, D, A, l_in, r_in, sigma, tau, q_val = identity_q_unif_result
    mpu = UniformMPU(A=A, l_vec=l_in, r_vec=r_in, N=4)
    expected_q = mpu.q_unif
    assert q_val == pytest.approx(expected_q, abs=1e-4)


# ---------------------------------------------------------------------
# 5.5: Optimizer result <= constructor result (rank-1 boundary)
# ---------------------------------------------------------------------
@pytest.mark.parametrize("cached_fixture", ALL_Q_UNIF_CACHED)
def test_optimize_q_unif_leq_constructor(cached_fixture, request):
    """
    The optimized q must be <= the constructor's q_unif, since the optimizer
    searches over all density matrices (rank >= 1) while the constructor
    uses rank-1 projectors from the input boundary vectors.
    """
    d, D, A, l_in, r_in, sigma, tau, q_optimized = request.getfixturevalue(
        cached_fixture
    )
    mpu = UniformMPU(A=A, l_vec=l_in, r_vec=r_in, N=4)
    q_constructor = mpu.q_unif
    assert q_optimized <= q_constructor + 1e-4


# ---------------------------------------------------------------------
# 5.6: Self-consistency – recompute q from returned density matrices
# ---------------------------------------------------------------------
@pytest.mark.parametrize("cached_fixture", ALL_Q_UNIF_CACHED)
def test_optimize_q_unif_self_consistency(cached_fixture, request):
    """
    Reconstruct L2 and R2 from the returned sigma/tau using the same
    einsum contractions, then manually compute q and compare to res.fun.
    """
    d, D, A, l_in, r_in, sigma, tau, q_val = request.getfixturevalue(cached_fixture)
    eps_reg = 1e-8

    T = A
    L_in = np.outer(l_in, l_in.conj())
    R_in = np.outer(r_in, r_in.conj())

    # Reconstruct boundary operators from returned density matrices
    L2 = np.einsum("ij, oipq, pm, ojmn -> qn", sigma, T, L_in, T.conj())
    R2 = np.einsum("ij, oipq, qn, ojmn -> pm", tau, T, R_in, T.conj())

    # Regularized pseudoinverse via eigendecomposition (same as in loss)
    def reg_inv(M):
        eigvals, eigvecs = np.linalg.eigh(M)
        inv_diag = np.diag(1.0 / (np.maximum(eigvals, 0.0) + eps_reg))
        return eigvecs @ inv_diag @ eigvecs.conj().T

    inv_L = reg_inv(L2)
    inv_R = reg_inv(R2)

    q_recomputed = np.sqrt(np.abs(np.real(np.trace(inv_R @ inv_L.T))))

    assert q_recomputed == pytest.approx(
        q_val, abs=1e-4
    ), f"Recomputed q ({q_recomputed}) != returned q ({q_val})"


# ---------------------------------------------------------------------
# 5.7: Determinism under fixed random seed
# ---------------------------------------------------------------------
def test_optimize_q_unif_deterministic(cz_interaction_mpu):
    """
    With the same random seed, two calls must return identical results.
    """
    d, D, A, l_in, r_in = cz_interaction_mpu

    np.random.seed(42)
    sigma_1, tau_1, q_1 = optimize_q_unif(A, l_in, r_in)

    np.random.seed(42)
    sigma_2, tau_2, q_2 = optimize_q_unif(A, l_in, r_in)

    assert q_1 == pytest.approx(q_2, abs=1e-12)
    assert np.allclose(sigma_1, sigma_2, atol=1e-12)
    assert np.allclose(tau_1, tau_2, atol=1e-12)


# ---------------------------------------------------------------------
# 5.8: Ill-conditioned (semisimple D=5) does not crash
# ---------------------------------------------------------------------


@pytest.mark.parametrize("eps_reg", [1e-6, 1e-8, 1e-12])
def test_optimize_q_unif_semisimple_no_crash(eps_reg, semisimple_v_mpu):
    """
    The D=5 semisimple case may be ill-conditioned; verify no NaN/Inf.
    """
    d, D, A, l_in, r_in = semisimple_v_mpu
    np.random.seed(8)

    _, _, q_val = optimize_q_unif(A, l_in, r_in, eps_reg=eps_reg)

    assert not np.isnan(q_val)
    assert not np.isinf(q_val)
    assert q_val > 0.0


# ---------------------------------------------------------------------
# 6: Merging operator
# ---------------------------------------------------------------------
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


def test_merging_operator_nonsquare_raises():
    L_inv = np.ones((2, 3))
    R_inv = np.ones((2, 3))
    with pytest.raises(ValueError, match="square matrices"):
        get_merging_operator(L_inv, R_inv)


def test_merging_operator_dimension_mismatch_raises():
    L_inv = np.eye(2)
    R_inv = np.eye(3)
    with pytest.raises(ValueError, match="same dimension"):
        get_merging_operator(L_inv, R_inv)
