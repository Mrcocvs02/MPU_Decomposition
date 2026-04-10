import pytest  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]
from mpu_decomposition.checks import check_mpo_unitarity, check_assumption_1
from mpu_decomposition.utils import optimize_q_unif
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
def identity_mpu():
    """Generates an exact identity MPU tensor with minimal canonical bond dimension (D=1)."""
    d, D = 2, 1
    A = np.einsum("ij,ab->ijab", np.eye(d), np.eye(D))
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
    A = np.einsum("ij,ab->ijab", X, np.eye(D))
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

    # Shape becomes (2, 2, 2, 2)
    A = np.array([[A_00, A_01], [A_10, A_11]], dtype=float)

    l_in = np.array([1.0, 1.0], dtype=float)
    r_in = np.array([1.0, -2.0], dtype=float)

    return d, D, A, l_in, r_in


@pytest.fixture(scope="module")
def random_complex_mpo():
    """Generates a random dense complex MPO tensor."""
    d, D = 3, 2
    np.random.seed(42)
    A = np.random.randn(d, d, D, D) + 1j * np.random.randn(d, d, D, D)
    l_in = np.random.randn(D) + 1j * np.random.randn(D)
    r_in = np.random.randn(D) + 1j * np.random.randn(D)
    return d, D, A, l_in, r_in


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
@pytest.mark.parametrize(
    "cached_fixture",
    ["identity_q_unif_result", "cz_q_unif_result", "semisimple_q_unif_result"],
)
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
@pytest.mark.parametrize(
    "cached_fixture",
    ["identity_q_unif_result", "cz_q_unif_result", "semisimple_q_unif_result"],
)
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
@pytest.mark.parametrize(
    "cached_fixture",
    ["identity_q_unif_result", "cz_q_unif_result", "semisimple_q_unif_result"],
)
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
@pytest.mark.parametrize(
    "cached_fixture",
    ["identity_q_unif_result", "cz_q_unif_result", "semisimple_q_unif_result"],
)
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
