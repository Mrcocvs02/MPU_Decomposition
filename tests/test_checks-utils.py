import pytest  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]
from mpu_decomposition.checks import check_mpo_unitarity, check_assumption_1

# ============================================================================
# ============================================================================
# ============================================================================
# UNIFORM BULK CASE TESTS
# ============================================================================
# ============================================================================
# ============================================================================


# FIXTURES
# ============================================================================


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

    # Shape becomes (2, 2, 2, 2)
    A = np.array([[A_00, A_01], [A_10, A_11]], dtype=float)

    l_in = np.array([1.0, 1.0], dtype=float)
    r_in = np.array([1.0, -2.0], dtype=float)

    return d, D, A, l_in, r_in


@pytest.fixture
def random_complex_mpo():
    """Generates a random dense complex MPO tensor."""
    d, D = 3, 2
    np.random.seed(42)
    A = np.random.randn(d, d, D, D) + 1j * np.random.randn(d, d, D, D)
    l_in = np.random.randn(D) + 1j * np.random.randn(D)
    r_in = np.random.randn(D) + 1j * np.random.randn(D)
    return d, D, A, l_in, r_in


# SECTION 1: MPO UNITARITY CHECKS
# ============================================================================


@pytest.mark.parametrize("N", [1, 5, 10, 50])
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


@pytest.mark.parametrize("N", [1, 5, 10, 50])
def test_check_mpo_unitarity_failure(random_complex_mpo, N):
    """
    A strictly random MPO violates unitarity.
    """
    _, _, A, l_in, r_in = random_complex_mpo

    is_unitary = check_mpo_unitarity(
        N_max=N, A_np=A, l_in=l_in, r_in=r_in, tol=1e-6, early_stop=True
    )
    assert not is_unitary


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
