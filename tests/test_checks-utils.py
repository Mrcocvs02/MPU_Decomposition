import pytest # pyright: ignore[reportMissingImports]
import numpy as np # pyright: ignore[reportMissingImports]
from mpu_decomposition.checks import check_mpo_unitarity, check_assumption_1

# ============================================================================
# ============================================================================
# ============================================================================
#UNIFORM BULK CASE TESTS
# ============================================================================
# ============================================================================
# ============================================================================





# FIXTURES
# ============================================================================

@pytest.fixture
def identity_mpu():
    """Generates an exact identity MPU tensor with minimal canonical bond dimension (D=1)."""
    d, D = 2, 1
    A = np.einsum('ij,ab->ijab', np.eye(d), np.eye(D))
    l_in = np.ones(D) / np.sqrt(D)
    r_in = np.ones(D) / np.sqrt(D)
    return d, D, A, l_in, r_in

@pytest.fixture
def local_unitary_mpu():
    """Generates an MPU applying a local Pauli-X gate globally (D=1)."""
    d, D = 2, 1
    X = np.array([[0.0, 1.0], [1.0, 0.0]])
    A = np.einsum('ij,ab->ijab', X, np.eye(D))
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
def random_complex_mpu():
    """Generates a random dense complex MPU tensor."""
    d, D = 3, 2
    np.random.seed(42)
    A = np.random.randn(d, d, D, D) + 1j * np.random.randn(d, d, D, D)
    l_in = np.random.randn(D) + 1j * np.random.randn(D)
    r_in = np.random.randn(D) + 1j * np.random.randn(D)
    return d, D, A, l_in, r_in

# SECTION 1: MPO UNITARITY CHECKS
# ============================================================================

@pytest.mark.parametrize("N", [1, 5, 10, 50])
@pytest.mark.parametrize("mpu_fixture", [
    "identity_mpu",
    "local_unitary_mpu",
    "cz_interaction_mpu"
])
def test_check_mpo_unitarity_valid(mpu_fixture, N, request):
    """
    Evaluates unitarity on physically sound MPU tensors.
    Locked to N_max=1 until the trace metric in check_mpo_unitarity is fixed.
    """
    _, _, A, l_in, r_in = request.getfixturevalue(mpu_fixture)
    
    is_unitary = check_mpo_unitarity(N_max=N, A_np=A, l_in=l_in, r_in=r_in, tol=1e-6)
    assert is_unitary

@pytest.mark.parametrize("N", [1, 5, 10, 50])
def test_check_mpo_unitarity_failure(random_complex_mpu, N):
    """
    A strictly random MPO violates unitarity.
    """
    _, _, A, l_in, r_in = random_complex_mpu
    
    is_unitary = check_mpo_unitarity(N_max=N, A_np=A, l_in=l_in, r_in=r_in, tol=1e-6, early_stop=True)
    assert not is_unitary

# SECTION 2: ASSUMPTION 1 (INJECTIVITY)
# ============================================================================

@pytest.mark.parametrize("mpu_fixture", [
    "identity_mpu",
    "local_unitary_mpu",
    "cz_interaction_mpu"
])
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