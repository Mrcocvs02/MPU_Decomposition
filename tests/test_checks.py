import pytest # pyright: ignore[reportMissingImports]
import numpy as np # pyright: ignore[reportMissingImports]
from mpu_decomposition.checks import check_unitary, check_assumption_1

def test_check_unitary_identity():
    """
    Verifies that the identity tensor (delta_ij * delta_ab) satisfies the 
    local isometry condition (Left Canonical Form).
    """
    d, D = 2, 3
    # Construct A_{i,j,a,b} = delta_{ij} delta_{ab}
    A = np.einsum('ij,ab->ijab', np.eye(d), np.eye(D))
    
    # Standard tolerance 1e-12 is sufficient for float64 precision
    assert check_unitary(A, tol=1e-12)

def test_check_unitary_failure():
    """
    Verifies that a non-isometric tensor (e.g., matrix of ones) fails the test.
    """
    d, D = 2, 2
    A = np.ones((d, d, D, D))
    assert not check_unitary(A, tol=1e-12)

def test_check_unitary_random_isometry():
    """
    Verifies the left canonical form for a non-trivial random isometry.
    """
    d, D = 2, 2
    np.random.seed(42)
    # Generate a random unitary matrix of size (d*D, d*D)
    H = np.random.randn(d*D, d*D) + 1j * np.random.randn(d*D, d*D)
    Q, _ = np.linalg.qr(H)
    
    # Reshape Q from (d*D, d*D) to indices (phys_in, bond_L, phys_out, bond_R)
    # then transpose to match A's signature: (phys_in, phys_out, bond_L, bond_R)
    A = Q.reshape(d, D, d, D).transpose(0, 2, 1, 3)
    
    assert check_unitary(A, tol=1e-12)

def test_check_assumption_1_identity():
    """
    The identity tensor must satisfy Assumption 1 as it has full rank.
    """
    d, D = 2, 1
    A = np.einsum('ij,ab->ijab', np.eye(d), np.eye(D))
    l = np.array([1.0])
    r = np.array([1.0])
    
    success, s_left, s_right = check_assumption_1(A, l, r)
    
    assert success
    assert len(s_left) == D
    assert np.all(s_left > 1e-12)
    assert np.all(s_right > 1e-12)

def test_check_assumption_1_random_valid():
    """
    A random complex tensor with d^2 >= D satisfies injectivity almost certainly.
    This verifies the numerical robustness of the SVD on generic tensors.
    """
    d, D = 4, 2
    np.random.seed(42)
    
    A = np.random.randn(d, d, D, D) + 1j * np.random.randn(d, d, D, D)
    l = np.random.randn(D) + 1j * np.random.randn(D)
    r = np.random.randn(D) + 1j * np.random.randn(D)
    
    success, s_left, s_right = check_assumption_1(A, l, r)
    
    assert success
    assert np.all(s_left > 1e-12)
    assert np.all(s_right > 1e-12)

def test_check_assumption_1_singular():
    """
    A degenerate tensor (e.g., null matrix) must fail Assumption 1.
    """
    d, D = 2, 2
    A = np.zeros((d, d, D, D))
    l = np.ones(D)
    r = np.ones(D)
    
    success, s_left, s_right = check_assumption_1(A, l, r)
    
    assert not success
    assert np.all(s_left < 1e-12)
    assert np.all(s_right < 1e-12)

def test_check_assumption_1_insufficient_physical_dimension():
    """
    If d^2 < D, the map from the virtual bond to the physical space 
    cannot be injective due to rank bounds. Assumption 1 must strictly fail.
    """
    d, D = 2, 5  # d^2 = 4, which is less than D = 5
    np.random.seed(42)
    
    A = np.random.randn(d, d, D, D) + 1j * np.random.randn(d, d, D, D)
    l = np.random.randn(D) + 1j * np.random.randn(D)
    r = np.random.randn(D) + 1j * np.random.randn(D)
    
    success, s_left, s_right = check_assumption_1(A, l, r)
    
    assert not success
    # The maximum possible rank is d^2 = 4, so at least one singular value is ~0
    assert np.sum(s_left > 1e-12) <= d**2
    assert np.sum(s_right > 1e-12) <= d**2
    