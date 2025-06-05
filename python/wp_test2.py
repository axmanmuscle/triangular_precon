## testing VMPPA from Parente et al.
import numpy as np
import scipy.linalg as sl

def prox_twonorm_element(x, lambd, A, b, i):
    n = x.size
    y = A.T @ b

    G = np.linalg.inv(A.T@A + 1/lambd * np.eye(n))
    Gi = G[i, :]

    pi = np.dot(Gi, y + x/lambd)

    return pi

def prox_twonorm(x, lambd, A, b):
    n = x.size

    y = A.T @ b
    G = np.linalg.inv(A.T@A + 1/lambd * np.eye(n))

    out = G @ (y + x / lambd)
    return out

def apply_weighted_prox(x, V, prox):
    """
    This is going to have a lower triangular matrix V and a diagonal matrix of the prox operators
    and solve the system using forward substitution
    """
    t = 0.5
    n = V.shape[0]
    if V.shape[1] != n:
        print('V must be square')
        return
    
    vx = V@x
    y = np.zeros((n, 1))
    
    v00 = V[0, 0]
    y[0, 0] = prox(vx[0, 0] / v00, t/v00)
    for i in range(1, n):
        tmp1 = np.dot(V[i, 0:i-1], y[0:i-1, 0])
        vii = V[i, i]
        x_in = vx[i, 0] - tmp1
        y[i, 0] = prox(x_in/vii, t/vii)

    return y

def apply_weighted_prox_element(x, V, prox):
    """
    This is going to have a lower triangular matrix V and a diagonal matrix of the prox operators
    and solve the system using forward substitution
    """
    t = 0.5
    n = V.shape[0]
    if V.shape[1] != n:
        print('V must be square')
        return
    
    vx = V@x
    y = np.zeros((n, 1))
    
    v00 = V[0, 0]
    y[0, 0] = prox(vx[0, 0] / v00, t/v00, 0)
    for i in range(1, n):
        tmp1 = np.dot(V[i, 0:i-1], y[0:i-1, 0])
        vii = V[i, i]
        x_in = vx[i, 0] - tmp1
        y[i, 0] = prox(x_in/vii, t/vii, i)

    return y

def twonorm():
    rng = np.random.default_rng(2025)
    iters = 10

    n = 10
    A = rng.normal(0, 1, size=(n, n))
    x = rng.normal(0, 1, size=(n, 1))
    b = A@x

    f = lambda z: 0.5 * np.linalg.norm(A@z - b)**2
    gradf = lambda z: A.T @ (A@z - b)

    # lambd = 0.5
    # hess = np.linalg.inv(A@A.T + 1/lambd * np.eye(n))
    # print(hess)
    L = np.linalg.norm(A,2)

    lambd = 1 / (L*L)

    print('**** gradient descent ****')
    x0 = np.zeros_like(x)
    xi = x0
    for i in range(iters):
        gf = gradf(xi)
        xi = xi - lambd*gf

        if i%2 == 0:
            print(f'iteration {i}: obj val {f(xi)}')

    print(f'output {xi}')
    print(f'error {np.linalg.norm(xi - x)}')

    print('**** prox point ****')
    xi = x0
    for i in range(iters):
        xi = prox_twonorm(xi, lambd, A, b)

        if i%2 == 0:
            print(f'iteration {i}: obj val {f(xi)}')

    print(f'output {xi}')
    print(f'error {np.linalg.norm(xi - x)}')

    print('**** weighted prox ****')
    xi = x0
    prox = lambda z, t: prox_twonorm(z, t, A, b)
    V = np.eye(A.shape[0])
    for i in range(iters):
        xi = apply_weighted_prox(xi, V, prox)

        if i%2 == 0:
            print(f'iteration {i}: obj val {f(xi)}')
    

    return 0

def test_proximal_point_l21(num_iters=20, tau=1.0, lam=0.1, verbose=True):
    """
    Solve min_X ||X - M||_F^2 + lambda * ||X||_{2,1} using the proximal point algorithm.
    
    Arguments:
        num_iters: number of proximal point iterations
        tau: step size (lambda in prox notation)
        lam: regularization strength in ||X||_{2,1}
        verbose: whether to print intermediate convergence info
    """
    def prox_l21(M, lam):
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        scale = np.maximum(0, 1 - lam / np.clip(norms, a_min=1e-8, a_max=None))
        return scale * M

    # Step 1: Create synthetic data (3x4 matrix)
    np.random.seed(42)
    M = np.random.randn(5, 4)
    M[0:2] *= 0.1  # Rows likely to shrink to zero
    M[2:4] *= 2.0  # Rows likely to survive
    M[4] *= 0.5    # Ambiguous

    # Step 2: Initialize X
    X = np.zeros_like(M)

    # Step 3: Run PPA
    for k in range(num_iters):
        X_old = X.copy()
        X = prox_l21(X - tau * (X - M), tau * lam)
        if verbose:
            delta = np.linalg.norm(X - X_old)
            print(f"Iter {k+1:02d}: Δ = {delta:.4e}, ||X||_2,1 = {np.sum(np.linalg.norm(X, axis=1)):.4f}")

    return X, M

def makeH(n):
    H = np.zeros((n, n))

    H[0, 0] = n/2
    H[0, n-1] = 5*n
    H[n-1, 0] = -5*n

    for i in range(1, n-1):
        H[i, i] = n + i - 1

    H[1:n-1, n-1] = 1
    for i in range(1, n-1):
        H[i, :i-1] = 1

    H[n-1, 1:n-1] = -1

    return H

def F(z, f, H):
    Hz  = H@z

    n = z.shape[0]
    ftilde = np.zeros_like(z)
    for i in range(0,n,2):
        coeff = (1 + (-1)**(i))/2 
        # print(f'coeff {coeff}')
        ftilde[i] = coeff * f(z[i])

    return ftilde + Hz


def main():
    n = 16
    H = makeH(n)

    f = lambda x: x + np.exp(-1*x*x)

    z = np.ones(n)
    print(F(z, f, H))

    # print(H)
    # symH = 0.5 * (H + H.T)
    # print(symH)
    # e = np.linalg.eig(symH)
    # print(e)
    return 0

def tests():
    A = sl.hilbert(15)
    b = np.ones(15)
    x = np.random.rand(15)
    lambd = 0.5
    p = prox_twonorm(x, lambd, A, b)
    print(p)

    pi = np.zeros(15)
    for i in range(15):
        pi[i] = prox_twonorm_element(x, lambd, A, b, i)

    print(np.linalg.norm(p - pi))

if __name__ == "__main__":
    test_proximal_point_l21(num_iters = 100, tau=0.25, lam=0.1)
    # tests()
    # twonorm()
    # main()