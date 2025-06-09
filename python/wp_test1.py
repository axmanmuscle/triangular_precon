"""
we want to test the idea about this weighted prox (at least that's what im calling it)
The prox of a function f can be written as the resolvent of its subdifferential
prox_tf(x) = (I + t del(f))^-1 x

Let V be a positive definite lower triangular matrix. Then we want to investigate the convergence of
(V + del(f))^-1
"""

import numpy as np
import scipy.io as sio

def f1(x): 
    return np.abs(x)

def apply_weighted_prox(x, V, prox):
    """
    This is going to have a lower triangular matrix V and a diagonal matrix of the prox operators
    and solve the system using forward substitution
    """
    t = 1
    n = V.shape[0]
    if V.shape[1] != n:
        print('V must be square')
        return
    
    vx = V@x
    y = np.zeros((n, 1))
    
    v00 = V[0, 0]
    y[0, 0] = prox(vx[0, 0] / v00, t/v00)
    for i in range(1, n):
        tmp1 = np.dot(V[i, 0:i], y[0:i, 0])
        vii = V[i, i]
        x_in = vx[i, 0] - tmp1
        y[i, 0] = prox(x_in/vii, t/vii)

    return y


def proxf(x, t):
    return np.max([np.abs(x) - t, 0]) * np.sign(x)

def prox_point(x0, prox, nmax = 50, tol = 1e-4):

    t = 0.5
    xi = x0
    for i in range(nmax):
        xip1 = np.zeros(xi.shape)
        for j in range(xi.shape[0]):
            xip1[j, 0] = prox(xi[j, 0], t)
        if np.linalg.norm(xip1 - xi) < tol:
            print(f'prox converged after {i} iterations')
            print(xip1)
            return xip1
        
        # print(xip1)
        xi = xip1
    print(xi)
    return xi

def weighted_prox_point(x0, prox, nmax = 50, tol = 1e-4):

    xi = x0
    V = np.array([[0.1, 0], [0.001, 0.1]])
    # V = np.eye(2, 2)
    for i in range(nmax):
        xip1 = prox(xi, V)
        if np.linalg.norm(xip1 - xi) < tol:
            print(f'weighted prox converged after {i} iterations')
            print(xip1)
            return xip1
        
        # print(xip1)
        xi = xip1
    print(xi)
    return xi

def main():
    x1 = 5
    x2 = -10

    nmax = 1000
    x0 = np.array([[x1], [x2]])

    xstar = prox_point(x0, proxf, nmax)
    print('*'*15)

    wp = lambda x, V: apply_weighted_prox(x, V, proxf)

    xstar2 = weighted_prox_point(x0, wp, nmax)

    return 0

def test_matlab():
    data = sio.loadmat('/mnt/e/triangular_precon/matlab/prox_test.mat')
    V = data['V']
    x0 = data['x0']
    print(f'x0: {x0}')

    print(f'Vx0: {V@x0}')

    test = apply_weighted_prox(x0, V, proxf)
    x = data['x']
    print(f'test: {test}')
    print(f'x: {x}')
    print(f'norm diff {np.linalg.norm(x - test)}')

if __name__ == "__main__":
    test_matlab()
    # main()