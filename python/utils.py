## let's start organizing this better
# this is a utils file

import numpy as np

def prox_twonormsq(x, lambd, A, b):
    """
    this is prox_lambda f(x)
    where f = 0.5 * || Ax - b ||_2^2
    """
    n = x.size

    y = A.T @ b
    G = np.linalg.inv(A.T@A + 1/lambd * np.eye(n))

    out = G @ (y + x / lambd)
    return out

def proxl1_element(x, t):
    return np.max([np.abs(x) - t, 0]) * np.sign(x)

def proxL1(x, sigma, t = 1, b = None):
    """
    computes 
    prox_(sigma f) (x)
    where f(x) = t * || x ||_1

    INPUTS:
        x - nx1 numpy array representing the input
        sigma - scaling for proximal operator
        t - scaling for the function (if needed), 1 by default
    """

    xshape = x.shape
    if len(xshape) > 1:
        assert xshape[1] == 1, 'gave proxL1 a two dimensional (or larger) array'
    if b is None:
        b = np.zeros(xshape)

    magX = np.abs(x - b)
    thresh = sigma * t

    scalingFactors = thresh / magX
    out = np.zeros(xshape)

    nonzeroIndices = magX > thresh

    out[nonzeroIndices] = x[nonzeroIndices] * (1 - scalingFactors[nonzeroIndices])

    out = out + b

    return out

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
        tmp1 = np.dot(V[i, 0:i-1], y[0:i-1, 0])
        vii = V[i, i]
        x_in = vx[i, 0] - tmp1
        y[i, 0] = prox(x_in/vii, t/vii)

    return y

if __name__ == "__main__":
    print('utils.py not meant to be run standalone.')
