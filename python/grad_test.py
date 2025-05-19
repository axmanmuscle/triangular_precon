import numpy as np 

def main():

    n =  10
    np.random.seed(2024052)
    A = np.random.randn(n, n)
    x = np.random.randn(n, 1)

    b = A@x

    x0 = np.random.randn(n, 1)

    init_error = np.linalg.norm(A@x0 - b)
    print(f'initial error: {init_error}')

    grad = lambda x: A.T @ (A@x - b)

    H = A.T @ A

    evals, evecs = np.linalg.eig(H)
    print(f'output of eig: {evals}')

    lambda1 = np.max(evals)
    print(f'max eigenvalue of A^TA: {lambda1}')
    print(f'spectral norm of A: {np.linalg.norm(A, 2)}')

    alpha = 1/lambda1 # step size of 2/spectral norm of A

    x1 = x0 - alpha*grad(x0)
    x2 = x1 - alpha*grad(x1)


    print(f'error after one step: {np.linalg.norm(A@x1 - b)}')
    print(f'error after two steps: {np.linalg.norm(A@x2 - b)}')

    vinv = np.random.uniform(0.01, 5, (n, n))
    vinv = np.tril(vinv)

    # v = np.linalg.inv(vinv)
    v = vinv
    v = v * alpha

    x1v = x0 - v @ grad(x0)
    x2v = x1v - v @ grad(x1v)

    print(f'error after one step (V): {np.linalg.norm(A@x1v - b)}')
    print(f'error after two steps (V): {np.linalg.norm(A@x2v - b)}')

    print(f'condition number of v: {np.linalg.cond(v)}')

    return 0

if __name__ == "__main__":
    main()