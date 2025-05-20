import numpy as np
import cvxpy as cvx

def lasso(A, b, lam):
    """
    let's just solve a lasso problem first lol
    """

    n = A.shape[0]

    x = cvx.Variable((n, 1))

    def obj(xi, A, b, lam):
        return 0.5*cvx.norm2(A@xi - b)**2 + lam*cvx.norm1(xi)
    
    prob = cvx.Problem(cvx.Minimize(obj(x, A, b, lam)))

    prob.solve()

    print(f'the optimal value is || Ax - b ||_2 + lambda || x ||_1 = {prob.value}')
    print(f'with x = {x.value}')

    return x.value

def forward_backward(A, b, lam, alpha, max_iter = 100):
    """
    here we'll solve lasso using forward backward splitting
    also known as proximal gradient descent
    since we want 0 in grad f(x) + del g(x)
    we take a gradient descent step of f and then a proximal step of g
    """

    x0 = np.zeros((A.shape[1], 1))

    def prox(x, t):
        return np.maximum(np.abs(x) - t, 0*x)*np.sign(x)

    xi = x0
    for _ in range(max_iter):
        grad_step = xi - alpha * A.T @ (A@xi - b)
        prox_step = prox(grad_step, alpha)

        xi = prox_step

    obj_val = 0.5*np.linalg.norm(A@xi - b)**2 + lam*np.linalg.norm(xi, 1)
    print(f'after {max_iter} steps we got to an obj value of {obj_val}')
    print(f'with an x val of {xi}')

    return xi

def weighted_forward_backward(A, b, lam, V, max_iter = 100):
    """
    here we'll solve lasso using our new weighted method
    so the iteration will be 
    x_{k+1} = (I + V^{-1}del g)^{-1}(I - V^{-1}grad f)x_k
    """

    x0 = np.zeros((A.shape[1], 1))

    def prox(x, t):
        return np.maximum(np.abs(x) - t, 0*x)*np.sign(x)
    
    Vinv = np.linalg.inv(V)

    xi = x0
    for _ in range(max_iter):
        grad_step = xi - Vinv @ A.T @ (A@xi - b)
        prox_step = apply_weighted_prox(grad_step, V, prox)

        xi = prox_step

    obj_val = 0.5*np.linalg.norm(A@xi - b)**2 + lam*np.linalg.norm(xi, 1)
    print(f'after {max_iter} steps we got to an obj value of {obj_val}')
    print(f'with an x val of {xi}')

    return xi

def apply_weighted_prox(x, V, proxf):
    """
    This is going to have a lower triangular matrix V and a diagonal matrix of the prox operators
    and solve the system using forward substitution
    """
    t = 1
    n = V.shape[0]
    if V.shape[1] != n:
        print('V must be square')
        return
    
    # vx = V@x
    vx = x
    y = np.zeros((n, 1))
    
    v00 = V[0, 0]
    # y[0, 0] = proxf(vx[0, 0]/v00, t/v00)
    y[0, 0] = proxf(vx[0, 0], t/v00)
    for i in range(1, n):
        tmp1 = np.dot(V[i, 0:i], y[0:i, 0])
        vii = V[i, i]
        x_in = vx[i, 0] - tmp1
        # y[i, 0] = proxf(x_in/vii, t/vii)
        y[i, 0] = proxf(x_in, t/vii)

    return y

def find_trace_const(H):
    n = H.shape[0]
    L = cvx.Variable((n, n))

    c1 = cvx.upper_tri(L) == 0
    c2 = cvx.norm(2*L@H - np.eye(n, n)) <= 0.9999
    c3 = L >= 0

    # use this to find a good tau value
    constraints = [c1, c2, c3]
    problem = cvx.Problem(cvx.Minimize(-1*cvx.trace(L)), constraints)

    problem.solve()

    print(f'the optimal value is -tr( L ) = {problem.value}')

    return -1*problem.value
    # print(f'with L = {L.value}')

def main():
    # first setup the problem

    np.random.seed(20240504)
    n = 5
    A = np.random.randn(n, n)

    x_init = np.random.randn(n, 1)

    b = A @ x_init

    H = A.T @ A
    beta = np.linalg.norm(H)
    print(f'spec norm of A^TA is {beta}')

    lam = 1
    xstar_cvx = lasso(A, b, lam)
    xstar_fb = forward_backward(A, b, lam, 1/(beta + 1), 100)
    xstar_weighted = weighted_forward_backward(A, b, lam, (beta+1)*np.eye(*H.shape), 100)

    obj_val_cvx = 0.5*np.linalg.norm(A@xstar_cvx - b)**2 + lam*np.linalg.norm(xstar_cvx, 1)
    obj_val_fb = 0.5*np.linalg.norm(A@xstar_fb - b)**2 + lam*np.linalg.norm(xstar_fb, 1)
    obj_val_weight = 0.5*np.linalg.norm(A@xstar_weighted - b)**2 + lam*np.linalg.norm(xstar_weighted, 1)

    print(f'obj value from cvx: {obj_val_cvx}')
    print(f'obj value from fb: {obj_val_fb}')
    print(f'obj value from weighted: {obj_val_weight}')

    print(f'norm diff: {np.linalg.norm(xstar_cvx - xstar_fb)}')

    """
    problem we're solving is
    
    min || Ax - b ||_2^2 + lambda || x ||_1
     x

    Hessian of this problem is H = A^TA

    want to 
    min || LH || (spectral norm)
     L
    
     st L is lower triangular (so this is something like triu(L, 1) == 0, or upper_trianglar(L).value == zeros)
     tr(L) > tau for some tau so L doesnt go to zero
     and || 2LH - I || < 0.9999 so it's firmly nonexpansive

     This gives us L = V^-1
     to solve the problem then we'll use forward-backward splitting and this idea for the prox step


     to find a good value for tau try doing this but maximize the trace (minimize -tr(L))
    """

    def obj(L, H):
        return cvx.norm2(L @ H)
    
    tau = find_trace_const(H) * 0.99
    L = cvx.Variable((n, n))

    c1 = cvx.upper_tri(L) == 0
    c2 = cvx.trace(L) >= tau
    c3 = cvx.norm(2*L@H - np.eye(n, n)) <= 0.9999
    c4 = L >= 0

    constraints = [c1, c2, c3, c4]
    problem = cvx.Problem(cvx.Minimize(obj(L, H)), constraints)

    # use this to find a good tau value
    # constraints = [c1, c3, c4]
    # problem = cvx.Problem(cvx.Minimize(-1*cvx.trace(L)), constraints)

    problem.solve()

    print(f'the optimal value is || L H || = {problem.value}')
    print(f'with L = {L.value}')

    # xstar_weighted_new = weighted_forward_backward(A, b, lam, L.value @ H, 2)
    xstar_weighted_new = weighted_forward_backward(A, b, lam, L.value, 2)
    obj_val_weight_new = 0.5*np.linalg.norm(A@xstar_weighted_new - b)**2 + lam*np.linalg.norm(xstar_weighted_new, 1)

    print(f'obj value from weighted with found mx: {obj_val_weight_new}')

    # print(f'upper tri: {cvx.upper_tri(L).value}')

    # lh = L.value @ H

    # xtest = np.random.randn(n, 1)
    # ytest = np.random.randn(n, 1)

    # lhs = np.linalg.norm(lh @ xtest - lh @ ytest)
    # rhs = np.linalg.norm(xtest - ytest)

    # print(f'if non expansive then lhs: {lhs} should be <= {rhs} : rhs')


if __name__ == "__main__":
    main()