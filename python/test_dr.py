import numpy as np
from utils import apply_weighted_prox, proxL1, prox_twonormsq, proxl1_element

def douglasRachford(x0, proxf, proxg, t, maxiter=15, eps=1e-4, obj=None):
    """
    basic version of Douglas Rachford splitting to solve
    min  f(x) + g(x)
     x
    """
    printo = False
    if obj is not None:
       printo = True
    

    yi = np.zeros(x0.shape)


    for iter in range(maxiter):
        xi = proxf(yi, t)
        wi = proxg(2*xi - yi, t)
        yi = yi + wi - xi

        if printo:
           xobj = proxf(yi, t)
           print(f'iteration {iter} objval {obj(xobj)}')

        if np.linalg.norm(wi - xi) < eps:
            break
    
    return proxf(yi, t)

def test_dr():
  """
  LASSO problem, let's minimize
  f(x) = 0.5 * || Ax - b ||_2^2 + mu*||x||_1
  """
  rng = np.random.default_rng(20250609)
  n = 10

  A = rng.normal(0, 1, (n, n))
  x = rng.normal(0, 1, (n, 1))

  b = A@x

  # mu = 0.1
  mu = 1

  f = lambda x_in: 0.5 * np.linalg.norm(A@x_in - b)**2
  g = lambda x_in: mu * np.linalg.norm(x_in, 1)

  obj = lambda x_in: f(x_in) + g(x_in)

  proxf = lambda x_in, t: prox_twonormsq(x_in, t, A, b)
  proxg = lambda x_in, t: proxL1(x_in, t, mu)

  x0 = np.zeros((n, 1))
  out = douglasRachford(x0, proxf, proxg, 2, obj=obj)

  print(f'after DR, obj value: {obj(out)}')
  print(f'two norm: {f(out)}')
  print(f'one norm: {g(out)}')
  print(f'norm diff from x: {np.linalg.norm(out - x)}')
  print(f'out: {out}')

  ## okay now just make DR use the weighted prox with some matrix
  V = 0.5*np.eye(n)
  proxg_element = lambda x_in, t: proxl1_element(x_in, mu*t)
  weighted_proxf = lambda x_in, t: apply_weighted_prox(x_in, V, proxf)
  weighted_proxg = lambda x_in, t: apply_weighted_prox(x_in, V, proxg_element)

  test_proxs2(proxg, weighted_proxg, n)

  out_weight = douglasRachford(x0, proxf, weighted_proxg, 2, obj=obj)

  print(f'after weighted DR, obj value: {obj(out_weight)}')
  print(f'two norm: {f(out_weight)}')
  print(f'one norm: {g(out_weight)}')
  print(f'norm diff from x: {np.linalg.norm(out_weight - x)}')
  print(f'out: {out_weight}')

  print('*'*25)

  print(f'difference between DR and weighted DR: {np.linalg.norm(out_weight - out)}')

  return 0

def test_proxs():
    rng = np.random.default_rng(20250609)
    n = 10
    sigma = 0.5
    t=2

    x = rng.normal(5, 10, (n,1))

    pf1 = proxL1(x, sigma, t)

    pf2 = np.zeros_like(x)
    for i in range(n):
       pf2[i] = proxl1_element(x[i].item(), t*sigma)

    print(f'norm diff pf1 - pf2 {np.linalg.norm(pf1 - pf2)}')

    V = ( 1 / (sigma*t) ) * np.eye(n)
    pf3 = apply_weighted_prox(x, V, proxl1_element)
    print(f'norm diff pf2 - pf3 {np.linalg.norm(pf2 - pf3)}')

def test_proxs2(prox1, prox2, n):
    rng = np.random.default_rng(20250609)

    x = rng.normal(5, 10, (n,1))

    sigma = 0.2
    test1 = prox1(x, sigma)
    test2 = prox2(x, sigma)

    error = np.linalg.norm(test1 - test2)

if __name__ == "__main__":
  # test_proxs()
  test_dr()