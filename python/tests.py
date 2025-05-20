from cvx_test import apply_weighted_prox
import numpy as np

def main():

  proxl1 = lambda x, t: np.maximum(np.abs(x) - t, 0*x)*np.sign(x)

  V = np.array([ [ 0.288728024031355, 0, 0],
                 [ -0.000713876520823, 0.247248781346549, 0], 
                 [ -0.000512356896019,  -0.023268148554025,   0.200721271692160]])
  
  x0 = np.array([[-12.980275303824813],
  [23.623320839310633],
  [-15.169065945673832]])

  # print(V@x0)

  y = apply_weighted_prox(x0, V, proxl1)
  print(y)
  return 0

if __name__ == "__main__":
  main()