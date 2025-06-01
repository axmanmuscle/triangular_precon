## testing VMPPA from Parente et al.
import numpy as np

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

if __name__ == "__main__":
    main()