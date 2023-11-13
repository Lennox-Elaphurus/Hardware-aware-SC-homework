import numpy as np

def lu_decomposition(A):
    n = len(A)
    L = np.zeros((n,n))
    U = np.zeros((n,n))

    for i in range(n):
        # U matrix
        for k in range(i, n):
            s = 0
            for j in range(i):
                s += L[i,j] * U[j,k]
            U[i,k] = A[i,k] - s

        # L matrix
        for k in range(i, n):
            if i == k:
                L[k,k] = 1
            else:
                s = 0
                for j in range(i):
                    s += L[k,j] * U[j,i]
                L[k,i] = (A[k,i] - s) / U[i,i]

    return L, U

# Example usage
A = np.array([[2, -1, 0],
              [-1, 2, -1],
              [0, -1, 2]])

L, U = lu_decomposition(A)

print("Lower triangular matrix (L):")
print(L)

print("\nUpper triangular matrix (U):")
print(U)
