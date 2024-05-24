import numpy as np

def cholesky_decomposition(A):  #cholesky1
    """
    A must be a symetric positive-definite matrix.
    """
    n = A.shape[0]
    R = np.zeros(A.shape)
    R[0,0] = np.sqrt(A[0,0])
    R[0,1:n] = A[0,1:n]/R[0,0]
    for i in range(1,n):
        R[i,i] = np.sqrt(A[i,i] - np.sum([R[k,i]**2 for k in range(i)]))
        for j in range(i+1,n):
            R[i,j] = (A[i,j]-np.sum([R[k,i]*R[k,j] for k in range(i)]))/R[i,i]
    return R
