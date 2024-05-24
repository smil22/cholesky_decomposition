import library
import numpy as np
import scipy.linalg as la

n = 3
A = np.random.rand(n,n)
A = np.dot(A.T,A)  #to ensure that the matrix is, at least, symetric

R = library.cholesky_decomposition(A)
E = np.dot(R.T,R) - A
E = la.norm(E)

print('Residual matrix norm: {0:3.4e}'.format(E))