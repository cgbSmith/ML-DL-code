import numpy as np
import scipy.sparse as sp

a = np.array([[1, 2, 3], [2, 3, 4], [1, 0, 0]])
print(a)
b = sp.csr_matrix(a)
print(b)
