import sys
import numpy as np

a = np.array([2,2,2,2,2])
b = np.array([2,21,2,22,2])
a = a[1,:] / b[1]
print(a)