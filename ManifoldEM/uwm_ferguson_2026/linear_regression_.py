def linear_regression(x,y):
# 
# copyright (c) Laura Williams & Russell Fung 2018, UW-Milwaukee
  import numpy as np
  x = x.flatten()
  y = y.flatten()
  n = len(x)
  sum_xx = np.sum(np.square(x))
  sum_x = np.sum(x)
  sum_xy = np.sum(x*y)
  sum_y = np.sum(y)
  A = np.empty((2,2))
  A[0][0] = sum_xx
  A[0][1] = sum_x
  A[1][0] = sum_x
  A[1][1] = n
  b = np.empty((2,1))
  b[0][0] = sum_xy
  b[1][0] = sum_y
  m,c = np.linalg.solve(A,b)
  return m,c

