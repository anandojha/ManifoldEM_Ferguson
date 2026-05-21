def A_ij(Dsq,sigma):
# copyright (c) Laura Williams & Russell Fung 2018, UW-Milwaukee
  import numpy as np
# make Dsq (sigma) a column (row) vector
# sum_kernel is calculated in a "bsxfun" way, and is flattened before returning
  Dsq = np.reshape(Dsq,[-1,1])
  sigma = np.reshape(sigma,[1,-1])
  sum_kernel = np.sum(np.exp(-Dsq/sigma**2),axis=0)
  sum_kernel = sum_kernel.flatten()
  return sum_kernel

