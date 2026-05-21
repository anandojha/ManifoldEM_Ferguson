def read_h5(filename,variable,h5=True,transpose=False):
# copyright (c) Laura Williams & Russell Fung 2018, UW-Milwaukee
# updated May 11, 2022
  if h5:
    import h5py
    import numpy as np
    f = h5py.File(filename,'r')
    x = np.array(f[variable])
  else:
    from scipy.io import loadmat
    f = loadmat(filename)
    x = f[variable]
  if transpose:
    x = x.T
  return x

