def analyze(yRow_yCol_yVal_file): 
# copyright (c) Laura Williams & Russell Fung 2018, UW-Milwaukee
# Updated by Umeshika Dissanayaka, 2026
  from .A_ij_              import A_ij
  from .fit_ramp_          import fit_ramp
  from .plot_              import plot
  from .sigma_of_interest_ import sigma_of_interest
  from .read_h5_ import read_h5
  import numpy as np
  Dsq = read_h5(yRow_yCol_yVal_file,'yVal')
  N = np.max(read_h5(yRow_yCol_yVal_file,'yRow'))
  sigma = sigma_of_interest(Dsq)
  num_sigma = len(sigma)
  # less memory required if for loop is used
  #A = A_ij(Dsq,sigma)
  A = np.zeros(num_sigma)
  for k in range(num_sigma):
    A[k] = A_ij(Dsq,sigma[k])[0] 
  tol = 0.05*np.log(N)
  p = 90
  x = np.log(sigma)  
  # Avoid log(0) replaces zeros with a very small positive value for numerical stability  
  A = np.clip(A, 1e-300, None) 
  # Take logarithm of kernel sum (used in Ferguson analysis)
  y = np.log(A) 
  # Replace -inf values (from extremely small A) with large negative finite numbers
  # This prevents NaNs in later computations (subtraction in ramp detection)  
  y = np.nan_to_num(y, neginf=-1e10) 
  xl,yl,x_mid,y_mid,dimensionality = fit_ramp(x,y,tol,p)
  sigma_opt = np.exp(x_mid)[0] 
  plot(x,y,xl,yl,sigma_opt,dimensionality[0])
  return sigma_opt,dimensionality

