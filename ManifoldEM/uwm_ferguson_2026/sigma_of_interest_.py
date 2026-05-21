def sigma_of_interest(Dsq): 
# copyright (c) Laura Williams & Russell Fung 2018, UW-Milwaukee
# updated April 11, 2022
  import numpy as np
  sigma_small = np.min(np.sqrt(Dsq[np.nonzero(Dsq)]))/3
  log_sigma_small = np.log10(sigma_small)
  rms = np.sqrt(np.mean(Dsq))
  log_rms = np.log10(rms)
  log_sigma_small = np.floor(log_sigma_small-log_rms)+log_rms
  log_sigma_large = 1.0+log_rms
  log_sigma = np.linspace(log_sigma_small,log_sigma_large,num=100,dtype='float64')
  log_sigma = np.append(np.arange(-10,0)+log_sigma_small,log_sigma)
  log_sigma = np.append(log_sigma,np.arange(1,11)+log_sigma_large)
  sigma = np.power(10,log_sigma)
  return sigma

