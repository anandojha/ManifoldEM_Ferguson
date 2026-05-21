def fit_ramp(x,y,tol,p): 
# copyright (c) Laura Williams & Russell Fung 2018, UW-Milwaukee
# updated April 11, 2022
  from .linear_regression_ import linear_regression
  import numpy as np
  y_min = np.min(y)
  y_max = np.max(y)
  # ramp is where y deviates from the plateaus more than tol
  ramp_portion = np.intersect1d(np.where(y-y_min>tol)[0],
                                np.where(y_max-y>tol)[0])
  ramp_y0 = np.min(y[ramp_portion])
  ramp_y1 = np.max(y[ramp_portion])
  ramp_height = ramp_y1-ramp_y0
  # the center p% of the ramp is assumed to be linear
  offset = 0.5*(1-0.01*p)*ramp_height
  linear_y0 = ramp_y0+offset
  linear_y1 = ramp_y1-offset
  y_mid = 0.5*(linear_y0+linear_y1)
  linear_portion = np.intersect1d(np.where(y>linear_y0)[0],
                                  np.where(y<linear_y1)[0])
  xl = x[linear_portion]
  yl = y[linear_portion]
  slope,y_intercept = linear_regression(xl,yl)
  yl = slope*xl+y_intercept
  x_mid = (y_mid-y_intercept)/slope
  return xl,yl,x_mid,y_mid,slope

