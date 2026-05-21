def report_runtime(job_id,t0,t1,cr=0): 
# copyright (c) Russell Fung 2019, UW-Milwaukee
# updated April 26, 2022
  message = job_id+' Elapsed time is %.6f seconds.'%(t1-t0)
  for jj in range(cr):
    print(' ')
  print(message)
  for jj in range(cr):
    print(' ')

