def write_h5(filename,value,variable): 
# copyright (c) Russell Fung 2018, UW-Milwaukee
  import h5py
  with h5py.File(filename,'a') as f:
    try:
      del f[variable]
    except:
      pass
    f.create_dataset(variable,data=value)

