# orthorec
Reconstruction of x,y,z orthogonal slices on GPU

# Dependencies
cupy, h5py, dxchange

# Execution
```
python orthorec.py fin center idx idy idz bin_level
```

```
Parameters
----------
fin : str
    Input h5 file
center : float
    Rotation center
idx,idy,idz : int
    x,y,z ids of ortho slices
bin_level: int
    binning level
    
```

Example of execution:        

```
python orthorec.py /local/data/423_coal5wtNaBr5p.h5 1224 512 512 512 1
```

