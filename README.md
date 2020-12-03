# orthorec
Reconstruction of x,y,z orthogonal slices on GPU

```
python orthorec.py fin fout center idx idy idz pchunk
```

```
Parameters
----------
fin : str
    Input h5 file.
fout : str
    Output tiff file for 3 merged orthoslices
center : float
    Rotation center
idx,idy,idz : int
    x,y,z ids of ortho slices.
pchunk : int
    Size of a projection chunk (to fit data into GPU memory), 
    e.g., data size is (1500,2048,2448), pchunk=100 gives splitting data into chunks (100,2048,2448)
    that are processed sequentially by a GPU        
```
Example of execution:        

```
python orthorec.py /local/data/423_coal5wtNaBr5p.h5 /local/data/rec.tiff 1224 512 512 512 100
```
