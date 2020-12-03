import sys
import dxchange
import h5py
import numpy as np
import cupy as cp
from cupyx.scipy.fft import rfft, irfft
import kernels


def backprojection(data, theta, center, idx, idy, idz):
    """Compute backprojection to orthogonal slices"""
    [nz, n] = data.shape[1:]
    obj = cp.zeros([n, 3*n], dtype='float32')
    obj[:nz, :n] = kernels.orthox(data, theta, center, idx)
    obj[:nz, n:2*n] = kernels.orthoy(data, theta, center, idy)
    obj[:n, 2*n:3*n] = kernels.orthoz(data, theta, center, idz)
    return obj

def fbp_filter(data):
    """FBP filtering of projections"""
    t = cp.fft.rfftfreq(data.shape[2])
    wfilter = t * (1 - t * 2)**3  # parzen
    wfilter = cp.tile(wfilter, [data.shape[1], 1])
    for k in range(data.shape[0]):
        data[k] = irfft(
            wfilter*rfft(data[k], overwrite_x=True, axis=1), overwrite_x=True, axis=1)
    return data

def darkflat_correction(data, dark, flat):
    """Dark-flat field correction"""
    for k in range(data.shape[0]):
        data[k] = (data[k]-dark)/cp.maximum(flat-dark, 1e-6)
    return data

def minus_log(data):
    """Taking negative logarithm"""
    data = -cp.log(cp.maximum(data, 1e-6))
    return data

def fix_in_nan(data):
    """Fix inf and nan values in projections"""
    data[cp.isnan(data)] = 0
    data[cp.isinf(data)] = 0

def orthorec(fin, fout, center, idx, idy, idz, pchunk):

    # init pointers to dataset in the h5 file
    fid = h5py.File(fin, 'r')    
    data = fid['exchange/data']
    flat = fid['exchange/data_white']
    dark = fid['exchange/data_dark']
    theta = fid['exchange/theta']

    # compute mean of dark and flat fields on GPU
    [ntheta, nz, n] = data.shape
    dark_gpu = cp.mean(cp.array(dark), axis=0).astype('float32')
    flat_gpu = cp.mean(cp.array(flat), axis=0).astype('float32')

    # recover x,y,z orthoslices by projection chunks
    obj_gpu = cp.zeros([n, 3*n], dtype='float32')
    for k in range(int(cp.ceil(ntheta/pchunk))):
        # load data to GPU
        data_gpu = cp.array(
            data[k*pchunk:min((k+1)*pchunk, ntheta)]).astype('float32')
        theta_gpu = cp.array(
            theta[k*pchunk:min((k+1)*pchunk, ntheta)]).astype('float32')*cp.pi/180.0
        # dark-flat field correction, -log, fix inf/nan, parzen filter, backprojection
        data_gpu = darkflat_correction(data_gpu, dark_gpu, flat_gpu)        
        data_gpu = minus_log(data_gpu)
        data_gpu = fbp_filter(data_gpu)
        obj_gpu += backprojection(data_gpu, theta_gpu, center, idx, idy, idz)

    obj_gpu /= ntheta

    # save result as tiff
    dxchange.write_tiff(obj_gpu.get(), fout, overwrite=True)


if __name__ == "__main__":
    """Recover x,y,z ortho slices on GPU
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

    Example of execution:        
    python orthorec.py /local/data/423_coal5wtNaBr5p.h5 /local/data/rec.tiff 1224 512 512 512 100
    """

    fin = sys.argv[1]
    fout = sys.argv[2]
    center = cp.float32(sys.argv[3])
    idx = cp.int32(sys.argv[4])
    idy = cp.int32(sys.argv[5])
    idz = cp.int32(sys.argv[6])
    pchunk = cp.int32(sys.argv[7])

    orthorec(fin, fout, center, idx, idy, idz, pchunk)
