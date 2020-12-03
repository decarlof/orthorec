import sys
import dxchange
import h5py
import numpy as np
import cupy as cp
from cupyx.scipy.fft import rfft, irfft
import concurrent.futures
import kernels
from utils import tic,toc


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

def read_data(data, theta, start, end):    
    return data[start:end],theta[start:end]

def gpu_copy(data,theta):
    return cp.array(data).astype('float32'), cp.array(theta).astype('float32')         

def recon(data, dark, flat, theta, center, idx, idy, idz):
    data = darkflat_correction(data, dark, flat)        
    data = minus_log(data)
    data = fbp_filter(data)
    obj = backprojection(data, theta*cp.pi/180.0, center, idx, idy, idz)
    return obj

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
    time_read = 0 
    time_gpucopy = 0 
    time_proc = 0
    data_part = np.zeros([2,pchunk,nz,n],dtype='uint8')
    data_gpu = cp.zeros([2,pchunk,nz,n],dtype='float32')
    theta_part = np.zeros([2,pchunk],dtype='float32')
    theta_gpu = cp.zeros([2,pchunk],dtype='float32')

    # obj_gpu += recon(data_gpu[0],dark_gpu,flat_gpu,theta_gpu[0],center,idx,idy,idz)        
    nchunk = int(cp.ceil(ntheta/pchunk))
    tic()
    with concurrent.futures.ThreadPoolExecutor(3) as executor:
        for k in range(0,nchunk+2):
            # load data to GPU
            if(k<nchunk):
                t1 = executor.submit(read_data, data,theta,k*pchunk,min((k+1)*pchunk, ntheta))
            # data_part[0],theta_part[0] = read_data(data,theta,k*pchunk,min((k+1)*pchunk, ntheta))          
            if(k>0 and k<nchunk+1): 
                t2 = executor.submit(gpu_copy, data_part[(k-1)%2],theta_part[(k-1)%2])
            # data_gpu[0], theta_gpu[0] = gpu_copy(data_part[0],theta_part[0])
            # dark-flat field correction, -log, fix inf/nan, parzen filter, backprojection
            if(k>1): 
                t3 = executor.submit(recon, data_gpu[(k-1)%2],dark_gpu,flat_gpu,theta_gpu[(k-1)%2],center,idx,idy,idz)
            # obj_gpu += recon(data_gpu[0],dark_gpu,flat_gpu,theta_gpu[0],center,idx,idy,idz)        
            if(k<nchunk):
                data_part[k%2],theta_part[k%2] = t1.result()
            if(k>0 and k<nchunk+1): 
                data_gpu[k%2],theta_gpu[k%2] = t2.result()
            if(k>1):                
                obj_gpu += t3.result()
    obj_gpu /= ntheta
    print('Total:', toc())
    # save result as tiff
    dxchange.write_tiff(obj_gpu.get(), fout, overwrite=True)
    # print('read from memory', time_read)
    # print('copy to gpu', time_gpucopy)
    # print('processing', time_proc)

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
