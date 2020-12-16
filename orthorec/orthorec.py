import sys
import dxchange
import h5py
import numpy as np
import cupy as cp
from cupyx.scipy.fft import rfft, irfft
import concurrent.futures
import signal
import os
from orthorec import kernels
from orthorec import utils


def backprojection(data, theta, args):
    """Compute backprojection to orthogonal slices"""
    [nz, n] = data.shape[1:]
    obj = cp.zeros([len(args.center), n, 3*n], dtype='float32')
    obj[:, :nz, :n] = kernels.orthox(data, theta, args.center, args.idx)
    obj[:, :nz, n:2*n] = kernels.orthoy(data, theta, args.center, args.idy)
    obj[:, :n, 2*n:3*n] = kernels.orthoz(data, theta, args.center, args.idz)
    return obj


def fbp_filter(data):
    """FBP filtering of projections"""
    t = cp.fft.rfftfreq(data.shape[2])
    wfilter = t * (1 - t * 2)**3  # parzen
    wfilter = cp.tile(wfilter, [data.shape[1], 1])
    # loop over slices to minimize fft memory overhead
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


def fix_inf_nan(data):
    """Fix inf and nan values in projections"""
    data[cp.isnan(data)] = 0
    data[cp.isinf(data)] = 0
    return data


def binning(data, args):
    for k in range(args.bin_level):
        data = 0.5*(data[..., ::2, :]+data[..., 1::2, :])
        data = 0.5*(data[..., :, ::2]+data[..., :, 1::2])
    return data


def gpu_copy(data, theta, start, end, args):
    data_gpu = cp.array(data[start:end]).astype('float32')
    theta_gpu = cp.array(theta[start:end]).astype('float32')
    data_gpu = binning(data_gpu, args.bin_level)
    return data_gpu, theta_gpu


def recon(data, dark, flat, theta, args):
    data = darkflat_correction(data, dark, flat)
    data = minus_log(data)
    data = fix_inf_nan(data)
    data = fbp_filter(data)
    obj = backprojection(data, theta*cp.pi/180.0, args.center, args.idx, args.idy, args.idz)
    return obj


def orthorec(args):

    # projection chunk size to fit data to gpu memory
    # e.g., data size is (1500,2048,2448), args.pchunk=100 gives splitting data into chunks (100,2048,2448)
    # that are processed sequentially by one GPU
    # args.pchunk = 32  # fine for gpus with >=8GB memory
    # change pars wrt binning
    args.idx //= pow(2, args.bin_level)
    args.idy //= pow(2, args.bin_level)
    args.idz //= pow(2, args.bin_level)
    args.center /= pow(2, args.bin_level)

    # init range of args.centers
    args.center = cp.arange(args.center-20, args.center+20, 0.5).astype('float32')

    print('Try args.centers:', args.center)

    # init pointers to dataset in the h5 file
    fid = h5py.File(args.fin, 'r')
    data = fid['exchange/data']
    flat = fid['exchange/data_white']
    dark = fid['exchange/data_dark']
    theta = fid['exchange/theta']
    # compute mean of dark and flat fields on GPU
    dark_gpu = cp.mean(cp.array(dark), axis=0).astype('float32')
    flat_gpu = cp.median(cp.array(flat), axis=0).astype('float32')
    dark_gpu = binning(dark_gpu, args.bin_level)
    flat_gpu = binning(flat_gpu, args.bin_level)
    print('1. Read data from memory')
    utils.tic()
    data = data[:]
    theta = theta[:]
    print('Time:', utils.toc())

    print('2. Reconstruction of orthoslices')
    utils.tic()
    # recover x,y,z orthoslices by projection chunks, merge them in one image
    # reconstruction pipeline consists of 2 threads for processing and for cpu-gpu data transfer
    obj_gpu = cp.zeros([len(args.center), data.shape[2]//pow(2, args.bin_level),
                        3*data.shape[2]//pow(2, args.bin_level)], dtype='float32')
    nchunk = int(cp.ceil(data.shape[0]/args.pchunk))
    data_gpu = [None]*2
    theta_gpu = [None]*2
    with concurrent.futures.ThreadPoolExecutor(2) as executor:
        for k in range(0, nchunk+1):
            # thread for cpu-gpu copy
            if(k < nchunk):
                t2 = executor.submit(
                    gpu_copy, data, theta, k*args.pchunk, min((k+1)*args.pchunk, data.shape[0]), args.bin_level)
            # thread for processing
            if(k > 1):
                t3 = executor.submit(recon, data_gpu[(
                    k-1) % 2], dark_gpu, flat_gpu, theta_gpu[(k-1) % 2], args.center, args.idx, args.idy, args.idz)

            # gather results from 2 threads
            if(k < nchunk):
                data_gpu[k % 2], theta_gpu[k % 2] = t2.result()
            if(k > 1):
                obj_gpu += t3.result()

    obj_gpu /= data.shape[0]
    print('Time:', utils.toc())
    # save result as tiff
    print('3. Cpu-gpu copy and save reconstructed orthoslices')
    utils.tic()
    
    obj = obj_gpu.get()
    recpath = "%s_rec/vn/try_rec/%s/bin%d/" % (os.path.dirname(fin),os.path.basename(fin)[:-3], args.bin_level)
    for i in range(len(args.center)):
        foutc = "%s/r_%.2f" % (recpath,args.center[i])
        dxchange.write_tiff(obj[i], foutc, overwrite=True)
    print('Out files: ', recpath)        
    print('Time:', utils.toc())

    cp._default_memory_pool.free_all_blocks()


def signal_handler(sig, frame):
    """Calls abort_scan when ^C is typed"""
    cp._default_memory_pool.free_all_blocks()
    print('Abort')
    exit()


if __name__ == "__main__":
    """Recover x,y,z ortho slices on GPU
    Parameters
    ----------
    fin : str
        Input h5 file.
    args.center : float
        Rotation args.center
    args.idx,args.idy,args.idz : int
        x,y,z ids of ortho slices
    args.bin_level: int
        binning level

    Example of execution:        
    python orthorec.py /local/data/423_coal5wtNaBr5p.h5 1224 512 512 512 1
    """
    # Set ^C interrupt to abort the scan
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTSTP, signal_handler)

    fin = sys.argv[1]
    args.center = cp.float32(sys.argv[2])
    args.idx = cp.int32(sys.argv[3])
    args.idy = cp.int32(sys.argv[4])
    args.idz = cp.int32(sys.argv[5])
    args.bin_level = cp.int32(sys.argv[6])

    orthorec(fin, args.center, args.idx, args.idy, args.idz, args.bin_level)
