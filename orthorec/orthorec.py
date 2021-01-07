import os
import sys
import dxchange
import h5py
import numpy as np
import cupy as cp
import concurrent.futures

from cupyx.scipy.fft import rfft, irfft

from orthorec import log
from orthorec import utils
from orthorec import kernels


def backprojection(data, theta, args):
    """Compute backprojection to orthogonal slices"""
    [nz, n] = data.shape[1:]
    obj = cp.zeros([len(args.centers), n, 3*n], dtype='float32')
    obj[:, :nz, :n] = kernels.orthox(data, theta, args.centers, args.idx_bin)
    obj[:, :nz, n:2*n] = kernels.orthoy(data, theta, args.centers, args.idy_bin)
    obj[:, :n, 2*n:3*n] = kernels.orthoz(data, theta, args.centers, args.idz_bin)
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
    data_gpu = binning(data_gpu, args)
    return data_gpu, theta_gpu


def recon(data, dark, flat, theta, args):
    data = darkflat_correction(data, dark, flat)
    data = minus_log(data)
    data = fix_inf_nan(data)
    data = fbp_filter(data)
    obj = backprojection(data, theta*cp.pi/180.0, args)
    return obj


def _orthorec(args):
    # projection chunk size to fit data to gpu memory
    # e.g., data size is (1500,2048,2448), args.pchunk=100 gives splitting data into chunks (100,2048,2448)
    # that are processed sequentially by one GPU
    # args.pchunk = 32  # fine for gpus with >=8GB memory
    # change pars wrt binning
    args.idx_bin = args.idx // pow(2, args.bin_level)
    args.idy_bin = args.idy // pow(2, args.bin_level)
    args.idz_bin = args.idz // pow(2, args.bin_level)
    center =  args.center / pow(2, args.bin_level)
    center_search_width = args.center_search_width / pow(2, args.bin_level)
    center_search_step = args.center_search_step / pow(2, args.bin_level)

    log.info('Reconstruct %s' % args.file_name)
    # init range of centers
    args.centers = cp.arange(center-center_search_width, center+center_search_width, center_search_step).astype('float32')
    log.info('   Try centers from  %f to %f in %f pixel' % (center-center_search_width, center+center_search_width, center_search_step))
    if (args.bin_level > 0):
        log.warning('   Center location and search windows are scaled by a binning factor of %d' % (args.bin_level))

    # init pointers to dataset in the h5 file
    fid = h5py.File(args.file_name, 'r')

    try:
        data = fid['exchange/data']
        flat = fid['exchange/data_white']
        dark = fid['exchange/data_dark']
        theta = fid['exchange/theta']
    except KeyError:
        log.error('Corrupted file. Skipping %s' % args.file_name)
        return
    # compute mean of dark and flat fields on GPU
    dark_gpu = cp.mean(cp.array(dark), axis=0).astype('float32')
    flat_gpu = cp.median(cp.array(flat), axis=0).astype('float32')
    dark_gpu = binning(dark_gpu, args)
    flat_gpu = binning(flat_gpu, args)
    utils.tic()
    data = data[:]
    theta = theta[:]
    log.info('   *** Time for reading data from memory: %3.2f s', utils.toc())

    utils.tic()
    # recover x,y,z orthoslices by projection chunks, merge them in one image
    # reconstruction pipeline consists of 2 threads for processing and for cpu-gpu data transfer
    obj_gpu = cp.zeros([len(args.centers), data.shape[2]//pow(2, args.bin_level),
                        3*data.shape[2]//pow(2, args.bin_level)], dtype='float32')
    nchunk = int(cp.ceil(data.shape[0]/args.pchunk))
    data_gpu = [None]*2
    theta_gpu = [None]*2
    with concurrent.futures.ThreadPoolExecutor(2) as executor:
        for k in range(0, nchunk+1):
            # thread for cpu-gpu copy
            if(k < nchunk):
                t2 = executor.submit(
                    gpu_copy, data, theta, k*args.pchunk, min((k+1)*args.pchunk, data.shape[0]), args)
            # thread for processing
            if(k > 1):
                t3 = executor.submit(recon, data_gpu[(
                    k-1) % 2], dark_gpu, flat_gpu, theta_gpu[(k-1) % 2], args)

            # gather results from 2 threads
            if(k < nchunk):
                data_gpu[k % 2], theta_gpu[k % 2] = t2.result()
            if(k > 1):
                obj_gpu += t3.result()

    obj_gpu /= data.shape[0]
    log.info('   *** Time for orthoslice reconstruction: %3.2f s', utils.toc())
    # save result as tiff
    utils.tic()
    
    obj = obj_gpu.get()
    recpath = "%s_rec/3D/try_rec/%s/bin%d/" % (os.path.dirname(args.file_name),os.path.basename(args.file_name)[:-3], args.bin_level)
    for i in range(len(args.centers)):
        label = (args.center - args.center_search_width) + (center_search_step * pow(2, args.bin_level) * i)
        foutc = "%s/r_%.2f" % (recpath, label)
        dxchange.write_tiff(obj[i], foutc, overwrite=True)
    log.info('   *** Time for cpu-gpu copy and save reconstructed orthoslices: %3.2f s', utils.toc())
    log.info('Output files: %s ', recpath)
    cp._default_memory_pool.free_all_blocks()


def orthorec(args):

    fname = args.file_name
    if os.path.isfile(fname):
        _orthorec(args)
        
    elif os.path.isdir(fname):
        # Add a trailing slash if missing
        top = os.path.join(fname, '')

        h5_file_list = list(filter(lambda x: x.endswith(('.h5', '.hdf')), os.listdir(top)))
        h5_file_list.sort()

        log.info("Found: %s" % h5_file_list)       
        for fname in h5_file_list:
            h5fname = top + fname
            args.file_name = h5fname
            _orthorec(args)
    else:
        log.info("Directory or File Name does not exist: %s " % fname)
