# import cupy as cp

source = """
extern "C" {    
    void __global__ orthox(float *f, float *g, float *theta, float* center, int ix, int n, int nz, int ntheta, int ncenter)
    {
        int ty = blockDim.x * blockIdx.x + threadIdx.x;
        int tz = blockDim.y * blockIdx.y + threadIdx.y;
        if (ty >= n || tz >= nz)
            return;
        float sp = 0;
        int s0 = 0;
        int ind = 0;
        float f0 = 0;
        for (int i=0; i<ncenter; i++)
        {                
            f0 = 0;            
            for (int k = 0; k < ntheta; k++)
            {
                sp = (ix - n / 2) * __cosf(theta[k]) - (ty - n / 2) * __sinf(theta[k]) + center[i]; //polar coordinate
                //linear interpolation
                s0 = roundf(sp);            
                ind = k * n * nz + tz * n + s0;
                if ((s0 >= 0) & (s0 < n - 1))
                    f0 += g[ind] + (g[ind+1] - g[ind]) * (sp - s0) / n; 
            }
            f[i*n*nz + ty + tz * n] = f0;
        }
    }
    void __global__ orthoy(float *f, float *g, float *theta, float* center, int iy, int n, int nz, int ntheta, int ncenter)
    {
        int tx = blockDim.x * blockIdx.x + threadIdx.x;
        int tz = blockDim.y * blockIdx.y + threadIdx.y;
        if (tx >= n  || tz >= nz)
            return;
        float sp = 0;
        float spc = 0;
        int s0 = 0;
        int ind = 0;
        float f0 = 0;
        for (int i=0; i<ncenter; i++)
        {            
            f0 = 0;            
            for (int k = 0; k < ntheta; k++)
            {
                sp = (tx - n / 2) * __cosf(theta[k]) - (iy - n / 2) * __sinf(theta[k]) + center[i]; //polar coordinate
                //linear interpolation
                s0 = roundf(sp);                        
                ind = k * n * nz + tz * n + s0;
                if ((s0 >= 0) & (s0 < n - 1))
                    f0 += g[ind] + (g[ind+1] - g[ind]) * (sp - s0) / n; 
            }
            f[i*n*nz + tx + tz * n] = f0;
        }
    }

    void __global__ orthoz(float *f, float *g, float *theta, float* center, int iz, int n, int nz, int ntheta, int ncenter)
    {
        int tx = blockDim.x * blockIdx.x + threadIdx.x;
        int ty = blockDim.y * blockIdx.y + threadIdx.y;
        if (tx >= n || ty >= n)
            return;
        float sp = 0;
        int s0 = 0;
        int ind = 0;
        float f0 = 0;
        for (int i=0; i<ncenter; i++)
        {            
            f0 = 0;            
            for (int k = 0; k < ntheta; k++)
            {
                sp = (tx - n / 2) * __cosf(theta[k]) - (ty - n / 2) * __sinf(theta[k]) + center[i]; //polar coordinate
                //linear interpolation
                s0 = roundf(sp);
                ind = k * n * nz + iz * n + s0;
                if ((s0 >= 0) & (s0 < n - 1))            
                    f0 += g[ind] + (g[ind+1] - g[ind]) * (sp - s0) / n; 
            }
            f[i*n*n + tx + ty * n] = f0;
        }
    }
}
"""

# module = cp.RawModule(code=source)
# orthox_kernel = module.get_function('orthox')
# orthoy_kernel = module.get_function('orthoy')
# orthoz_kernel = module.get_function('orthoz')

def orthox(data, theta, center, ix):
    """Reconstruct the ortho slice in x-direction on GPU"""
    [ntheta, nz, n] = data.shape
    objx = cp.zeros([len(center), nz, n], dtype='float32')
    orthox_kernel((int(cp.ceil(n/32)), int(cp.ceil(nz/32))), (32, 32),
                  (objx, data, theta, center, ix, n, nz, ntheta, len(center)))
    return objx

def orthoy(data, theta, center, iy):
    """Reconstruct the ortho slice in y-direction on GPU"""    
    [ntheta, nz, n] = data.shape
    objy = cp.zeros([len(center), nz, n], dtype='float32')
    orthoy_kernel((int(cp.ceil(n/32)), int(cp.ceil(nz/32))), (32, 32),
                  (objy, data, theta, center, iy, n, nz, ntheta, len(center)))
    return objy

def orthoz(data, theta, center, iz):
    """Reconstruct the ortho slice in z-direction on GPU"""        
    [ntheta, nz, n] = data.shape
    objz = cp.zeros([len(center), n, n], dtype='float32')
    orthoz_kernel((int(cp.ceil(n/32)), int(cp.ceil(n/32))), (32, 32),
                  (objz, data, theta, center, iz, n, nz, ntheta, len(center)))
    return objz
