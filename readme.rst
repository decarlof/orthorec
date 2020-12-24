========
ORTHOREC
========


`orthorec <https://github.com/xray-imaging/orthorec>`_ is a Python script that performs a tomograhic reconstruction of x, y, z orthogonal slices on GPU.

Dependencies
------------

cupy, h5py, dxchange

Installation
------------

First, you must have `Conda <https://docs.conda.io/en/latest/miniconda.html>`_
installed.

Next, create a new Conda environment called ``orthorec`` by running::

    $ conda create --name orthorec

and activate it::

    $ conda activate orthorec

then install the orthorec dependecies (cupy, h5py, dxchange) and orthorec::

    $ git clone https://github.com/xray-imaging/orthorec.git
    $ cd orthorec
    $ python setup.py install


Usage
-----

::

    $ orthorec -h
    
    usage: orthorec [-h] [--config FILE]  ...

    optional arguments:
      -h, --help     show this help message and exit
      --config FILE  File name of configuration

    Commands:
      
        init         Create configuration file
        show         Show status
        recon        Start ortorec


    $ orthorec init
        Creates a orthorec.conf default file

    $ orthorec show 
        Show the last used orthorec parameters

    $ orthorec recon -h
    usage: orthorec recon [-h] [--bin-level BIN_LEVEL] [--center CENTER] [--center-search-step CENTER_SEARCH_STEP]
                          [--center-search-width CENTER_SEARCH_WIDTH] [--fin FIN] [--idx IDX] [--idy IDY] [--idz IDZ]
                          [--pchunk PCHUNK] [--config FILE] [--logs-home FILE] [--verbose]


    optional arguments:
      -h, --help            show this help message and exit
      --bin-level BIN_LEVEL
                            binning level (default: 2)
      --center CENTER       Output tiff file for 3 merged orthoslices (default: 1024)
      --center-search-step CENTER_SEARCH_STEP
                            Center search step size (pixel) (default: 0.5)
      --center-search-width CENTER_SEARCH_WIDTH
                            +/- center search width (pixel) (default: 20.0)
      --fin FIN             Input hdf5 file (default: )
      --idx IDX             x ids of ortho slices (default: 1024)
      --idy IDY             y ids of ortho slices (default: 1024)
      --idz IDZ             z ids of ortho slices (default: 1024)
      --pchunk PCHUNK       Size of a projection chunk (to fit data into GPU memory), e.g., data size is (1500,2048,2448),
                            pchunk=100 gives splitting data into chunks (100,2048,2448) that are processed sequentially by a GPU
                            (default: 32)
      --config FILE         File name of configuration (default: /home/beams/TOMO/orthorec.conf)
      --logs-home FILE      Log file directory (default: /home/beams/TOMO/logs)
      --verbose             Verbose output (default: True)