========
ORTHOREC
========


`orthorec <https://github.com/xray-imaging/orthorec>`_ is a Python script that performs a tomograhic reconstruction of x, y,z orthogonal slices on GPU.

Dependencies
------------

cupy, h5py, dxchange

Installation
------------

Install from `Anaconda <https://www.anaconda.com/distribution/>`_ python3.x, then install orthorec::

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
	usage: orthorec recon [-h] [--center CENTER] [--fin FIN] [--fout FOUT] [--idx IDX] [--idy IDY] [--idz IDZ] [--pchunk PCHUNK] [--config FILE] [--logs-home FILE] [--verbose]

	optional arguments:
	  -h, --help        show this help message and exit
          --bin-level       BIN_LEVEL
	  --center CENTER   Output tiff file for 3 merged orthoslices (default: 1024)
	  --fin FIN         Input h5 file (default: None)
	  --fout FOUT       Output tiff file for 3 merged orthoslices (default: None)
	  --idx IDX         x ids of ortho slices (default: 512)
	  --idy IDY         y ids of ortho slices (default: 512)
	  --idz IDZ         z ids of ortho slices (default: 512)
	  --pchunk PCHUNK   Size of a projection chunk (to fit data into GPU memory), e.g., data size is (1500,2048,2448), 
	                    pchunk=32 gives splitting data into chunks (32,2048,2448) that are processed sequentially by a GPU (default: 32)
	  --config FILE     File name of configuration (default: /Users/decarlo/orthorec.conf)
	  --logs-home FILE  Log file directory (default: /Users/decarlo/logs)
	  --verbose         Verbose output (default: True)
