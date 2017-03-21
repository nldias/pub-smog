2017-02-22T11:03:36

pub-smog is a self-contained set of python modules for the calculation
of smoothed ogives and spectra. It is released under the GPL license.
There are 3 interdependent modules (libraries):

smog.py
nspectrum.py
turbstat.py

and 2 example programs that calculate smoothed spectra and ogives:

uw-ogiv50-biased.py
uw-ogiv50-unbiased.py

which calculate respectively biased and corrected ogives and
spectra. Depending on your python installation, you will need to
change the "she-bang" in the first line of these. I assume you know
how to do it. They are set to use my miniconda
(https://conda.io/miniconda.html) Python installation.

All programs and modules are heavily dependent on numpy. You will need
to install numpy in your system. Look for it at www.numpy.org.

The corresponding output files are

uw-ogiv50-biased.out
uw-ogiv50-unbiased.out

They are overwritten each time the programs are run. Both programs use
as input the data file

20130812-0310.out

with 20-Hz measured turbulence data.

If you know how to use numba (http://numba.pydata.org/) and have it
installed, un-comment all references to numba and to @jit. With numba,
the programs will run considerably faster. The lines referring to
numba and @jit are distributed with comments, and they assume that
numba is not available.

pub-smog is put together in the simplest way: programs and modules
will work if they are in the same directory. For my own use, I put
smog.py, nspectrum.py and turbstat.py in a separate directory listed
in my PYTHONPATH environment variable. As far as I am concerned, there
will never be a system-wide pub-smog installation, nor any type of
more sophisticated Python packaging.

I cannot garantee that the code is correct under any circumstances of
use; use it at your own risk (see the GPL license); and I cannot
provide support for the code. However, I hope this is useful for
research and/or educational purposes.

Nelson Luís Dias
nldias@ufpr.br
