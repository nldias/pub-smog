#!/home/nldias/miniconda3/bin/python3
# --------------------------------------------------------------------
# License
#
# Copyright Nelson Luís da Costa Dias, 2017
#
# This file is part of pub-smog.
# pub-smog is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# pub-smog is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with pub-smog.  If not, see <http://www.gnu.org/licenses/>.
# --------------------------------------------------------------------
from numpy import zeros, arctan2
from numpy import sqrt as asqrt
from numpy.fft import fft
#from numba import jit
from math import pi, cos, sqrt
I = complex(0.0,1.0)



#it@jit
def cross_density(
deltat,          # sampling acquisition interval
z                # z[k] = x[k] + i y[k]   numpy array
) :
   ''' 
   ------------------------------------------------------------------------- -->
   cross_density: calculates a cross-spectrum according to Bendat and Piersol
   (1986), Random Data -- Analysis and Measurement Procedures, 2nd Ed. John
   Wiley-Interscience, chapters 5 and 11. *With Alias Reducing (ra)*
        
      input must be in the form of a complex array z[] whose members are
 
      z[k] = x[k] + i y[k]
 
      where x and y are the original blocks of real data from the two distinct
      time series under analysis.  This corresponds to step 3 in section 11.6.3
      of Bendat & Piersol, and must be done OUTSIDE this function.
 
      2011-03-14T16:17:10 NOTE that G has (n/2) and that the density at zero
                          freq is no longer returned!!!!
 
      ALSO NOTE that no tapering is being done here
 
      AS A BONUS, this function returns the calculated spectra of the
      time series x and y in gxx and gyy.
   -----------------------------------------------------------------------------
   '''
   n = len(z)
   m = n//2
   den = float(n)
   minusI = -I   # 1/i == -i                                             
# ---------------------------------------------------------------------------------------------
# allocates memory for x and y
# ---------------------------------------------------------------------------------------------
   x = zeros(m+1,complex)
   y = zeros(m+1,complex)
# ---------------------------------------------------------------------------------------------
# allocates memory for the return variables
# ---------------------------------------------------------------------------------------------
   Coxy = zeros(m+1,float)   # one-sided cospectrum   numpy array
   Quxy = zeros(m+1,float)   # one-sided quadspectrum numpy array
   Gxx = zeros(m+1,float)    # spectrum of x          numpy array                          
   Gyy = zeros(m+1,float)    # spectrum of y          numpy array
# ---------------------------------------------------------------------------------------------
# step 4: compute the FFT of z
# ---------------------------------------------------------------------------------------------
   z = fft(z)
# ---------------------------------------------------------------------------------------------
# step 5: with FFT of z, obtain FFT's of original time series via eqn 11.72 of Bendat & Piersol
#
# note however that k cannot be == 0, and that there is a typo in the eq.
# for x, which should read the conjugate of z at n MINUS k !
# ---------------------------------------------------------------------------------------------
   x[0] = z[0].real
   y[0] = z[0].imag
   for k in range(1,m+1): 
      zk = z[k]
      zl = z[n-k].conjugate()
      x[k] = 0.5*(zk + zl )
      y[k] = 0.5*minusI*(zk - zl )
   pass
# ------------------------------------------------------------------------
# step 7: one-sided spectra and cross-spectrum
# ------------------------------------------------------------------------ 
   for k in range(m+1):
# ------------------------------------------------------------------------------------------------
# reduce alias: the cross-spectrum receives the same treatment as the spectrum
# see: paper by Gobbi, Chamecki and Dias, in WRR
# see: deltaf05
# see: deltaf06
# 20040702: alias reduction is done here, to take advantage of the k-loop
# 2016-09-27T16:13:34 no more anti-aliasing, for now...
# ------------------------------------------------------------------------------------------------
#      ra = (1.0 + cos(k*pi/n))/2.0     # anti-aliasing term
      ra = 1.0
      xx = x[k]*ra
      xc = xx.conjugate()
      yy = y[k]*ra
      gg = xc*yy
      gg = gg*deltat*2.0/den
# ------------------------------------------------------------------------------------------------
# Now calculates the 4 output arrays
# ------------------------------------------------------------------------------------------------
      Coxy[k] = gg.real
# ------------------------------------------------------------------------------------------------
# 20040707: this minus sign for Qu complies with Bendat and Piersol's convention
# ------------------------------------------------------------------------------------------------
      Quxy[k] = -gg.imag
      Gxx[k] = 2.0 * deltat * abs(xx)**2  / den
      Gyy[k] = 2.0 * deltat * abs(yy)**2  / den
   pass
   return (Gxx,Gyy,Coxy,Quxy)


def docross(
nblocks,	    # the number of blocks
length,             # the length of one block
deltat,	            # deltat
x,		    # x-data (real, numpy array)
y                   # y-data (real, numpy array)
) :
# ------------------------------------------------------------------------------
# --> docross: does the whole job of estimating cross-spectra
#     
# returns:
# 
#    Nyquist        -- the Nyquist frequency
#    nn		    -- the frequencies
#    Sxxmed         -- one spectrum
#    sxd	    -- its uncertainty
#    Syymed	    -- other spectrum
#    syd	    -- its uncertainty
#    Coxymed        -- the co-spectrum
#    cxyd	    -- its uncertainty
#    Quxymed	    -- the quad-spectrum
#    qxyd	    -- its uncertainty
#    Thetamed       -- the average phase
#    thetaxyd 	    -- its uncertainty
# -------------------------------------------------------------------------------
   assert isinstance(nblocks,int)       # it is better to be careful
   assert isinstance(length,int)        # ditto 
   assert isinstance(deltat,float)      # ditto
   npoints = nblocks*length             # total number of points 
   assert ( length % 2 == 0 )           # makes sure that length is divisble by 2
   half_length = length // 2            # length//2
# -------------------------------------------------------------------------------
# allocates storage for the spectral densities of each block
# -------------------------------------------------------------------------------
   Sxx   = zeros(half_length,float)
   Syy   = zeros(half_length,float)
   Coxy  = zeros(half_length,float)
   Quxy  = zeros(half_length,float)
   theta = zeros(half_length,float)
# -------------------------------------------------------------------------------
# puts the data in a complex array
# -------------------------------------------------------------------------------
   zxy = x + I*y
# -------------------------------------------------------------------------------
# fills avgs and stdevs with with 0's
# -------------------------------------------------------------------------------
   Sxxmed = zeros(half_length,float)
   Syymed = zeros(half_length,float)
   Coxymed= zeros(half_length,float)
   Quxymed= zeros(half_length,float)
   Thetamed=zeros(half_length,float)
   sxd=     zeros(half_length,float)
   syd=     zeros(half_length,float)
   cxyd=    zeros(half_length,float)
   qxyd=    zeros(half_length,float)
   thetaxyd=zeros(half_length,float)
# ------------------------------------------------------------------------------
# loop over blocks
# ------------------------------------------------------------------------------
   for bl in range(nblocks):
# ------------------------------------------------------------------------------
# copies this block to zzz, whose content will be replaced by its FFT
# ------------------------------------------------------------------------------
      zzz = zxy[length*bl:length*(bl+1)]
# ------------------------------------------------------------------------------
# this spectrum for this block
# ------------------------------------------------------------------------------
      (Sxx,Syy,Coxy,Quxy) = cross_density(deltat,zzz)
# ------------------------------------------------------------------------------
# discard the zero-index
# ------------------------------------------------------------------------------
      Coxy = Coxy[1:half_length+1]
      Quxy = Quxy[1:half_length+1]
      Sxx  = Sxx[1:half_length+1]
      Syy  = Syy[1:half_length+1]
# ------------------------------------------------------------------------------
# accumulate statistics
# ------------------------------------------------------------------------------
      Sxxmed += Sxx; sxd += Sxx**2
      Syymed += Syy; syd += Syy**2
      Coxymed += Coxy; cxyd += Coxy**2
      Quxymed += Quxy; qxyd += Quxy**2
# ------------------------------------------------------------------------------
# the pahse spectrum for this block
# ------------------------------------------------------------------------------
      theta = arctan2(Quxy,Coxy)
      Thetamed += theta; thetaxyd += theta**2
   pass
# ------------------------------------------------------------------------------
# smooth spectra by averaging
# ------------------------------------------------------------------------------
   sqn = sqrt(nblocks)
   Sxxmed /= nblocks; sxd = asqrt(sxd/nblocks - Sxxmed**2)/sqn
   Syymed /= nblocks; syd = asqrt(syd/nblocks - Syymed**2)/sqn
   Coxymed /= nblocks; cxyd = asqrt(cxyd/nblocks - Coxymed**2)/sqn
   Quxymed /= nblocks; qxyd = asqrt(qxyd/nblocks - Coxymed**2)/sqn
   Thetamed /= nblocks; thetaxyd = asqrt(thetaxyd/nblocks - Thetamed**2)/sqn
# ------------------------------------------------------------------------------
# calculates the frequencies
# ------------------------------------------------------------------------------
   (Nyquist, nn) = nfrequencies(half_length,deltat)
   return (Nyquist, nn, Sxxmed, sxd, Syymed, syd, 
           Coxymed, cxyd, Quxymed, qxyd, Thetamed, thetaxyd)


def nfrequencies(N, deltat) :
   '''
   -----------------------------------------------------------------------------
   --> nfrequencies: with the time interval of the data, calculates the 
       Nyquist frequency and the corresponding cyclic frequencies in Hz.

   Nelson Luis Dias
   2016-09-07T11:23:47
   -----------------------------------------------------------------------------
   '''
# ------------------------------------------------------------------------------
# calculates the Nyquist frequency
# ------------------------------------------------------------------------------
   Nyquist = 1.0/(2.0*deltat)
   nn = zeros(N,float)
# ------------------------------------------------------------------------------
#  calculates the frequencies
# ------------------------------------------------------------------------------
   for k in range(N):
      naux = Nyquist * (k+1) / N ;
      nn[k] = naux
   pass
   return (Nyquist,nn) 

