#!/home/nldias/miniconda3/bin/python3
# -*- coding: iso-8859-1 -*-
# --------------------------------------------------------------------
# smog: spectral estimates from smooth ogives
#
# 2016-10-08T09:46:10
# 2016-12-17T14:13:44 cosmetic changes
# 2016-12-27T13:39:05 the omega fix
# 2016-12-29T10:48:08 finally, the variances
# 2017-01-06T10:25:39 freezing the gamma factor correction
# 2017-01-10T13:57:58 completely revamped the classical correction; no
# omega yet.
# 2017-01-20T11:15:05 fixing bug: Og[k-ow[k]] - Og[k+ow[k]+1] in smog
# 2017-01-27T16:04:19 numba does not like variables created inside
# ifs and loops. Moreover, numba is uncomfortable with *args kinds of
# things. Simplified things for numba: tilogvar uses numpy only, while
# _otilg now does the numba optimization. Seems to be working.
# --------------------------------------------------------------------
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
from numpy import zeros, ones, arange, cumsum, flipud, sqrt as asqrt
#from numba import jit
from math import copysign, log, exp
from sys import exit

# ------------------------------------------------------------------------------
# corrections
# ------------------------------------------------------------------------------
class correction:
   def __init__(self,M1):
      self.me = zeros(M1,float)
      self.gamma_O = ones(M1,float)
      self.gamma_G = ones(M1,float)
      self.omega = 0.0



def wog(p,a):
   '''
   wog returns the widths of the smoothing over array a:
   0,1,2, ... p, p, p, ..., p,p-1,p-2, ... 0

   you must run wog *before* smog
   '''
   M1 = len(a)
   w = p*ones(M1,int)
   w[0] = 0
   w[1:p+1] = arange(0,p,1)
   w[M1-p-1:M1-1] = flipud(w[1:p+1])
   w[M1-1] = 0
   return w

def ogive(deltaf,G):
   '''
   ogive(deltaf,G): use very simple integration to calculate the ogive 
   from a spectrum G with data sampled at frequency deltaf.
   2017-01-10T09:40:31 going back to a single frequency for Os and Gs
   '''
# --------------------------------------------------------------------
# 2016-10-08T09:47:12 re-created with numpy
# --------------------------------------------------------------------
   M1 = len(G)
   Og = zeros(M1,float)
   Og[0:M1] = cumsum(flipud(G[0:M1]))
   Og *= deltaf
   Og = flipud(Og)
   return Og




def _omegazero(sign,omega,deltaf,Gs,fGcor,Oszero):
   '''
   returns sum (gamma_G[k])^omega Gs[k] - Os[0]
   _omegazero is used in _unbias below
   
   2017-01-17T16:27:44
   '''
   M1 = len(Gs)
   locor = fGcor**omega
   Gcor = Gs*deltaf*locor
   ha = sign*Gcor[1:M1].sum()
   hb = sign*Oszero
#   print('ha, hb', ha, hb, ha-hb)
   return ha - hb 

#@jit
def _unbias(sign,deltaf,ow,Os,Gs):
   '''
   new bias correction algorithm based on exact arithmetic means for 
   both Os and Gs: 
   2017-01-10T13:50:18
   '''
# --------------------------------------------------------------------
# algorithm starts now
# --------------------------------------------------------------------   
   pOs = sign*Os                        # positive copies of the ogive
   pGs = sign*Gs                        # and the spectrum
   M1 = len(Os)                         # the length of Os
   MM = M1 - 1                          # == NN//2
# --------------------------------------------------------------------
# allocates exponents and correction factors
# --------------------------------------------------------------------
   cor = correction(M1)
   # me    = zeros(M1,float)              # the m's for Os
   # fOcor = ones(M1,float)               # the correction factor for Os
   # fGcor = ones(M1,float)               # the correction factor for Gs
   for k in range(MM-2,2,-1):           # only k's for which ow[k] >= 1
      fa = (k-ow[k])*deltaf             # left frequency
      fb = (k+ow[k])*deltaf             # right frequency
      fc = k*deltaf                     # center (arith mean) frequency
      alpha = pOs[k]
      if ( alpha <= 0.0 ):
         continue
      beta = pGs[k]
      if ( beta < 0.0 ):
         continue
      m = _mroot(fa,fb,beta/alpha)
      if ( m <= 0.0 ):                  # sometimes m misbehaves
         m = 0.01
      elif ( m > 5.0 ):
         m = 5.0
      pass
# --------------------------------------------------------------------
# the bias correction factors for the ogive
# --------------------------------------------------------------------
      fg = _gmeanf(fa,fb)               # the geometric-mean frequency
      gm = _gmeano(fa,fb,1.0,m)
      am = _ameano(fa,fb,1.0,m)
      cor.me[k] = m
      cor.gamma_O[k] = (gm/am)*(fg/fc)**m
      Os[k] *= cor.gamma_O[k]           # bias correction
# --------------------------------------------------------------------
# the bias correction for the spectrum
# --------------------------------------------------------------------
      gm = _gmeang(fa,fb,1.0,m)
      am = _ameang(fa,fb,1.0,m)
      cor.gamma_G[k] = (gm/am)*(fg/fc)**(m+1)
#      Gs[k] *= fGcor[k]                 # bias correction NOT YET
   pass
# --------------------------------------------------------------------
# the omega correction
# --------------------------------------------------------------------
   omega_a = -10.0
   omega_b = +10.0
   omega = (omega_a + omega_b)/2.0
   assert _omegazero(sign,omega_a,deltaf,Gs,cor.gamma_G,Os[0]) > 0.0
   assert _omegazero(sign,omega_b,deltaf,Gs,cor.gamma_G,Os[0]) < 0.0
   eps = 1.0e-4
   while (abs(omega_b - omega_a) > eps) :
      if _omegazero(sign,omega,deltaf,Gs,cor.gamma_G,Os[0]) >= 0.0 :
         omega_a = omega
      else:
         omega_b = omega
      pass
      omega = (omega_a + omega_b)/2.0
   pass
   cor.omega = omega
   Gs *= (cor.gamma_G**omega)   
   return (Os,Gs,cor)     # return corrections as well




#@jit
def smog(ow,deltaf,Or,unbiased=True):
   '''
   smoothes a raw ogive Or with a windows of half-width ow and sampling
   frequency deltaf and returns a smoothed ogive Os, and a smoothed 
   spectrum Gs.
   
   input:
   ow        -- half-length array of the smoothing window
   deltaf    -- frequency interval
   Or        -- the raw ogive
   nobias    -- to reduce, or not, the bias in Os

   output:
   Os        -- the smoothed ogive
   Gs        -- the smoothed spectrum
   '''
   M1 = len(Or)               # Or is the original ogive
   MM = M1 - 1                # this is NN//2
   Os = zeros(M1,float)       # Os is the smoothed ogive
   Gs = zeros(M1,float)       # Os is the smoothed (co)spectrum
   sign = copysign(1.0,Or[0]) # prevailing sign of the
                              # (co)spectrum
   Os[0] = Or[0]              # endpoints unaltered
   Os[1] = Or[1]
   for k in range(2,MM-1):    # smoothing loop
      Os[k] = Or[k-ow[k]:k+ow[k]+1].sum()/(2*ow[k]+1)
   pass
   Os[MM-1] = Or[MM-1]
   Os[MM] = Or[MM]            # endpoints unaltered
# --------------------------------------------------------------------
# og-derived smooth spectrum
# --------------------------------------------------------------------
   Gs[0] = 0.0                # endpoints unaltered
   Gs[1] = (Or[1]-Or[2])/deltaf
   for k in range(2,MM-1):
      Gs[k] = (Or[k-ow[k]] - Or[k+ow[k]+1])/((2*ow[k]+1)*deltaf)
   pass
   Gs[MM-1] = (Or[MM-1]-Or[MM])/deltaf
   Gs[MM] = Or[MM]/deltaf     # endpoints unaltered
   if unbiased :
      (Os,Gs,cor)  = _unbias(sign,deltaf,ow,Os,Gs)
      return (Os,Gs,cor)
   else:
      return (Os,Gs)
   pass






#@jit
def _otilg(sig2m,ow,deltaf):
   M1 = len(sig2m)
   gtilvar = zeros(M1,float)  # will return this
   otilvar = zeros(M1,float)  # will return this too
# --------------------------------------------------------------------
# the spectrum is simpler
# --------------------------------------------------------------------
   for k in range(1,M1-1):    # calculating loop
      gtilvar[k] = sig2m[k-ow[k]:k+ow[k]+1].sum()/(2*ow[k]+1)**2
   pass
# --------------------------------------------------------------------
# the ogive is more complicated
# --------------------------------------------------------------------
   ovar = 0.0
   for k in range(M1):
      ovar = 0.0
      for l in range (k-ow[k]+1,k+ow[k]+1):
         ovar += ((l - k + ow[k]+1)**2)*sig2m[l]
      pass
      ovar /= (2*ow[k]+1)**2
      for l in range(k+ow[k],M1):
         ovar += sig2m[l]
      pass
      otilvar[k] = ovar*(deltaf**2)    
   pass
   return (otilvar,gtilvar)


# --------------------------------------------------------------------
# do not @jit me!
# --------------------------------------------------------------------
def tilogvar(which,ow,deltaf,*args):
   '''
   Calculates the variance of the smoothed spectrum and smoothed ogive

   "which" is a string that gives you the case:
   'gxx' -- the autospectrum     :: tilogvar('gxx',ow,Gxx)
   'cxy' -- the cospectrum       :: tilogvar('cxy',ow,Gxx,Gyy,Coxy,Quxy)
   'qxy' -- the quadspectrum     :: tilogvar('qxy',ow,Gxx,Gyy,Coxy,Quxy)
   '''
# --------------------------------------------------------------------
# Which case are we talking about?
# --------------------------------------------------------------------
   if which == 'gxx':         # the autospectrum and its ogive
      assert len(args) == 1
      Gxx = args[0]
      sig2m = Gxx**2          # over the whole array
   elif which == 'cxy' :
      assert len(args) == 4   # the cospectrum and its ogive
      Gxx = args[0]
      Gyy = args[1]
      Coxy = args[2]
      Quxy = args[3]
      sig2m = (Gxx*Gyy + Coxy**2 - Quxy**2)/2.0
   elif which == 'qxy' :
      assert len(args) == 4   # the quadspectrum and its ogive
      Gxx = args[0]
      Gyy = args[1]
      Coxy = args[2]
      Quxy = args[3]
      sig2m = (Gxx*Gyy - Coxy**2 + Quxy**2)/2.0
   else :
      print('tilogvar --> first argument must be one of "gxx", "cxy" or "qxy".')
      print('             Fix me, please. Bye.')
      exit(1)
   pass
   return _otilg(sig2m,ow,deltaf)



def _ameano(fa,fb,C,m):
   '''
   arithmetic mean, power-law ogive
   '''
   if m == 1.0:
      return (C/(fb-fa))*log(fb/fa)
   else:
      return (C/((fb-fa)*(-m+1.0)))*(fb**(-m+1.0) - fa**(-m+1.0))
   pass

def _ameang(fa,fb,C,m):
   '''
   arithmetic mean, power-law spectrum
   '''
   return (C/(fb-fa))*(fa**(-m) - fb**(-m))

def _gmeano(fa,fb,C,m):
   '''
   geometric mean, power-law ogive
   '''
   aux = log(C)
   aux += m
   aux -= (m/(fb-fa))*(fb*log(fb) - fa*log(fa))
   return exp(aux)

def _gmeang(fa,fb,C,m):
   '''
   geometric mean, power-law spectrum
   '''
   aux = log(m*C)
   aux += (m+1.0)
   aux -= ((m+1.0)/(fb-fa))*(fb*log(fb) - fa*log(fa))
   return exp(aux)

def _gmeanf(fa,fb):
   '''
   geometric-mean frequency
   '''
   aux = fb*log(fb) - fa*log(fa) + fa - fb
   aux /= (fb - fa)
   return exp(aux)




#@jit
def _mroot(fa,fb,goa):
   '''
   _mroot Calculates the numerical solution of

      goa = _ameang(fa,fb,1.0,m)/_ameano(fa,fb,1.0,m)

   for the unknown m, using bisection.
   '''
   ma = -1.0
   mb = 5.0
   m = (ma + mb)/2.0
   epsm = 1.0e-4
   kontrol = 0
   while abs(mb - ma) > 1.0e-4 :
      res = _ameang(fa,fb,1.0,m)/_ameano(fa,fb,1.0,m) - goa
      if res >= 0.0 :
         mb = m
      else :
         ma = m
      pass
      kontrol += 1
      if kontrol > 1000:
         print('too many iterations')
         exit(1)
      pass
      m = (ma + mb)/2.0
   pass
   return m
         
      

