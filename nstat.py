#!/home/nldias/miniconda3/bin/python3
# -*- coding: iso-8859-1 -*-
# 
# nstat: A statistics library that one day will grow up
#
# 2010-01-14T16:33:50
# 2010-01-14T16:33:54
# ------------------------------------------------------------------------------
from math import sqrt, exp, log, floor, ceil

from numpy import argmax,zeros, log as alog
from copy import deepcopy
from numba import jit

@jit
def stat1(x):
   '''
   -----------------------------------------------------------------------------
      xm = stat1(x) calculates the mean xm of x

      Nelson Luís Dias
      2010-01-14T16:06:55
      2010-01-14T16:06:59
   -----------------------------------------------------------------------------
   '''
# ------------------------------------------------------------------------------
# be careful with empty arrays
# ------------------------------------------------------------------------------
   n = len(x)
   assert n > 0
   fn = float(n) # just in case
# ------------------------------------------------------------------------------
# first the mean
# ------------------------------------------------------------------------------
   xm = 0.0
   for xi in x:
      xm += xi 
   xm /= fn
   return xm

@jit
def stat2(x):
   '''
   -----------------------------------------------------------------------------
      (xm,xv) = stat2(x) calculates the mean xv and variance xv of x

      Nelson Luís Dias
      2009-05-12T09:18:31
      2009-05-12T09:18:38
   -----------------------------------------------------------------------------
   '''
# ------------------------------------------------------------------------------
# be careful with empty arrays
# ------------------------------------------------------------------------------
   n = len(x)
   if n == 0:
      return ( float('nan'), float('nan') )
   fn = float(n) # just in case
# ------------------------------------------------------------------------------
# first the mean
# ------------------------------------------------------------------------------
   xm = 0.0
   for xi in x:
      xm += xi 
   xm /= fn
# ------------------------------------------------------------------------------
# now the variance
# ------------------------------------------------------------------------------
   xv = 0.0
   for xi in x:
      xv += (xi - xm)**2
   xv /= fn
   return (xm,xv)

@jit
def covar(x,y,xmed,ymed):
   '''
   -----------------------------------------------------------------------------
      sxy = covar(x,y,xmed,ymed) gives the covariance
         sxy = (1/n) sum_{k=1}^n (x_i - xmed)(y_i - ymed).

      Nelson Luís Dias
      2010-01-14T16:25:12
      2010-01-14T16:25:15
   -----------------------------------------------------------------------------
   ''' 
   n = len(x)
   if n == 0:
      return float('nan')
   assert n == len(y)
   sxy = 0
   fn = float(n) # just in case
   for (xi,yi) in zip(x,y):
      sxy += (xi - xmed)*(yi - ymed)
   return sxy/fn 

@jit
def reglin(x,y):
   '''
   -----------------------------------------------------------------------------
   calculates the linear regression y = ax + b, plus 
   the correlation coefficient r 

   Nelson Luís Dias
   2010-02-11T15:23:42
   2010-02-11T15:23:45
   -----------------------------------------------------------------------------
   '''
   n = len(x)
   if n == 0:
      exit("don't give me empty stuff") 
   assert n == len(y)
   (xavg,xvar) = stat2(x)
   (yavg,yvar) = stat2(y)
   coxy = covar(x,y,xavg,yavg)
   a = coxy / xvar 
   b = yavg - (a * xavg) 
   r = coxy / sqrt( xvar * yvar ) 
   return (a,b,r)


def reglina(x,y):
   '''
   -----------------------------------------------------------------------------
   calculates the linear regression y = a x , the correlation coefficient r, 
   the standard deviation sy of the distribution of y given x, and the standard 
   deviation sa the estimator of a 
   -----------------------------------------------------------------------------
   '''
# ------------------------------------------------------------------------------
# check dimensions
# ------------------------------------------------------------------------------
   nx = len(x)
   ny = len(y)
   assert (nx == ny)
   fn = float(nx)
# ------------------------------------------------------------------------------
# starts by calculating central moments
# ------------------------------------------------------------------------------
   (xavg,xvar) = stat2(x)
   (yavg,yvar) = stat2(y)
   coxy = covar(x,y,xavg,yavg)
# ------------------------------------------------------------------------------
# translates to non-central moments
# ------------------------------------------------------------------------------
   sx20 = fn * ( xvar + (xavg*xavg) )
   sy20 = fn * ( yvar + (yavg*yavg) )
   sxy0 = fn * ( coxy + (xavg*yavg) )
# ------------------------------------------------------------------------------
# obtains the slope
# ------------------------------------------------------------------------------
   a = sxy0 / sx20 
# ------------------------------------------------------------------------------
# 2007-09-21T01:40 each time I look at this I get a different result; at least 
# this time I have documented the equation in stat.tex
# ------------------------------------------------------------------------------
   sa = sqrt( yvar / sx20 ) 
# ------------------------------------------------------------------------------
# now calculates mean square error and correlation coefficient via the
# definition that r is the ratio of explained to total variance of y
# ------------------------------------------------------------------------------
   s2 = ( sy20 - 2.0 * a * sxy0 + a * a * sx20 ) / fn 
   sy = sqrt(s2)
   if (s2 > yvar):
      r = 0.0 
   else:
      r = sqrt(1.0 - s2/yvar)
   return (a,r,sy,sa)

def reglinb(a,x,y):
   '''
   -----------------------------------------------------------------------------
   calculates the linear regression y = a x + b, with a fixed.
   2011-03-09T10:22:09
   -----------------------------------------------------------------------------
   '''
# ------------------------------------------------------------------------------
# check dimensions
# ------------------------------------------------------------------------------
   nx = len(x)
   ny = len(y)
   assert (nx == ny)
   fn = float(nx)
# ------------------------------------------------------------------------------
# starts by calculating central moments
# ------------------------------------------------------------------------------
   xavg = stat1(x)
   yavg = stat1(y)
   return yavg - a*xavg 



def rwin(x,y,ll):
   '''
   -----------------------------------------------------------------------------
      (a,b,r) = rwin(x,y,ll) gives a windowed linear regression of x versus y
      over their common length.  Each linear regression involves l+1 points,
      centered at 'k'

      Nelson Luís Dias
      2009-12-17T16:14:08
      2009-12-17T16:14:11
   -----------------------------------------------------------------------------
   ''' 
   m = len(ll)
   n = len(x)
   ny = len(y)
   assert n > 0 
   assert n == ny
# ------------------------------------------------------------------------------
# initalizes a list of coefficients
# ------------------------------------------------------------------------------
   aa = zeros((m,n),float)
   bb = zeros((m,n),float)
   rr = zeros((m,n),float)
   rm = zeros((m,n),float)
   for i in range(m):
      l = ll[i]
      hl = l//2
# ------------------------------------------------------------------------------
# for each point:
# ------------------------------------------------------------------------------
      for k in range(hl,n-hl):
# ------------------------------------------------------------------------------
# arrange l points around (x[k],y[k]) in just one way
# ------------------------------------------------------------------------------
         kmin = k-hl
         kmax = k+hl 
         xx = x[kmin:kmax+1]
         yy = y[kmin:kmax+1]
         xxmed = stat1(xx)
         yymed = stat1(yy)
         sx2 = covar(xx,xx,xxmed,xxmed)
         sy2 = covar(yy,yy,yymed,yymed)
         sxy = covar(xx,yy,xxmed,yymed)
         a = sxy/sx2
         b = yymed - a*xxmed
         r = sxy/sqrt(sx2*sy2)
# ------------------------------------------------------------------------------
# add coefficients to the list
# ------------------------------------------------------------------------------
         aa[i,k] = a
         bb[i,k] = b
         rr[i,k] = r
         rm[i,k] = abs(r)
# ------------------------------------------------------------------------------
# now fill the voids (fingers crossed!)
# ------------------------------------------------------------------------------
      for k in range(0,hl):
         aa[i,k] = aa[i,hl]
         bb[i,k] = bb[i,hl]
         rr[i,k] = rr[i,hl]
         rm[i,k] = rm[i,hl]
      for k in range(n-hl,n):
         aa[i,k] = aa[i,n-hl-1]
         bb[i,k] = bb[i,n-hl-1]
         rr[i,k] = rr[i,n-hl-1]
         rm[i,k] = rm[i,n-hl-1]
# ------------------------------------------------------------------------------
# get the best correlation coefficient!
# ------------------------------------------------------------------------------
   af = []
   bf = []
   rf = []
   for k in range(n):
      imax = argmax(rm[0:m,k])
      af.append(aa[imax,k])
      bf.append(bb[imax,k])
      rf.append(rr[imax,k])
   return (af,bf,rf)



def median(x):
   '''
   ---------------------------------------------------------------------------
   median(x) returns the median of the numpy array x (must be a numpy array)

   2012-08-21T08:55:53
   ---------------------------------------------------------------------------
   '''
   n = len(x)
   if n == 0:
      print ('stat-->median: empty array')
      exit(1)
   pass
   indx = x.argsort()
#  ---------------------------------------------------------------------------
#  is n even or odd ?
#  ---------------------------------------------------------------------------
   if n == 1:
      xmedian = x[0]
   elif ( (n % 2) == 0 ):
      xmedian = (x[indx[n//2 - 1]] + x[indx[n//2]]) / 2 
   else:
      xmedian = x[indx[n//2]]
   return xmedian

def quantile(x,q):
   '''
   ---------------------------------------------------------------------------
   quantile(x,q) returns the q quantile of array x (must be a numpy array)

   2012-08-21T13:47:01
   ---------------------------------------------------------------------------
   '''
   n = len(x)
   if n == 1 :
      return x[0]
   pass
   k = int(floor(q*n))
   l = int(ceil(q*n))
   while l > n-1 : l -= 1
   assert l >= k
   if k == l :
      return x[k]
   pass
   indx = x.argsort()
   xk = x[indx[k]]
   xl = x[indx[l]]
   y = xk + ((xl - xk)/(k - l))*(q*n - l)
   return y

def classlogmedian(p,z,w):
   '''
   ----------------------------------------------------------------------------
   given an array z: take the logs, and divide into p classes logarithmically;
   obtain the median of each class; return an array of p elements with the
   medians, and another array of p elements of the 68% quantile of absolute
   deviations.  p must be a numpy array

   2012-08-22T17:59:32
   -----------------------------------------------------------------------------
   '''
   n = len(z)
   assert n == len(w)    # two arrays of equal length?
   x = alog(z)           # take logs
   y = alog(w)
   xmin = x.min()
   xmax = xmin + (x.max() - xmin)*(10001.0/10000.0)
   dx = (xmax - xmin)/p  # interval of each class
# ------------------------------------------------------------------------------
# there may be a better way, but here I initialize a list of p empty lists, each
# one of them different (with a different pointer)
# ------------------------------------------------------------------------------
   xcls = []
   ycls = []
   ncls = []
   for k in range(p):
      xcls.append([])
      ycls.append([])
      ncls.append(0)
# -------------------------------------------------------------------------------
# loop over array, to distribute into classes
# -------------------------------------------------------------------------------
   for i in range(n) :
      k = int(floor((x[i] - xmin)/dx))
      assert 0 <= k < p
      xcls[k].append(x[i])
      ycls[k].append(y[i])
      ncls[k] += 1
   pass
# -------------------------------------------------------------------------------
# there may be empty classes! cleanup.
# -------------------------------------------------------------------------------
   pp = p
   for k in range(p):
      while ncls[k] == 0:
         ncls[k:p-1] = ncls[k+1:p]
         xcls[k:p-1] = xcls[k+1:p]
         ycls[k:p-1] = ycls[k+1:p]
         pp -= 1
      pass
   pass
   ncls = ncls[0:pp]
   xcls = xcls[0:pp]
   ycls = ycls[0:pp]
# ------------------------------------------------------------------------------
# now there are no more empty classes
# ------------------------------------------------------------------------------      
   from numpy import array,zeros
   xmed = zeros(pp,float) # the median in each class
   xdev = zeros(pp,float) # the 68% quantile of absolute devs in each class
   ymed = zeros(pp,float) # the median in each class
   ydev = zeros(pp,float) # the 68% quantile of absolute devs in each class
   for k in range(pp):
      xwork   = array(xcls[k])
      ywork   = array(ycls[k])
      xmed[k] = median(xwork)
      xdif    = abs(xwork - xmed[k])  # xdif is local to the loop
      xdev[k] = quantile(xdif,0.68)
      ymed[k] = median(ywork)
      ydif    = abs(ywork - ymed[k])
      ydev[k] = quantile(ydif,0.68)
   pass
# ------------------------------------------------------------------------------
# amazingly, that should do it!
# ------------------------------------------------------------------------------
   from numpy import exp
   return (pp,exp(xmed),exp(ymed),exp(xmed - xdev), exp(xmed + xdev),exp(ymed - ydev), exp(ymed+ydev))


def nanstat2(x):
   '''
   -----------------------------------------------------------------------------
      (xm,xv) = nanstat2(x) calculates the mean xv and variance xv of x, but
      if elements of the *list* x are not floats, they are simply dismissed

      Nelson Luís Dias
      2014-07-30T18:08:14
   -----------------------------------------------------------------------------
   '''
# ------------------------------------------------------------------------------
# eliminate unsocial elements
# ------------------------------------------------------------------------------
   y = deepcopy(x)
   bign = len(x)
   k = 0
   for xi in x:
      if ((type(xi) == float) or (type(xi) == int)):
         y[k] = xi
         k += 1
      pass
   pass
   x = y[0:k]
# ------------------------------------------------------------------------------
# be careful with empty arrays
# ------------------------------------------------------------------------------
   n = len(x)
   if n == 0:
      return ( float('nan'), float('nan') )
   fn = float(n) # just in case
# ------------------------------------------------------------------------------
# first the mean
# ------------------------------------------------------------------------------
   xm = 0.0
   for xi in x:
      xm += xi 
   xm /= fn
# ------------------------------------------------------------------------------
# now the variance
# ------------------------------------------------------------------------------
   xv = 0.0
   for xi in x:
      xv += (xi - xm)**2
   xv /= fn
   return (xm,xv)

@jit
def stat_arm(
n,             # number of elements
k,             # window for calculation of initial  running mean
l,             # window for calculation of running means
x              # data array
):
   '''
   --> stat_f4_arm: calculates the overall mean and variance
       with respect to a running mean from an array x. The
       running mean is recursively obtained as
         
       m[i] = (1/l)*x[i] + ( 1 - (1/l))*m[i-1]
    
       where l is the window width of the running mean.  An
       initial running mean m[0] is calculated with an
       initial width k.
   '''
# ----------------------------------------------------------
# start
# ----------------------------------------------------------
   fl = float(l)
# ----------------------------------------------------------
# auxiliary array, to warm up the running mean
# ----------------------------------------------------------
   y = x[0:k]
# ----------------------------------------------------------
# loop to calculate initial running mean, xm[0]
# ----------------------------------------------------------
   sum = 0.0
   ym = y[0]
   for i in range(1,k):
      sum  = sum + y[i] - ym
      ym      = sum / fl
   pass 
# ----------------------------------------------------------
# the real thing
# ----------------------------------------------------------
   xm = zeros(n,float)
   xm[0] = ym
   for i in range(1,n):
      sum  = sum + x[i] - xm[i-1] 
      xm[i]   = sum / fl
   pass
   return(xm)


@jit
def stat_crm(
w,             # window for calculation of running means
x              # data array
):
   '''
   --> stat_crm: centered running mean (width l) of x
         
       m[i] = (1/l)*sum_{i-l}^{i+l} x[i] 
   '''
# ----------------------------------------------------------
# start
# ----------------------------------------------------------
   n = len(x)
   xm = x.copy()                # make a copy
   for k in range(n):
      lw = min(k,w)             # look left
      lw = min(n-1-k,lw)        # look right (take smallest 
                                # in both directions)
      aux = x[k-lw:k+lw+1].sum()
      xm[k] = aux/(2*lw+1)
   pass 
   return(xm)

@jit
def stat_grm(
l,             # window for calculation of running means
x              # data array
):
   '''
   --> stat_crm: centered geometric running mean (width l) of x
         
       m[i] = (1/l)*sum_{i-l}^{i+l} x[i] 
   '''
# ----------------------------------------------------------
# start
# ----------------------------------------------------------
   n = len(x)
   xm = x.copy()
   for i in range(l,n-l):
      prod = 1.0
      for k in range(i-l,i+l+1):
         prod *= x[k]
      pass
      xm[i] = prod**(1.0/l)
   pass 
   xm[0:l] = xm[l]             # fills beginning of array
   xm[n-l:n] = xm[n-1]         # fills end of array
   return(xm)


from numpy import amax, amin

@jit
def gmean(x):
   '''
   the geometric mean is tricky!  best to take logs, and pay the price
   '''
   n = len(x)
   fn = float(n)
#   xl = alog(x)
   xl = zeros(n,float)
   for k in range(n):
      if x[k] <= 0.0:
         exit('problem here!')
      xl[k] = log(x[k])
   pass
   xml = stat1(xl)
   xgmean = exp(xml)
   return xgmean


def classlogavg (#
maxcl,           # the maximum # of classes desired
pr_sign,         # prevailing sign 
indx,            # the index ordering x from smallest to biggest
x,               # the big array of abscissas
y                # the big array of ordinates
):
   '''
   classlogavg: class-averages a big array: the abscissas are divided into maxcl
       classes; then it returns the mean of the abscissas and the ordinates for
       each class; this function contemplates the possibility that some classes
       may be empty.

   returns:
      nsm              # the number of non-empty classes
      npclsign         # the number of points used for averaging per class
      xsm              # the smoothed array of abscissas
      ysm              # the smoothed array of ordinates
   '''
# ------------------------------------------------------------------------------
# the length of the big array
# ------------------------------------------------------------------------------
   ntotal = len(x)
# ------------------------------------------------------------------------------
# number of points per class: will need two of them!
# ------------------------------------------------------------------------------
   npclsign = zeros(maxcl,int)
   npclass = zeros(maxcl,int)
# ------------------------------------------------------------------------------
# the smoothed arrays
# ------------------------------------------------------------------------------
   xsm = zeros(maxcl,float)
   ysm = zeros(maxcl,float)
# ------------------------------------------------------------------------------
# log increments
# ------------------------------------------------------------------------------
   logxmin = log(x[indx[0]])
   logxmax = log(x[indx[ntotal-1]])
# ------------------------------------------------------------------------------
# counts the number of points per class
# ------------------------------------------------------------------------------
   i = 0
   classe = 0
   while ( i < ntotal ):
      npclass[classe] = 0
# ------------------------------------------------------------------------------
# warning! truncation errors?
# ------------------------------------------------------------------------------
      alfa = float(classe+1)/float(maxcl)
      assert ( alfa <= 1.0 )
      logxsup = alfa * logxmax + (1.0 - alfa)*logxmin
      while ( (i < ntotal) and ( log(x[indx[i]]) <= logxsup) ):
         npclass[classe] += 1
         i += 1
      pass
      classe += 1
   pass
# ------------------------------------------------------------------------------
# at this point, class should be equal to maxcl
# ------------------------------------------------------------------------------
   assert ( classe == maxcl )
# ------------------------------------------------------------------------------
# begins averaging
# ------------------------------------------------------------------------------
   nsm = 0                              # the number of "valid" classes
   first = 0                            # the first pt in each class
   for classe in range(maxcl): 
      if ( npclass[classe] > 0 ) :
# ------------------------------------------------------------------------------
# needs a local array for storing contiguously the members of a class   
# ------------------------------------------------------------------------------
         xlocal = zeros( npclass[classe],float) 
         ylocal = zeros( npclass[classe],float)
# ------------------------------------------------------------------------------
# contiguous storage
# ------------------------------------------------------------------------------
         k = 0 
         for j in range(npclass[classe]):
            indlocal = indx[first+j]
# ------------------------------------------------------------------------------
# checks index consistency
# ------------------------------------------------------------------------------
            if ( indlocal > ntotal - 1 ):
               print("stat-->classlogavg: incorrect index in class averaging\n")
               exit(1)
            pass
# ------------------------------------------------------------------------------
# checks signal consistency
# ------------------------------------------------------------------------------
            yy = pr_sign * y[indlocal] 
            if ( yy > 0.0 ) :
               if x[indlocal] <= 0.0:
                  exit('I should not be here')
               xlocal[k] = x[indlocal] 
               ylocal[k] = yy 
               k += 1
            pass
         pass
# ------------------------------------------------------------------------------
# the VALID number of points per class is now limited to those points having the
# prevailing sign: calculates the x- and y- averages of each class for non-empty
# classes
# ------------------------------------------------------------------------------
         if ( k > 0 ) :
            xavg = gmean(xlocal[0:k])
            yavg = gmean(ylocal[0:k])
            xsm[nsm] = xavg 
            ysm[nsm] = pr_sign * yavg 
            npclsign[nsm] = k 
# ------------------------------------------------------------------------------
# increments the number of non-empty classes
# ------------------------------------------------------------------------------
            nsm += 1
         pass
      pass
# ------------------------------------------------------------------------------
# increments "first" to be the starting index of the next class
# ------------------------------------------------------------------------------
      first += npclass[classe] 
   pass
   return (nsm, npclsign, xsm, ysm)

@jit
def autocovar(x,xm,lags):
   '''
   -----------------------------------------------------------------------------
   cxx = autocovar(x) calculates the autocovariance function of x, whose mean
   is xm, calculated at all lags prescribed by the array of integers "lags"

   Nelson Luís Dias
   2017-09-12T14:50:54
   2017-09-12T17:48:24
   -----------------------------------------------------------------------------
   '''
# ------------------------------------------------------------------------------
# be careful with empty arrays
# ------------------------------------------------------------------------------
   n = len(x)
   assert n > 0
   fn = float(n) # just in case
   ml = len(lags)
   print('n, ml = ',n, ml)
# ------------------------------------------------------------------------------
# first the mean
# ------------------------------------------------------------------------------
   cxx = zeros(ml)
   for k in range(0,ml):
      ss = 0.0
      lag = lags[k]
      print('lag = ',lag)
      for k in range(0,n-lag):
         ss += (x[k] - xm)*(x[k+lag]-xm)
      pass
      cxx[lag] = ss/(n-lag)
   pass
   return cxx
