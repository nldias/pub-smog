#!/home/nldias/miniconda3/bin/python3
# -*- coding: iso-8859-1 -*-
# 
# turbstat: specific processing of turbulence statistics
#
# 2016-12-17T15:34:42 a new turbstat is born, independent from smog
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
from numpy import array, arange, dot, zeros
from nstat import reglin
#from numba import jit, float64
from math import atan2, sqrt, sin, cos
def lindetrend(x):
   '''
   -----------------------------------------------------------------------------
   linear detrending: extract the linearly varying "mean"
   -----------------------------------------------------------------------------
   '''
# ------------------------------------------------------------------------------
# be careful with empty arrays
# ------------------------------------------------------------------------------
   n = len(x)
   assert n > 0
# ------------------------------------------------------------------------------
# make an array of times
# ------------------------------------------------------------------------------
   t = arange(0,n,1,float)
# ------------------------------------------------------------------------------
# first the mean
# ------------------------------------------------------------------------------
   (a,b,r) = reglin(t,x)
   mx = a*t + b               # the amazing vectorized operaton
   x = x - mx                 # another amazing vectorized operation
   return x


#@jit
def uvwrot(u,v,w):
   '''
   uvwrot rotates 3 arrays of sonic velocities
   '''
   ubar = u.mean()
   vbar = v.mean()
   wbar = w.mean()
   alphax = atan2(vbar,ubar)
   alphaz = atan2(wbar,sqrt(ubar**2 + vbar**2))
   CC = array( [ [ cos(alphax)*cos(alphaz),  sin(alphax)*cos(alphaz), sin(alphaz)],
                 [-sin(alphax),              cos(alphax),             0],
                 [-cos(alphax)*sin(alphaz), -sin(alphax)*sin(alphaz), cos(alphaz)] ] )
   n = len(u)
   for k in range(n):
      (u[k],v[k],w[k]) = dot(CC,array([u[k],v[k],w[k]]))
   pass
   return (u,v,w)

   
