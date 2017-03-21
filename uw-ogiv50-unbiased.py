#!/home/nldias/miniconda3/bin/python3
# -*- coding: iso-8859-1 -*-
'''
uw-ogiv50.py: spectra from ogives. 


Nelson L Dias 

2016-09-08T09:45:49 -- all over again.
2016-09-26T11:40:20 -- now using smogg
2016-10-08T14:28:46 -- now using smog.py
2016-12-29T09:58:50 -- including the calculation of variances
2017-01-05T15:17:51 -- deltaf in tilogvar
2017-02-22T10:42:15 -- public version
'''
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
# --------------------------------------------------------------------
# documentation
# --------------------------------------------------------------------
from argparse import ArgumentParser
parser = ArgumentParser(description=__doc__)
args = parser.parse_args()
# --------------------------------------------------------------------
# imports
# --------------------------------------------------------------------
from turbstat import lindetrend
from smog import ogive, wog, smog, tilogvar
from nspectrum import cross_density
from os import system
# --------------------------------------------------------------------
# let's allocate arrays
# --------------------------------------------------------------------
from numpy import zeros, cov, arange, argsort
NN = 36000
MM = NN//2
M1 = MM+1
u = zeros(NN,float)
v = zeros(NN,float)
w = zeros(NN,float)
# --------------------------------------------------------------------
# let's read 30-min. *all* data from the disk:
# --------------------------------------------------------------------
fin = open('20130812-0310.out','rt')
k = 0
for line in fin:
   campo = line.split()
   u[k] = float(campo[0])
   v[k] = float(campo[1])
   w[k] = float(campo[2])
   k += 1
pass
fin.close()
# --------------------------------------------------------------------
# just checking if rotation is here
# --------------------------------------------------------------------
umean = u.mean()
vmean = v.mean()
wmean = w.mean()
# --------------------------------------------------------------------
# linear detrending
# --------------------------------------------------------------------
u = lindetrend(u)
w = lindetrend(w)
# --------------------------------------------------------------------
# now let's look at the cross-spectrum between u and w 
# --------------------------------------------------------------------
I = complex(0.0,1.0)
z = u + I*w
deltat = 0.05
(Guu,Gww,Couw,Quuw) = cross_density(deltat,z)
# ---------------------------------------------------------------------
# frequencies
# ---------------------------------------------------------------------
deltaf = 1.0/(NN*deltat)
freq = arange(0.0,10.0+deltaf,deltaf,float)
# ---------------------------------------------------------------------
#  the smoothed ogives of the Guu, Gww and Couw spectra; after
#  integration and smoothing, I will re-use Guu, Gww and Couw
#  --------------------------------------------------------------------
ow = wog(50,Guu)
(Oguu,Guu,cor_uu) = smog(ow,deltaf,ogive(deltaf,Guu),True)
(Oguuvar,Guuvar) = tilogvar('gxx',ow,deltaf,Guu)
(Ogww,Gww,cor_ww) = smog(ow,deltaf,ogive(deltaf,Gww),True)
(Ogwwvar,Gwwvar) = tilogvar('gxx',ow,deltaf,Gww)
# --------------------------------------------------------------------
# the cospectrum
# --------------------------------------------------------------------
(OgCuw,Couw,cor_uw) = smog(ow,deltaf,ogive(deltaf,Couw),True)
# --------------------------------------------------------------------
# the quadrature spectrum
# --------------------------------------------------------------------
(OgQuw,Quuw,qor_uw) = smog(ow,deltaf,ogive(deltaf,Quuw),True)
# --------------------------------------------------------------------
# the variances can only be calculated with both Couw and Quuw
# obtained
# --------------------------------------------------------------------
(OgCuwvar,Couwvar) = tilogvar('cxy',ow,deltaf,Guu,Gww,Couw,Quuw)
(OgQuwvar,Quuwvar) = tilogvar('qxy',ow,deltaf,Guu,Gww,Couw,Quuw)
# --------------------------------------------------------------------
# print, and we are done
# --------------------------------------------------------------------
fou = open('uw-ogiv50-unbiased.out','wt',encoding='iso-8859-1')
def seeme(fou):
   fou.write('#    1     f')
   fou.write('     2   Oguu')
   fou.write('     3    Guu')
   fou.write('     4   Ogww')
   fou.write('     5    Gww')
   fou.write('     6  OgCuw')
   fou.write('     7   Couw')
   fou.write('     8  OgQuw')
   fou.write('     9   Quuw')
   fou.write('   10 varOguu')
   fou.write('   11 var Guu')
   fou.write('   12 varOgww')
   fou.write('   13 var Gww')
   fou.write('   14 varOgCuw')
   fou.write('   15 var Couw')
   fou.write('   16 varOgQuw')
   fou.write('   17 var Quuw')
   fou.write('   18     meuu')
   fou.write('   19 gammaOuu')
   fou.write('   20 gammaGuu')
   fou.write('   21     meww')
   fou.write('   22 gammaOww')
   fou.write('   23 gammaGww')
   fou.write('   24     meuw')
   fou.write('   25 gammaOuw')
   fou.write('   26 gammaGuw')
   fou.write('   27     mquw')
   fou.write('   28 gammqOuw')
   fou.write('   29 gammqGuw')
   fou.write('\n')
pass
for k in range(M1):
   if (k % 25) == 0:
      seeme(fou)
   pass
   fou.write(('%12.6g '*29 + '\n') % 
                (freq[k],Oguu[k],Guu[k],Ogww[k],Gww[k],OgCuw[k],Couw[k],OgQuw[k],Quuw[k],
                         Oguuvar[k],Guuvar[k],Ogwwvar[k],Gwwvar[k],OgCuwvar[k],Couwvar[k],OgQuwvar[k],Quuwvar[k],
                         cor_uu.me[k],cor_uu.gamma_O[k],cor_uu.gamma_G[k],
                         cor_ww.me[k],cor_ww.gamma_O[k],cor_ww.gamma_G[k],
                         cor_uw.me[k],cor_uw.gamma_O[k],cor_uw.gamma_G[k],
                         qor_uw.me[k],qor_uw.gamma_O[k],qor_uw.gamma_G[k]))
pass
fou.close()
