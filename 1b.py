#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 15:56:28 2021

@author: hatus
"""

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

hdulist = fits.open('image0.fits')
hdulist.info()

data = hdulist[0].data

print(data.shape)

# Plot the 2D array
plt.imshow(data, cmap=plt.cm.viridis)
plt.xlabel('x-pixels (RA)')
plt.ylabel('y-pixels (Dec)')
plt.colorbar()
plt.show()


def load_fits(archivo):
  hdulist = fits.open(archivo)
  data = hdulist[0].data
  ind = np.unravel_index(np.argmax(data, axis=None), data.shape)
  return ind

def mean_fits(archivos):
  listado = []
  for archivo in archivos:
    hdulist = fits.open(archivo)
    listado.append(hdulist[0].data)
  contador = 0
  suma = np.array(0)
  for mapas in listado:
    contador = contador + 1
    suma = suma + mapas
  return suma / contador