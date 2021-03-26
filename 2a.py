#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 16:44:34 2021

@author: 
"""

# Write your list_stats function here.

def list_stats(arreglo):
  if len(arreglo) == 1:
    return (arreglo[0], arreglo[0])
  arreglo.sort()
  mean = sum(arreglo)/len(arreglo)
  mid = len(arreglo) // 2
  if len(arreglo) % 2 == 1:
    return (arreglo[mid], mean)
  else:
    return ((arreglo[mid] + arreglo[mid-1])/2, mean) 
  return(len(arreglo) % 2, sum(arreglo)/len(arreglo))
  
import time, numpy as np
n = 10**7
data = np.random.randn(n)
start = time.perf_counter()
# potentially slow computation
mean = sum(data)/len(data)
seconds = time.perf_counter() - start

print('That took {:.2f} seconds.'.format(seconds))

start = time.perf_counter()
# potentially slow computation
mean = np.mean(data)
seconds = time.perf_counter() - start

print('That second took {:.2f} seconds.'.format(seconds))

for i in range(10):
    print (i)

import sys, numpy as np

a = np.array([])
b = np.array([1, 2, 3])
c = np.zeros(10**6)

for obj in [a, b, c]:
    print('sys:', sys.getsizeof(obj), 'np:', obj.nbytes)
    
import numpy as np

a = np.zeros(5, dtype=np.int32)
b = np.zeros(5, dtype=np.float64)

for obj in [a, b]:
  print('nbytes         :', obj.nbytes)
  print('size x itemsize:', obj.size*obj.itemsize)
  
  # Write your function median_FITS here:
import sys, time, numpy as np
from astropy.io import fits

def median_fits(archivos):
  listado = []
  start = time.perf_counter()
  tamano = 0
  for archivo in archivos:
    hdulist = fits.open(archivo)
    listado.append(hdulist[0].data)
    tamano = tamano + hdulist[0].data.nbytes / 1024
  imagen = np.zeros_like(listado[0])
  np.median(listado, axis = 0, out = imagen)
  fin = time.perf_counter() - start
  return imagen, fin, tamano #sys.getsizeof(listado)

# Write your median_bins and median_approx functions here.

import numpy as np

def median_bins(listado, B):
  lista = np.array(listado)
  prom = np.mean(lista)
  std = np.std(lista)
  min_v = prom-std
  ancho = 2 * std / B
  cant_min_v = len( np.extract(lista<min_v, lista) )
  bins = np.array(range(B))
  for i in range(B):
    min_B = min_v + ancho * i
    max_B = min_v + ancho * (i+1)
    cond_1 = lista >= min_B
    cond_2 = lista < max_B
    cant_B = len( np.extract(cond_1 & cond_2, lista) )
    bins[i] = cant_B
  return (prom, std, cant_min_v, bins)

def median_approx(listado, B):
  largo = len(listado)
  [prom, std, cant_min_v, bins] = median_bins(listado, B)
  acum = cant_min_v
  for i in range(B):
    acum = acum + bins[i]
    if acum >= (largo+1)/2:
      return prom - std + ( 2 * std / B)* (i+ 0.5) 
  return prom - std + ( 2 * std / B)* (B - 0.5) 


# Import the running_stats function
from helper import running_stats
# Write your median_bins_fits and median_approx_fits here:

import numpy as np
from astropy.io import fits

def median_bins_fits(listado, B):
  mean, std = running_stats(listado)
  min_v = mean - std
  ancho = 2 * std / B
  range_bins = []
  n = 0
  for i in range(B+1):
    range_bins.append(min_v + ancho * i)
  for archivo in listado:
    hdulist = fits.open(archivo)
    data = hdulist[0].data
    if n == 0:
      left_bin = np.zeros_like(data)
      bins = []
      for i in range(B):
        bins.append(np.zeros_like(data))
    n+=1
    left_bin += data < min_v
    for i in range(B):
      bins[i] += np.bitwise_and(data >= range_bins[i], data < range_bins[i+1])
    hdulist.close()
  return mean, std, left_bin, np.moveaxis(bins, 0, -1)

def median_approx_fits(listado, B):
  mean, std, left_bin, bins = median_bins_fits(listado, B)
  largo = len(listado)
  acum = left_bin
  salida = np.zeros_like(left_bin)
  for i in range(B):
    acum = acum + bins[:, :, i]
    bin_value = mean - std + (2* std / B) * (i + 0.5)
    cond = np.bitwise_and(acum >= (largo + 1) / 2, salida == 0)
    salida += cond * bin_value
  bin_value = mean - std + (2* std / B) * (B - 0.5)
  salida += (salida == 0) * bin_value
  return salida
