#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 18:03:04 2021

@author: 
"""

# Write your hms2dec and dms2dec functions here
def hms2dec(hour, minutes, seconds):
  return 15*(hour + minutes/60 + seconds/(60*60))

def dms2dec(degree, minutes, seconds):
  return (degree/abs(degree))*(abs(degree) + minutes/60 + seconds/(60*60))

# Write your angular_dist function here.
import numpy as np

def angular_dist(ra1, dec1, ra2, dec2):
  [ra1_rad, dec1_rad, ra2_rad, dec2_rad] = np.radians([ra1, dec1, ra2, dec2])
  a = np.sin(np.abs(dec1_rad - dec2_rad)/2)**2
  b = np.cos(dec1_rad)*np.cos(dec2_rad)*np.sin(np.abs(ra1_rad-ra2_rad)/2)**2
  d = 2*np.arcsin(np.sqrt(a+b))
  return np.degrees(d)


# http://cdsarc.u-strasbg.fr/viz-bin/Cat?J/MNRAS/384/775
# http://ssa.roe.ac.uk/allSky
  
def import_bss():
  cat = np.loadtxt('bss.dat', usecols=range(1,7))
  bss = []
  idx = 1
  for i in cat:
    bss.append((idx, hms2dec(i[0], i[1], i[2]), dms2dec(i[3], i[4], i[5])))
    idx+=1
  return bss

def import_super():
  cat = np.loadtxt('super.csv', delimiter = ',', skiprows=1, usecols=[0, 1])
  super = []
  idx = 1
  for i in cat:
    super.append((idx, i[0], i[1]))
    idx+=1
  return super


def find_closest(cat, asc, dec):
  min_dist = 0
  min_index = 0
  for pos in cat:
    (idx, asc_i, dec_i) = pos
    dist = angular_dist(asc, dec, asc_i, dec_i)
    if (min_index == 0 or dist < min_dist):
      min_dist = dist
      min_index = idx
  return (min_index, min_dist)

def crossmatch(bss, super, max_dist):
  matches = []
  no_matches = []
  for pos in bss:
    (idx_bss, asc_bss, dec_bss) = pos
    (idx_closest, closest) = find_closest(super, asc_bss, dec_bss)
    if closest < max_dist:
      matches.append((idx_bss, idx_closest, closest))
    else:
      no_matches.append(idx_bss)
  return matches, no_matches