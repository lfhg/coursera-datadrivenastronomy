#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 12:01:36 2021

@author: 
"""

# Write your crossmatch function here.
import numpy as np
import time
from astropy.coordinates import SkyCoord
from astropy import units as u

def angular_dist_rad(ra1_rad, dec1_rad, ra2_rad, dec2_rad):
  #[ra1_rad, dec1_rad, ra2_rad, dec2_rad] = np.radians([ra1, dec1, ra2, dec2])
  a = np.sin(np.abs(dec1_rad - dec2_rad)/2)**2
  b = np.cos(dec1_rad)*np.cos(dec2_rad)*np.sin(np.abs(ra1_rad-ra2_rad)/2)**2
  return 2*np.arcsin(np.sqrt(a+b))

def find_closest(cat, asc, dec):
  min_dist = 0
  min_index = None
  for i in range(len(cat)):
    [asc_i, dec_i] = cat[i]
    dist = angular_dist_rad(asc, dec, asc_i, dec_i)
    if (min_index == None or dist < min_dist):
      min_dist = dist
      min_index = i
  return (min_index, min_dist)

def crossmatch (cat1, cat2, min_dist):
  matches = []
  no_matches = []
  cat1 = np.radians(cat1)
  cat2 = np.radians(cat2)
  min_dist = np.radians(min_dist)
  start = time.perf_counter()
  for i in range(len(cat1)):
    [asc_i, bsc_i] = cat1[i]
    [j, dist_j] = find_closest(cat2, asc_i, bsc_i)
    #print(asc_i, bsc_i, dist_j, min_rad)
    if dist_j < min_dist:
      matches.append((i, j, dist_j))
    else:
      no_matches.append(i)
  stop = time.perf_counter() - start
  return matches, no_matches, stop

def crossmatch_multi (cat1, cat2, min_dist):
  matches = []
  no_matches = []
  cat1 = np.radians(cat1)
  cat2 = np.radians(cat2)
  min_dist = np.radians(min_dist)
  
  ra2s = cat2[:, 0]
  dec2s = cat2[:, 1]
  start = time.perf_counter()
  for i in range(len(cat1)):
    [asc_i, bsc_i] = cat1[i]
    dists = angular_dist_rad(asc_i, bsc_i, ra2s, dec2s)
    min_id = np.argmin(dists)
    dist_id = dists[min_id]
    if dist_id < min_dist:
      matches.append((i, min_id, dist_id))
    else:
      no_matches.append(i)
  stop = time.perf_counter() - start
  return matches, no_matches, stop

def find_closest_break(cat, asc, dec, max_break):
  min_dist = np.inf
  min_index = 0
  min_range = dec - max_break
  max_range = dec + max_break
  start_point = np.searchsorted(cat[:, 1], min_range, side='left')
  for i in range(start_point, len(cat)):
    [asc_i, dec_i] = cat[i]
    if dec_i > max_range:
      break
    dist = angular_dist_rad(asc, dec, asc_i, dec_i)
    if (dist < min_dist):
      min_dist = dist
      min_index = i
  return (min_index, min_dist)

def crossmatch_break (cat1, cat2, min_dist):
  matches = []
  no_matches = []
  cat1 = np.radians(cat1)
  cat2 = np.radians(cat2)
  cat2_sort = np.argsort(cat2[:, 1])
  cat2_ordered = cat2[cat2_sort]
  min_dist = np.radians(min_dist)
  start = time.perf_counter()
  for i in range(len(cat1)):
    [asc_i, bsc_i] = cat1[i]
    [j, dist_j] = find_closest_break(cat2_ordered, asc_i, bsc_i, min_dist)
    #print(asc_i, bsc_i, dist_j, min_rad)
    if dist_j < min_dist:
      matches.append((i, cat2_sort[j], dist_j))
    else:
      no_matches.append(i)
  stop = time.perf_counter() - start
  return matches, no_matches, stop

def crossmatch_sky (cat1, cat2, min_dist):
  matches = []
  no_matches = []
  sky_cat1 = SkyCoord(cat1*u.degree, frame='icrs')
  sky_cat2 = SkyCoord(cat2*u.degree, frame='icrs')
  #min_dist = np.radians(min_dist)
  start = time.perf_counter()
  
  closest_ids, closest_dists, closest_dists3d = sky_cat1.match_to_catalog_sky(sky_cat2)
  for i in range(len(cat1)):
    if closest_dists[i].value < min_dist:
      matches.append((i, closest_ids[i], closest_dists[i].value))
    else:
      no_matches.append(i) 
  stop = time.perf_counter() - start
  return matches, no_matches, stop