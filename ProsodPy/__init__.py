import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import librosa   
import pickle

def get_boundaries(fpath):
  '''Parse .cha file and extract timestamps'''

  # read in file as string
  with open(fpath) as f: 
    t = f.read()

  # split at occurences of \x15 unicode character (denote beginning/end of timestamp)
  a = t.split('\x15') 

  # get odd numbered values (in between the break chars)
  IUs = [x for i,x in enumerate(a) if i%2==1]

  # split substrings by underscore to separate beginning and end
  boundaries = [x.split('_') for x in IUs]

  # for each boundary, store time values
  a = []
  for b in boundaries:
    a.append(int(b[0]))
    a.append(int(b[1]))

  # sort extracted bounds
  boundaries = sorted(list(set(a))) 

  # convert to seconds
  boundaries = (np.array(boundaries))/1000 
  return boundaries



def moving_average(a, n=200):
  '''smooths input array with a window of n'''

  # get cumulative sum for each point
  ret = np.cumsum(a, dtype=float)
  ret[n:] = ret[n:] - ret[:-n]
  return ret[n - 1:] / n



def bound_opts(audio_array,transcribed_bounds, start):
  '''Returns true and false IU heuristic boundaries \ngiven array-type audio data and transcribed labels'''
  
  # remove sign and smooth out noise
  abs_arr = moving_average(np.abs(audio_array),n=200) 

  # take every nth value
  abs_arr = abs_arr[::200] 

  # if value is smaller than it's neighbors, append to list of potential bounds
  t_f = (abs_arr<np.roll(abs_arr,1))&(abs_arr<np.roll(abs_arr,-1)) 

  # returns indicies of potential bounds
  boundary_options = np.where(t_f==1)[0] 

  # algorithm: if removing a boundary wouldn't include meaningful sound, then remove it
  for j in range(len(boundary_options)): 
    for i,b in enumerate(boundary_options):
      if i < len(boundary_options)-2:

        # arbitrary pause threshold of the standard deviation of abs values divided by 5
        if max(abs_arr[b:boundary_options[i+2]]) < np.std(abs_arr)/5: 
          boundary_options = np.delete(boundary_options, [i+1], axis=0)

  # realign heuristic bounds to initial sample rate
  boundary_options = boundary_options*200+100 + start
  
  # to apply heuristic only
  if type(transcribed_bounds) == bool:
    if transcribed_bounds == False:
      return boundary_options

  # snap transcribed bounds to closest heuristic bound
  true_labels = []
  for t_b in transcribed_bounds:
    abs_diff = np.abs(boundary_options-t_b)
    m = np.argmin(abs_diff)
    if m <= 400:
      true_labels.append(m)

  # append true and false labels by index
  false_labels = np.array([boundary_options[i] for i in range(len(boundary_options)) if i not in true_labels])
  true_labels = np.array([boundary_options[i] for i in range(len(boundary_options)) if i in true_labels])

  return true_labels, false_labels



def gen_data(audio_array, boundaries, plot=False):
  '''generates boundary position and label for audio file and accompanying boundaries'''

  # define number of 5-second segments (and one final partial segment with remainder)
  segments = int(np.ceil(len(audio_array)/20000)) 

  # initialize X and y
  X, y = [],[]

  # get true and false boundaries for each segment
  for i in range(segments):

    # segments defined by start and stop sample number
    start, stop = (20000*i,20000*(i+1))

    # subset complete audio file to isolate chunk
    chunk = audio_array[start:stop]

    # relevant boundaries only exist with the segment
    bound_set = boundaries[np.where(np.logical_and(boundaries>=start, boundaries<=stop))[0]]

    # concatenate all segment data together
    try: #catches case where segment contains no potential boundaries
      true_labels, false_labels = bound_opts(chunk, bound_set, start)

      X += list(true_labels) #preserve order of features
      X += list(false_labels)
      y += [1]*len(true_labels) #... and labels
      y += [0]*len(false_labels)

      # plots each segment if specified
      if plot: 

        # plot audio
        plt.plot(range(start,stop),chunk)

        # plot false potential boundaries
        plt.vlines(false_labels,np.min(chunk),np.max(chunk),linewidth=2,color='r')

        # plot true potential boundaries
        plt.vlines(true_labels,np.min(chunk),np.max(chunk),linewidth=2,color='g')

        # plot unaltered timestamps
        plt.vlines(bound_set,np.min(chunk),np.max(chunk),linewidth=2,linestyle='--')
        plt.show()

    except Exception as E: # empty segment cases are ignored
      # print(E)
      pass
    
  return np.array(X), np.array(y)



def return_feats(MFCC, X, hop_length, n_frames):
  '''defines window ranges for relevant MFCC data'''
  
  # boundary data from 156 frame radius (312 total)
  R_3 = MFCC[:,int(np.floor(X/hop_length))-round(n_frames/2):int(np.floor(X/hop_length))+round(n_frames/2)]

  return R_3



def MFCC_preprocess(audio_array, boundaries, hop_length=32, n_mfcc = 12, n_fft=743, n_frames = 312, normalize=True):
  '''returns preprocessed MFCC data for an audio file'''

  # initialize empty lists to append data to
  boundary_mfcc, labels, usable_bounds = [],[],[]

  # store length of audio (in samples)
  l = len(audio_array)

  # generate MFCC (n_fft proportional to 2048 @ 22050hz)
  MFCC = librosa.feature.mfcc(audio_array, n_mfcc = n_mfcc, n_fft=n_fft, hop_length=hop_length)

  # normalize MFCC
  if normalize:
    MFCC = ((MFCC.T-MFCC.mean(axis=1))/MFCC.std(axis=1)).T

  # store X and y from data generation function
  X, y = gen_data(audio_array, boundaries, plot=False)
        
  # append data to corresponding lists
  for feat,lab in zip(X,y):

    # buffer defined as total sampled needed (number of frames * len of each frame)
    buffer = n_frames*hop_length
    
    
    # only run boundaries which can be fully evaluated
    if (feat > buffer)&(feat<(l-buffer)):
      usable_bounds.append(feat)

      # generate pre/post/boundary segments
      R_3 = return_feats(MFCC, feat, hop_length, n_frames)

      # append to lists
      boundary_mfcc.append(R_3)
      labels.append(lab)

  # convert lists to numpy arrays
  boundary_mfcc = np.array(boundary_mfcc)
  labels = np.array(labels)
  usable_bounds = np.array(usable_bounds)

  return boundary_mfcc, labels, usable_bounds

def boundary_heuristic(audio_array):
  '''return all heuristic boundaries for an input audio array'''

  # define number of 20k sample "chunks"
  segments = int(np.ceil(len(audio_array)/20000))

  # initialize empty boundary list
  bounds = []

  # for each segment
  for i in range(segments):

    # identify start and stop samples
    start, stop = (20000*i,20000*(i+1))

    # define chunk
    chunk = audio_array[start:stop]

    # run algorith
    potential_bounds = bound_opts(chunk, False,start)

    # append identified boundaries to list
    bounds.append(potential_bounds) 

  # merge boundaries together
  bounds = np.concatenate(bounds)

  return bounds

def plot_heuristic_stages(audio_array, color='black'):
  '''generates heuristic visualization on audio array'''

  # set image size and quality
  plt.rcParams["figure.figsize"] = (4,2.5)
  plt.rcParams["figure.dpi"] = 250

  # step 1: Take the absolute value of given waveform
  abs_arr = np.abs(audio_array)

  # step 2: Compute the moving average over n samples (n=0.025s=200)
  ma_arr = moving_average(abs_arr,n=200)

  #step 3: Subset every nth value
  sparse_ma_arr = ma_arr[::200]

  

  # step 4: Index local minima
  t_f = (sparse_ma_arr<np.roll(sparse_ma_arr,1))&(sparse_ma_arr<np.roll(sparse_ma_arr,-1))
  prov_boundary_options = np.where(t_f==1)[0] 

  # step 5: Potential boundaries exist where minima separate speech above given threshold
  boundary_options = prov_boundary_options
  for j in range(len(boundary_options)): 
    for i,b in enumerate(boundary_options):
      if i < len(boundary_options)-2:

        # arbitrary pause threshold of the standard deviation of abs values divided by 5
        if max(ma_arr[b:boundary_options[i+2]]) < 0.004: 
          boundary_options = np.delete(boundary_options, [i+1], axis=0)

  # create and join all six plots
  fig, axs = plt.subplots(6, 1)

  axs[0].plot(audio_array,color=color)
  axs[0].axis('off')
  axs[0].text(-.09, .3, '[in]', transform=axs[0].transAxes,
    fontsize=10, verticalalignment='bottom', fontfamily='monospace')

  axs[1].plot(abs_arr,color=color)
  axs[1].axis('off')
  axs[1].text(-.02, .3, '1', transform=axs[1].transAxes,
    fontsize=10, verticalalignment='center', fontfamily='monospace')

  axs[2].plot(ma_arr,color=color)
  axs[2].axis('off')
  axs[2].text(-.02, .3, '2', transform=axs[2].transAxes,
    fontsize=10, verticalalignment='center', fontfamily='monospace')

  axs[3].plot(sparse_ma_arr,color=color)
  axs[3].axis('off')
  axs[3].text(-.02, .35, '3', transform=axs[3].transAxes,
    fontsize=10, verticalalignment='center', fontfamily='monospace')

  axs[4].plot(sparse_ma_arr,color=color)
  axs[4].vlines(prov_boundary_options,np.min(sparse_ma_arr),np.max(sparse_ma_arr), zorder=3,linestyle=':')
  axs[4].axis('off')
  axs[4].text(-.02, .4, '4', transform=axs[4].transAxes,
    fontsize=10, verticalalignment='center', fontfamily='monospace')

  axs[5].plot(audio_array[200:],color=color)
  axs[5].vlines(boundary_options*200,np.min(audio_array),np.max(audio_array), zorder=3,linestyle='-')
  axs[5].axis('off')
  axs[5].text(-.02, .4, '5', transform=axs[5].transAxes,
    fontsize=10, verticalalignment='center', fontfamily='monospace')
