import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
# import wandb
import os
import pickle
# import torch
import io
import warnings
import librosa
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import numpy.linalg as la
import numpy.random as rn
from scipy.stats import multivariate_normal as Gauss
import matplotlib.pyplot as plt
from time import time
from easydict import EasyDict as edict
from sklearn.cluster import SpectralClustering, AgglomerativeClustering, KMeans
from scipy.ndimage import gaussian_filter

from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate, DiarizationPurity, DiarizationCoverage
from pyannote.metrics.detection import DetectionAccuracy
from pyannote.core import notebook

import pandas as pd
from IPython.display import display, Audio


def matrixGrayScale(M, scale=255):
  """
  Maps a matrix between the range [0, 1]*scale"""
  M = M-np.min(M)*np.ones_like(M)
  M = M/np.max(M)*scale
  return M

def pltImage(img, title = '', cmap = None, size = (10,6)):
  figure, ax = plt.subplots(1)
  figure.set_size_inches(*size)
  ax.imshow(img, cmap = cmap)
  ax.set_title(title)
  return figure

def plotAudioSignals(segmentDict, y, sr, load_audio = '', size = (10,6)):
  """Masks the audio array with the segmentDict dictionary and discriminates each speaker per plot"""
  plotColor = "bgrcmykw"
  timeScale = np.linspace(0,len(y)/(sr/1000), len(y))

  figure, ax = plt.subplots(len(segmentDict.keys())+1)
  figure.set_size_inches(*size)
  figure.tight_layout(pad=4.0)

  for iax, spk in enumerate(segmentDict.keys(),2):
    ax[0].set_title(load_audio)
    ax[0].plot(timeScale, y*segmentDict[spk], alpha = 0.6, color=plotColor[(iax-2)%len(plotColor)])
    ax[0].set_ylabel('Mixed speakers')
    ax[iax-1].plot(timeScale, y*segmentDict[spk], alpha = 1, color=plotColor[(iax-2)%len(plotColor)])
    ax[iax-1].set_ylabel('Speaker {}'.format(spk))
    ax[iax-1].legend(spk, loc ="lower right")
  
  plt.xlabel('Time/[milliseconds]')
  return figure

def plotClusteredMasks(segmentDictCluster, load_audio = ''):
  """Plot the masks found after clustering"""
  plotColor = "bgrcmykw"
  figure, ax = plt.subplots(len(segmentDictCluster.keys())+1)
  figure.set_size_inches(10,6)
  figure.tight_layout(pad=4.0)

  for iax, spk in enumerate(segmentDictCluster.keys(),2):
    ax[0].set_title(load_audio)
    ax[0].plot(segmentDictCluster[spk], alpha = 0.6, color=plotColor[(iax-2)%len(plotColor)])
    ax[0].set_ylabel('Mixed speakers')

    ax[iax-1].plot(segmentDictCluster[spk], color=plotColor[(iax-2)%len(plotColor)])
    ax[iax-1].set_ylabel('Speaker {}'.format(spk))
    ax[iax-1].legend(str(spk), loc ="lower right")
  plt.xlabel('Time/[milliseconds]')
  return figure

def visualizeSegments(reference, hypothesis, load_audio = '', clusterAlg = ''):
  # Visualize the segments
  figure, ax = plt.subplots(2)
  figure.set_size_inches(15,6)
  figure.tight_layout(pad=5.0)
  notebook.plot_annotation(reference,ax=ax[0],time=True, legend=True)
  ax[0].set_title('Ground truth labels for {}'.format(load_audio))
  notebook.plot_annotation(hypothesis,ax=ax[1],time=True, legend=True)
  ax[1].set_title('Predicted labels for {}\nAlgorithm: {}'.format(load_audio, clusterAlg))
  return figure

def annotate(labelsList):
  """Converts both 'audioN_labels['segs']' and 'clusterLabels' into
  the tuples required for pyannote metrics"""
  annotation = Annotation()
  for spk, lbls in labelsList.items():
    for lbl in lbls:
      annotation[Segment(lbl[0]/1000, lbl[1]/1000)] = str(spk)
  return annotation

def diarisationMetrics(reference, hypothesis, audioLength, collar_val = 0.5):
  """audio length in milliSeconds"""
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    metrics = {}
    # for collar, skip_overlap, expName in [(0.0, False, 'NoCollarOverlap'),(collar_val, False, 'CollarOverlap'), (0.0, True, 'NoCollarNoOverlap'), (collar_val, True, 'CollarNoOverlap')]:
    for collar, skip_overlap, expName in [(collar_val, False, 'CollarOverlap'), (collar_val, True, 'CollarNoOverlap')]:
      diarizationErrorRate = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
      jaccardErrorRate = JaccardErrorRate(collar=collar, skip_overlap=skip_overlap)
      purity = DiarizationPurity(collar=collar, skip_overlap=skip_overlap)
      coverage = DiarizationCoverage(collar=collar, skip_overlap=skip_overlap)
      detectionAccuracy = DetectionAccuracy(collar=collar, skip_overlap=skip_overlap)

      print('*'*10, 'Collar: {}ms   Skip Overlap: {}'.format(collar, skip_overlap), '*'*10)
      print("DER = {0:.5f}".format(diarizationErrorRate(reference, hypothesis, uem=Segment(0, audioLength))))
      print("JER = {0:.5f}".format(jaccardErrorRate(reference, hypothesis, uem=Segment(0, audioLength))))
      print("Optimal mapping = {}".format(diarizationErrorRate.optimal_mapping(reference, hypothesis)))
      print("Purity = {0:.5f}".format(purity(reference, hypothesis, uem=Segment(0, audioLength))))
      print("Coverage = {0:.5f}".format(coverage(reference, hypothesis, uem=Segment(0, audioLength))))
      dtAcc = detectionAccuracy.compute_components(reference, hypothesis)
      # print("Detection Accuracy: FN = {:.5f}, FP = {:.5f}, TN = {:.5f}, TP = {:.5f}\n".format(dtAcc['false negative'],dtAcc['false positive'],dtAcc['true negative'],dtAcc['true positive']))

      metrics[expName]={}

      keys = ['DER','JER', 'mapping', 'purity', 'coverage', 'detectionAccuracy']
      values =  diarizationErrorRate(reference, hypothesis, detailed=True, uem=Segment(0, audioLength)),\
                jaccardErrorRate(reference, hypothesis, detailed=True, uem=Segment(0, audioLength)),\
                diarizationErrorRate.optimal_mapping(reference, hypothesis),\
                purity(reference, hypothesis, uem=Segment(0, audioLength)),\
                coverage(reference, hypothesis, uem=Segment(0, audioLength)),\
                detectionAccuracy.compute_components(reference, hypothesis)

      metrics[expName] = dict(zip(keys, list(values)))
    metrics = edict(metrics)
    return metrics