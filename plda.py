import os, os.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import numpy.linalg as la
import numpy.random as rn
from scipy.stats import multivariate_normal as Gauss
import matplotlib.pyplot as plt
from time import time
from easydict import EasyDict as edict
from sklearn.cluster import SpectralClustering, AgglomerativeClustering, KMeans
from sklearn_extra.cluster import KMedoids
from scipy.ndimage import gaussian_filter
from plotUtilities import *

class PLDA:
  def __init__(self,Xk_list, dim = -1, crop = False):
    self.W, self.c, self.sigmas_b, self.R_w, self.R_b, self.lam_w, self.lam_b = ddt(Xk_list)
    # print(self.lam_w.shape, self.lam_b.shape)
    if crop:
      self.dim = dim if dim < min(self.real_rank_by_svalues(self.lam_w), self.real_rank_by_svalues(self.lam_b)) and dim > 0 \
                    else min(self.real_rank_by_svalues(self.lam_w), self.real_rank_by_svalues(self.lam_b))
      print('Decorrelated embeddings will have {} dimensions'.format(self.dim))
      self.W, self.sigmas_b, self.lam_w, self.lam_b = self.W[:,:self.dim], self.sigmas_b[:self.dim], self.lam_w[:self.dim], self.lam_b[:self.dim]
    s = self.sigmas_b; s2 = s*s
    s2 = np.diag(self.W.T @ self.R_b @ self.W)
    self.psi = s2
    self.psip0 = 1/(1+s2); self.psip = s2/(1+s2)
  def predict_class_mean(self,x): return predict_class_mean(self,x)
  def same_class_probability(self,uq,u): return same_class_probability(self,uq,u)
  def classify_data(self,xq,xk_list,Probs=None): 
    return classify_data(self,xq,xk_list,Probs=Probs)
  def getRvalue(self,x,xx): return getRvalue(self,x,xx)
  def getRmatrix(self,X,XX): return getRmatrix(self,X,XX)
  def real_rank_by_svalues(self, sigmas, fraction=1-1e-5): return real_rank_by_svalues(self, sigmas, fraction)
  def decorrelate(self, X): return decorrelate(self, X)
  
  def clusterEmbeddings(self, X, algorithm = SpectralClustering, nClusters = 2, linkage = 'average', affinity = 'precomputed'): return clusterEmbeddings(self, X, algorithm, nClusters, linkage, affinity)
  def upsampleCluster (self, clusteredLabels, section_length = 4, section_overlap = 0.75): return upsampleCluster (self, clusteredLabels, section_length, section_overlap)
  def cosineSimilarityMatrix(self, X, factor = lambda C:C): return cosineSimilarityMatrix(self, X, factor)
  def normalize(self, X, function = lambda x : 1/(1+np.exp(-5*x))): return normalize(self, X, function)
  def labelFormat(self, segmentDictCluster): return labelFormat(self, segmentDictCluster)
  def segmentAudioLabels(self, audioN_labels, y, fs=8000): return segmentAudioLabels(self, audioN_labels, y, fs)
  def refineSimilarityMatrix(self, similarityMatrix): return refineSimilarityMatrix(self, similarityMatrix)

def predict_class_mean(plda,x):
  c = plda.c; W = plda.W;  A = la.pinv(W.T)
  return c + np.dot(A*plda.psip,np.dot(W.T,x-c))
def same_class_probability(plda,uq,u):
  psip = plda.psip
  g = Gauss(mean=u*psip, cov=np.diag(1.+psip))
  return g.pdf(uq)
def classify_data(plda,xq, xk_list,Probs=None):
  W = plda.W; c =  plda.c
  uq = np.dot(W.T,xq-c)
  uk_list = [np.dot(W.T,xk-c) for xk in xk_list]
  probs = np.array([plda.same_class_probability(uq,uk) for uk in tqdm(uk_list)])
  if Probs: probs *= Probs                                  
  return np.argmax(probs)
def getRvalue(plda,u,uu):
  """input: decorrelated embeddings (np.dot(W.T,x-c)) """
  W = plda.W; c =  plda.c; p = plda.psip0
  # u = np.dot(W.T,x-c)
  # uu = np.dot(W.T,xx-c)
  return 0.5*(u*p @ u + uu*p @ uu - 0.5*(u-uu)@(u-uu))
# Prof. Skarbek
# def getRmatrix(plda,X,XX):
#   xRxx = [[plda.getRvalue(x,xx) for x in X] for xx in tqdm(XX)]
#   return np.array(xRxx)

# Computes only half of the matrix (this is symmetric): CDMJ
# def getRmatrix(plda, Xp, Xq):
#   assert Xp.shape == Xq.shape, "Matrices must have the same dimensions"
#   rMatrix = np.zeros((len(Xp), len(Xp)))
#   for diag in tqdm(range(0,len(Xp))):
#     for row in range(0, len(Xq)-diag):
#       col = row + diag
#       rMatrix[col,row] = rMatrix[row,col] = plda.getRvalue(Xp[row,:], Xq[col,:])
#   return rMatrix


# ****************** Fast R - matrix https://kaldi-asr.org/doc/kaldi-math_8h_source.html
def getRmatrix(plda, X, Xb):
        """
        Computes plda affinity matrix using Loglikelihood function
        Parameters
        ----------
        X : TYPE
            X-vectors 1 X N X D
        Returns
        -------
        Affinity matrix TYPE
            1 X N X N 
        """
        M_LOG_2PI = 1.8378770664093454835606594728112
        psi = plda.psi
        mean = psi/(psi+1.0)
        mean = mean.reshape(1,-1)*X # N X D , X[0]- Train xvectors
        
        # given class computation
        variance_given = 1.0 + psi/(psi+1.0)
        logdet_given = np.sum(np.log(variance_given))
        variance_given = 1.0/variance_given
        
        # without class computation
        variance_without =1.0 + psi
        logdet_without = np.sum(np.log(variance_without))
        variance_without = 1.0/variance_without
        
        sqdiff = X #---- Test x-vectors
        nframe = X.shape[0]
        dim = X.shape[1]
        loglike_given_class = np.zeros((nframe,nframe))
        for i in range(nframe):
            sqdiff_given = sqdiff - mean[i]
            sqdiff_given  =  sqdiff_given**2
            
            loglike_given_class[:,i] = -0.5 * (logdet_given + M_LOG_2PI * dim + \
                                   np.matmul(sqdiff_given, variance_given))
        sqdiff_without = sqdiff**2
        loglike_without_class = -0.5 * (logdet_without + M_LOG_2PI * dim + \
                                     np.matmul(sqdiff_without, variance_without))
        loglike_without_class = loglike_without_class.reshape(-1,1) 
        # loglike_given_class - N X N, loglike_without_class - N X1
        loglike_ratio = loglike_given_class - loglike_without_class  # N X N
        
        return loglike_ratio
# ******************


def real_rank_by_svalues(plda, sigmas, fraction=1-1e-5):
  """This is done to compute how many dimensions to take from U_Y"""
  if fraction>=1.0: fraction = 1-1e-10
  csum = np.cumsum(sigmas); threshold = csum[-1]*fraction
  ind, = np.where(csum>threshold); rank = ind[0]+1
  return rank
def crop(plda, P, dim=-1):
  if dim == -1:
    return P
  return P[:,:dim]
def decorrelate(plda, X):
  """Ainv = W.T; Ainv(X-m)"""
  # print('Shapes', X.T.shape, plda.c.shape, plda.W.T.shape)
  temp = (X - plda.c).T
  # print(temp.shape)
  return plda.W.T @ temp
def clusterEmbeddings(plda, X, algorithm = SpectralClustering, nClusters = 2, linkage = 'average', affinity = 'precomputed'):
  """AgglomerativeClustering, KMedoids, KMeans or SpectralClustering
  X: Similarity matrix (main diagonal contains the highest scores)"""
  if algorithm == SpectralClustering:
    clustering = algorithm(n_clusters = nClusters,
                          random_state=0,
                          assign_labels="discretize",
                          affinity = 'precomputed').fit(X) # Similarity matrix
  elif algorithm == AgglomerativeClustering:
    clustering = algorithm(n_clusters = nClusters,
                          linkage = 'average',
                          affinity = 'precomputed').fit(1-X) # Distance matrix
  elif algorithm == KMedoids:
    clustering = algorithm(n_clusters = nClusters,
                            metric = 'precomputed',
                            method = 'pam', 
                            init = 'k-medoids++').fit(1-X) # Distance matrix
  elif algorithm == KMeans:
    clustering = algorithm(n_clusters=nSpeakers, random_state=0).fit(X)
  else:
    raise Exception('Clustering algorithm is not defined')
  return clustering.labels_
def upsampleCluster (plda, clusteredLabels, section_length = 4, section_overlap = 0.75):
  """Upsamples the clustered labels to match the lenght of the audio.
  returns the flatten segment cluster and the segment cluster per speaker (mask)"""
  # Upsample cluster label based on section time and section overlap
  segmentCluster = []
  for spk in clusteredLabels:
    length_appended = int(section_length*1000)
    # print(length_appended)
    if segmentCluster == []:
      segmentCluster=[spk]*length_appended
    else:
      segmentCluster[-int(section_length*1000*section_overlap):]=[spk]*length_appended # consider the overlap
  # create masks per speaker
  segmentDictCluster = dict((k, np.zeros_like(segmentCluster)) for k in set(segmentCluster))
  for spk in set(segmentCluster):
    segmentDictCluster[spk][np.where(np.array(segmentCluster)==spk)[0]] = 1
  return segmentCluster, segmentDictCluster
def cosineSimilarityMatrix(plda, X, factor = lambda x:x):
    """factor can be: lambda x:(x+1)/2"""
    C = np.inner(X,X)
    norm = np.sqrt(np.diag(C))
    C = C/np.outer(norm,norm)
    return factor(C)
def labelFormat(plda, segmentDictCluster):
  """segmentDictCluster contains an array of '0' and '1', where '1' indicates
  presence of speaker. Algorithm: if there is change from '0' to '1' this is START
  else END.
  segmentDictCluster can be obtained from PLDA.upsampleCluster
  label format: {0: [[18000, 160000], [163000, 194000], [198000, 304999]],
  #  1: [[0, 18000], [160000, 163000], [194000, 198000]]}"""
  startOn = False
  endOn = False
  clusterLabels = dict((k, []) for k in segmentDictCluster.keys())
  for (spk, seq) in segmentDictCluster.items():
    startOn = False
    endOn = False
    for i, onOff in enumerate(seq):
      if onOff > 0 and startOn == False:
        init = i
        startOn = True
      if (onOff == 0 and startOn == True) or (startOn == True and i == len(seq)-1):
        end = i
        endOn = True
      if startOn and endOn:
        clusterLabels[spk].append([init, end])
        startOn, endOn = False, False
  return clusterLabels
def segmentAudioLabels(plda, audioN_labels, y, fs=8000):
  """Matches the length of the audio array with the real labels. Returns a dictionary.
  fs: sample rate (8Khz by default)
  y: audio as a numpy array
  audioN_labels: Contains the labels of the current audio.
  audioN_labels: { 'duration': 304790,
            'segs': { 'A': [[1690, 5860],...],
                      'B': [[0, 2260],...]}}
  returns: a mask of the same length of the audio array where '1' is voice activity and '0' is no activity
            { 'A': array([0., 0., 0., ..., 0., 0., 0.]),
              'B': array([1., 1., 1., ..., 1., 1., 1.])}"""
  milliTime = np.linspace(0,len(y)/(fs/1000), len(y))
  segmentDict = {}
  for spk, timeStamp in audioN_labels['segs'].items():
    segmentDict[spk] = np.zeros_like( milliTime).copy()
    for seg in timeStamp:
      segmentDict[spk][np.where((milliTime >= seg[0]) & (milliTime <= seg[1]))] = 1
  return segmentDict

def refineSimilarityMatrix(plda, similarityMatrix):
  affMatA = similarityMatrix.copy()
  np.fill_diagonal(affMatA, 0)
  np.fill_diagonal(affMatA, np.max(affMatA, axis=1))
  affMatA_blurred = gaussian_filter(affMatA,1.3)
  affMatA_threshold = np.vstack([np.where(x>=np.percentile(x,20), x, 0.0001) for x in affMatA_blurred])
  affMatA_symmetrization = np.maximum(affMatA_threshold, affMatA_threshold.T)
  affMatA_diffusion = np.dot(affMatA_symmetrization, affMatA_symmetrization.T)
  return matrixGrayScale(affMatA_diffusion, scale =1)

def normalize(plda, X, function = lambda x : 1/(1+np.exp(-5*x))):
  """normalize the similarity matrix with 'function'
  default function: 1/(1+np.exp(-5*x))"""
  return function(X)

def ddt(Xk_list):
  # N - Dimension of the embedding; L - Number of examples; K - Number of classes
  rndLabels = []; rndTrSet = []
  for k, x in enumerate(Xk_list):
    rndLabels.append([k for _ in range(x.shape[1])]); rndTrSet.append(x)
  train_embeddings_sorted, train_labels_sorted = np.hstack(rndTrSet).T, np.hstack(rndLabels)
  # (N, L), each column is an embedding
  X, labels_sorted = train_embeddings_sorted.T, train_labels_sorted  
  N, L = X.shape; K = len(set(labels_sorted))
  assert L == len(labels_sorted), 'number of examples and number of labels are not the same'

  meanSet = np.mean(X, axis=1); X__c = (X.T - meanSet).T # (N, L)
  Xtemp = X.T.copy(); X_k_c = []; x_k_mean = []; c_mean = np.zeros(N)
  for i, k in enumerate(set(labels_sorted)):
    x_k = Xtemp[np.where(labels_sorted == k)] # (Number of examples per class, dimension of embedding)
    x_k_mean.append(np.mean(x_k, axis=0)) # mean is (dimension of embedding,)
    c_mean += (len(x_k)/L)*x_k_mean[-1] # sum(P_k*c_k)
    X_k_c.append(x_k - x_k_mean[-1]) # examples per class centered with its own mean
  C = np.vstack(x_k_mean).T # (N, K)
  C_c = (C.T - c_mean).T # (N, K)
  X_c_c = np.vstack(X_k_c).T # (N, L) embeddings centered with its class' own mean
  X_k_mean = np.vstack(x_k_mean).T # (N, K) centers of all the classes
  # Covariance Matrices
  Rw = sum([(len(xkc)/L)*(1/len(xkc) * xkc.T @ xkc) for i, xkc in enumerate(X_k_c)])
  Rb = (np.unique(labels_sorted, return_counts = True)[1]*C_c @ C_c.T)/L 
  ### Diagonalization
  lambda_Rw, E_Rw = np.linalg.eigh(Rw)
  # Sort the eigenvalues
  idx_lambda_Rw = lambda_Rw.argsort()[::-1]; lambda_Rw = lambda_Rw[idx_lambda_Rw]; E_Rw = E_Rw[:,idx_lambda_Rw]
  Sig_w_inv = np.diag(1/np.sqrt(lambda_Rw)) # (N, N)
  W_prime = E_Rw @ Sig_w_inv
  lambda_Rb, E_Rb = np.linalg.eigh(W_prime.T @ Rb @ W_prime)
  # Sort the eigenvalues
  idx_lambda_Rb = lambda_Rb.argsort()[::-1]; lambda_Rb = lambda_Rb[idx_lambda_Rb]; E_Rb = E_Rb[:,idx_lambda_Rb]
  W = W_prime @ E_Rb; Lw = W.T @ Rw @ W; Lb = W.T @ Rb @ W
  # Psi = lambda_Rb/lambda_Rw
  Psi = Lb/Lw
  # Psi = W.T @ Rb @ W
  A = np.linalg.pinv(W.T); Ainv = W.T
  phi_w = A @ A.T; phi_b = A @ np.diag(Psi) @ A.T; sigmas_b = np.sqrt(lambda_Rb) 
  return W, meanSet, sigmas_b, Rw, Rb, lambda_Rw, lambda_Rb

if __name__ == "__main__":
    L = 100# examples
    N = 512 # dimension
    X = np.random.rand(L, N)
    labels = np.random.randint(10, size = L)

    newXSet = []
    for k in set(labels):
      newXSet.append(X[np.where(labels == k)].T) # class embeddings represented as a list of list of embeddings
    plda = PLDA(newXSet)