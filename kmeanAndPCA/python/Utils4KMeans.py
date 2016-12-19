import numpy as np
import matplotlib.pyplot as plt
import scipy.io #Used to load the OCTAVE *.mat files
import scipy.misc #Used to show matrix as an image
from scipy import optimize
import random

def findClosedCentroid(X, centroids):
    """
    return the indexs of the centroids for each sample in X
    """
    ids = np.zeros((X.shape[0],1)) * (-1)
    for i in xrange(X.shape[0]):
        ids[i] = findClosest(X[i],centroids)
    return ids

def findClosest(point,centroids):
    """
    find the closest one in centroids array for the given sample
    return the index of the closed one
    """
    result = np.zeros((len(centroids),1))
    for i in xrange(len(centroids)):
        result[i] = distSquared(point, centroids[i])
    return np.argmin(result,axis=0)

def distSquared(point1, point2):
    """
    return the distance of these 2 points
    """
    assert point1.shape == point2.shape
    return np.sum(np.square(point1 - point2))

def computeCentroids(X, ids):
    """
    first, according the distinguish values in ids to get the value:K
    then, make good use of K to compute the K-centroids points(mean value)
    """
    k = len(np.unique(ids))
    
    result = np.empty((k,X.shape[1]))
    #print result.shape
    for i in xrange(k):
        result[i]=np.mean(X[ids.flatten()==i],axis=0)
    return result

def calMeanK(X, centroids, maxIter=100):
    cent_history=[]
    for i in xrange(maxIter):
        cent_history.append(centroids)
        ids = findClosedCentroid(X,centroids)
        cent = computeCentroids(X,ids)
        if np.sum(cent!=centroids) == 0:
            break
        else:
            centroids = cent
    cent_history.append(cent)
    return ids,cent_history
