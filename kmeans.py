# Euclidean distance
import numpy as np
from copy import deepcopy

# Euclidean distance
def distance(x,y):
    diffs = [ (x[i] - y[i])*(x[i] - y[i]) for i in range(len(x)) ]
    return np.sqrt(sum(diffs))

# Judge converged based on :
# whether the distance between old and new centroid is less than tolerance.
def converged(currcoord, prevcoord, tol = 0.0001):
    if distance(currcoord, prevcoord) < tol:
        return True
    return False

# The k means algorithm implementation
def kmeans(word_vecs, k, max_iter = 1000, tol = 0.0001 ):
    if k > len(word_vecs):
        print("More cluster than points! Please review your parameters.\n")
        return

    centroids = [0] * k
    # initialize all centroids
    for i in range(k):
        centroids[i] = word_vecs[i]

    # initialize the class
    # classes are stored in a dictionary:
    # class 1 -> []
    # class 2 -> []
    # class 3 -> []
    for j in range(max_iter):
        classes = dict(zip(range(k), [[] for _ in range(k)]))
        classlabels = dict(zip(range(k), [[] for _ in range(k)]))

        # assign initial classes
        # what we have is:
        # class 1 -> [ vector1/article1, vector2/article2 ]
        # class 2 -> [ vector3/article3, vector4/article4 ]
        # class 3 -> [... ]
        for i, features in enumerate(word_vecs):
            distances = [ distance(features, centroid) for centroid in centroids ]
            classindex = distances.index(min(distances))
            classes[classindex].append(features)
            classlabels[classindex].append(i)

        #store the previous centroid
        old_centroids = deepcopy(centroids)
        allconverged = []

        for classindex in classes:
            #recalculate centroids
            centroids[classindex] = np.average(classes[classindex], axis = 0)
            allconverged.append(converged(old_centroids[classindex], centroids[classindex], tol))

        if sum(allconverged) == len(allconverged):
            break


    return classes, classlabels
