#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Voronoi diagram from a list of points
# Copyright (C) 2011  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def circumcircle_bk(P1,P2,P3):
    ''' 
    Adapted from:
    http://local.wasp.uwa.edu.au/~pbourke/geometry/circlefrom3/Circle.cpp
    '''
    delta_a = P2 - P1
    delta_b = P3 - P2
    if np.abs(delta_a[0]) <= 0.000000001 and np.abs(delta_b[1]) <= 0.000000001:
        center_x = 0.5*(P2[0] + P3[0])
        center_y = 0.5*(P1[1] + P2[1])
    else:
        aSlope = delta_a[1]/delta_a[0]
        bSlope = delta_b[1]/delta_b[0]
        if np.abs(aSlope-bSlope) <= 0.000000001:
            return None
        center_x= (aSlope*bSlope*(P1[1] - P3[1]) + bSlope*(P1[0] + P2 [0]) \
                        - aSlope*(P2[0]+P3[0]) )/(2* (bSlope-aSlope) )
        print center_x,P1[0],P2[0],aSlope,P1[1],P2[1]
        print (center_x - (P1[0]+P2[0])/2)/aSlope
        center_y = -1*(center_x - (P1[0]+P2[0])/2)/aSlope +  (P1[1]+P2[1])/2;
    return center_x, center_y
    
def circumcircle(P1,P2,P3):
    ''' 
    Adapted from:
    http://local.wasp.uwa.edu.au/~pbourke/geometry/circlefrom3/Circle.cpp
    '''
    delta_a = P2 - P1
    delta_b = P3 - P2
    if np.abs(delta_a[0]) <= 0.000000001 and np.abs(delta_b[1]) <= 0.000000001:
        center_x = 0.5*(P2[0] + P3[0])
        center_y = 0.5*(P1[1] + P2[1])
    elif np.abs(delta_a[0]) > 0.000000001 and np.abs(delta_b[0]) > 0.000000001:
        aSlope = delta_a[1]/delta_a[0]
        bSlope = delta_b[1]/delta_b[0]
        if np.abs(aSlope-bSlope) <= 0.000000001:
            return None
        center_x= (aSlope*bSlope*(P1[1] - P3[1]) + bSlope*(P1[0] + P2 [0]) \
                        - aSlope*(P2[0]+P3[0]) )/(2* (bSlope-aSlope) )
        #print (center_x - (P1[0]+P2[0])/2)/aSlope
        #print aSlope,aSlope<=0.000000001
        if np.abs(aSlope)>0.000000001:
            center_y = -1*(center_x - (P1[0]+P2[0])/2)/aSlope +  (P1[1]+P2[1])/2
        else:
            center_y = -1*(center_x - (P2[0]+P3[0])/2)/bSlope +  (P2[1]+P3[1])/2
    elif np.abs(delta_a[0]) <= 0.000000001 and np.abs(delta_b[0]) > 0.000000001:
        bSlope = delta_b[1]/delta_b[0]
        center_x= - (bSlope*(P1[1] - P3[1])- (P2[0]+P3[0]) )/2
        center_y = -1*(center_x - (P2[0]+P3[0])/2)/bSlope +  (P2[1]+P3[1])/2
        #print center_x,P1[0],P2[0],aSlope,P1[1],P2[1]
        
    elif np.abs(delta_a[0]) > 0.000000001 and np.abs(delta_b[0]) <= 0.000000001:
        aSlope = delta_a[1]/delta_a[0]
        center_x= (aSlope*(P1[1] - P3[1]) + (P1[0] + P2 [0]))/2
        center_y = -1*(center_x - (P1[0]+P2[0])/2)/aSlope +  (P1[1]+P2[1])/2
        #print center_x,P1[0],P2[0],aSlope,P1[1],P2[1]
    return center_x, center_y

def voronoi(X,Y):
    P = np.zeros((X.size+4,2))
    P[:X.size,0], P[:Y.size,1] = X, Y
    # We add four points at "infinity"
    m = max(np.abs(X).max(), np.abs(Y).max())*1e5
    P[X.size:,0] = -m, -m, +m, +m
    P[Y.size:,1] = -m, +m, -m, +m
    D = matplotlib.tri.Triangulation(P[:,0],P[:,1])
    T = D.triangles
    n = T.shape[0]
    C = np.zeros((n,2))
    for i in range(n):
        C[i] = circumcircle(P[T[i,0]],P[T[i,1]],P[T[i,2]])
    X,Y = C[:,0], C[:,1]
    segments = []
    for i in range(n):
        for j in range(3):
            k = D.neighbors[i][j]
            if k != -1:
                segments.append( [(X[i],Y[i]), (X[k],Y[k])] )
    return segments
    
def voronoi_bis(X,Y):
    P = np.zeros((X.size+4,2))
    P[:X.size,0], P[:Y.size,1] = X, Y
    # We add four points at "infinity"
    m = max(np.abs(X).max(), np.abs(Y).max())*1e5
    P[X.size:,0] = -m, -m, +m, +m
    P[Y.size:,1] = -m, +m, -m, +m
    #print P
    #print
    D = matplotlib.tri.Triangulation(P[:,0],P[:,1])
    T = D.triangles
    n = T.shape[0]
    C = np.zeros((n,2))
    for i in range(n):
        C[i] = circumcircle(P[T[i,0]],P[T[i,1]],P[T[i,2]])
    X,Y = C[:,0], C[:,1]
    segments = []
    for i in range(n):
        for j in range(3):
            k = D.neighbors[i][j]
            if k != -1:
                segments.append((i,k))
    return segments,X,Y

if __name__ == '__main__':
    
    X = np.random.random(200)
    Y = np.random.random(200)
    fig = plt.figure(figsize=(10,10))
    axes = plt.subplot(1,1,1)
    plt.scatter(X,Y)
    segments = voronoi(X,Y)
    lines = matplotlib.collections.LineCollection(segments, color='0.75')
    axes.add_collection(lines)
    plt.axis([0,1,0,1])
    plt.show()