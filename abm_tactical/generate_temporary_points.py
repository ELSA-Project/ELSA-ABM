# -*- coding: utf-8 -*-

import os
import pylab as pl
from shapely.geometry import Polygon, Point #,LineString
import random as rd

#DIR='/home/profeta/Dropbox/ELSA/SIM_GER/'

#os.chdir(DIR)

main_dir = os.path.abspath(__file__)
main_dir = os.path.split(os.path.dirname(main_dir))[0]

gall_pet = lambda x :  ( 6371000.*pl.pi*x[1]/ (180.*pl.sqrt(2)) , 6371000.*pl.sqrt(2)*pl.sin(pl.pi*(x[0]/180.)) )

def get_p(file_r):
    with open(file_r,'r') as fr:
        x = fr.read()
        
    x = [map(float,a.split('\t')) for a in x.split('\n')[:-1]]
    #x=[gall_pet(map(float,a.split('\t'))) for a in x.split('\n')[:-1]]
    
    return x

def compute_temporary_points(N, bounds, save_file=main_dir + '/abm_tactical/config/bound_latlon.dat'):
    pbound = Polygon(bounds)
    gbound = Polygon(map(gall_pet, bound))
    xa, ya, xb, yb = pbound.bounds
    tmp = [[] for i in range(N)]
    i = 0

    while i<N-1:
        T = (rd.uniform(xa,xb), rd.uniform(ya,yb))
        gT = Point(gall_pet(T))

        if gT.within(gbound):
            tmp[i] = T
            i += 1

    with open(save_file, 'w') as f:
        f.write("".join([str(a[0])+'\t'+str(a[1])+'\n' for a in tmp]))

    return tmp

if __name__ == '__main__':
    pass
    # ACC = list(set([a.split('_')[6] for a in os.listdir('trajectories/M1/') if 'stat' in a and 'direct' in a]))

    # for acc in ACC:
    #     print acc
    #     bound = get_p('ELSA-ABM_' + acc + '/abm_tactical/config/bound_latlon.dat')
    #     pbound = Polygon(bound)
    #     gbound = Polygon(map(gall_pet, bound))
    #     xa, ya, xb, yb = pbound.bounds
    #     N = 50000
    #     tmp = [[] for i in range(N)]
    #     i = 0

    #     while i<N-1:
    #         T = (rd.uniform(xa,xb), rd.uniform(ya,yb))
    #         gT = Point(gall_pet(T))

    #         if gT.within(gbound):
    #             tmp[i] = T
    #             i += 1
    #         #if i>N-1: break

    #     with open('ELSA-ABM_'+acc+'/abm_tactical/config/temp_nvp.dat','w') as fw:
    #         fw.write("".join([str(a[0])+'\t'+str(a[1])+'\n' for a in tmp]))
    