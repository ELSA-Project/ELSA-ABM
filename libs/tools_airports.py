#!/usr/bin/env python

import sys
import os
sys.path.insert(1, os.path.dirname(__file__))
from paths import path_codes, path_utilities, path_modules
sys.path.insert(1, path_modules)

from string import split
from  os.path import join as jn
from math import *
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
from scipy.stats.mstats import mquantiles
from shapely.geometry import Polygon,Point
from shapely.ops import cascaded_union
from time import time
#import matplotlib.cm as cm
import _mysql
from MySQLdb.constants import FIELD_TYPE
import numpy as np
from matplotlib.colors import LinearSegmentedColormap as lsc
from scipy.optimize import curve_fit
from descartes.patch import PolygonPatch
#import pandas as pd
import datetime

from general_tools import date_db, delay, date_human, date_st, cumulative, draw_network_and_patches, draw_zonemap, TrajConverter

__version__='5.8'

col = ['ro-','bs-','c*-','m^-','kh-','gp-','rv-','gh-','rH-','b+-','kx-','ro-','bs-','c*-','m^-','kh-','gp-','rv-','gh-','rH-','b+-',\
     'kx-','ro-','bs-','c*-','m^-','kh-','gp-','rv-','gh-','rH-','b+-','kx-']

color = ['midnightblue','red','blue','green','brown','gold','maroon','cyan','magenta','black','turquoise']
color2 = ['midnightblue','lightskyblue','forestgreen','orange','limegreen','red','yellowgreen','gold','turquoise','mediumorchid','dodgerblue','brown', 'salmon','magenta','black','plum',\
        'dimgray','purple','slategrey', 'teal','darkgreen','firebrick','darkseagreen','maroon','lightcoral','aquamarine','goldenrod','green','sandybrown','navy','cyan','royalblue','turquoise']#,'cadetblue']


with open(jn(path_codes, 'StatiEurocontrol.txt'),'r') as list_f:
    list1 = list_f.readlines()
list_countries = [split(o)[1] for o in list1]
#pattern_delay=1
#standard_delay=15 # in minutes
with open(jn(path_codes, 'aircraft.pic'),'r') as f:
    aircraft = pickle.load(f)

with open(jn(path_codes, 'companies.pic'),'r') as f:
    companies = pickle.load(f)
companies_c = companies.values()  

my_conv = { FIELD_TYPE.LONG: int, FIELD_TYPE.FLOAT: float, FIELD_TYPE.DOUBLE: float }

FABS={'LE':'SW FAB', 'LP': 'SW FAB', 'LX':'SW FAB', 'DA':'SW FAB', 'GM':'SW FAB', 'GC':'SW FAB',
      'LF': 'FAB EC', 'LS':'FAB EC', 'LN':'FAB EC',  'EB':'FAB EC', 'ED':'FAB EC', 'ET':'FAB EC', 'EH':'FAB EC', 'EL':'FAB EC',
      'LI':'Blue MED', 'LG':'Blue MED', 'LC':'Blue MED', 'LA':'Blue MED', 'LM':'Blue MED', 'LL':'Blue MED', 'OL':'Blue MED','OS':'Blue MED', 'DT':'Blue MED',
      'LO':'FAB CE', 'LK':'FAB CE', 'LJ':'FAB CE', 'LQ':'FAB CE', 'LD':'FAB CE', 'LH':'FAB CE', 'LZ':'FAB CE',
      'LR':'DANUBE', 'LB':'DANUBE', 
      'EE':'NEFAB', 'EF':'NEFAB', 'EN':'NEFAB', 'EV':'NEFAB', 'BI':'NEFAB',
      'EG':'UK-IR', 'EI':'UK-IR',
      'EK':'DK-SE','ES':'DK-SE',
      'EP':'BALTIC','EY':'BALTIC',
      'UK':'EAST', 'LU':'EAST', 'UM':'EAST', 'UU':'EAST','UL':'EAST', 'UR':'EAST',
      'UD':'TURKEY', 'LT':'TURKEY', 'UB':'TURKEY', 'UG':'TURKEY','OI':'TURKEY', 
      'LY':'SERBIA','LW':'SERBIA'}
      
ICAO={'LE':'Spain', 'LP':'Portugal', 'LX':'Gibraltar', 'DA':'Algeria', 'GM':'Morocco', 'LF':'France', 'LS':'Switzerland', 'LN':'Monaco', 'EB':'Belgium',\
        'ED':'Germany Civil', 'ET':'Germany Military', 'EH':'Netherlands', 'EL':'Luxembourg', 'LI':'Italy', 'LG':'Greece', 'LC':'Cyprus', 'LA':'Albania',\
        'LM':'Malta', 'LL':'Israel', 'OL':'Lebanon', 'OS':'Syria', 'LO':'Austria', 'LK':'Czech Republic', 'LJ':'Slovenia', 'LQ':'Bosnia and Herzegovina',\
        'LD':'Croatia', 'LH':'Hungary', 'LZ':'Slovakia', 'LR':'Romania', 'LB':'Bulgaria', 'EE':'Estonia', 'EF':'Finland', 'EN':'Norway', 'EV':'Latvia',\
        'BI':'Iceland', 'EG':'United Kingdom', 'EI':'Ireland', 'EK':'Danemark', 'ES':'Sweden', 'EP':'Poland', 'EY':'Lithuania', 'UK':'Ukraine', 'LU':'Moldova',\
        'UM':'Belarus', 'UU':'Russia', 'UL':'Russia', 'UR':'Russia', 'UD':'Armenia', 'LT':'Turkey', 'UB':'Azerbaijan', 'UG':'Georgia', 'OI':'Iran',\
        'LY':'Serbia and Montenegro', 'LW':'Macedonia'}
    


def country_of_ICAO(icao): # Useless
    return ICAO[icao]
    
def ICAO_from_country(country):
    return [k for k in ICAO.keys() if ICAO[k]==country][0]

def from_file_to_date(datetimestamp): # Changed in v2.0
    date,time=split(datetimestamp,' ')
    date=split(date,'-')
    time=split(time,':')
    return [int(date[0]),int(date[1]),int(date[2]),int(time[0]),int(time[1]),int(time[2])]

def convert(ds,output='h'): #convert a delay in seconds into minutes, hour, etc... # A REPRENDRE
    if ds!=0:
        ads=abs(ds)
    else:
        ads=1
    if output=='w':
        return (ds/ads)*(((ads/60)/60)/24)/7
    elif output=='d':
        return (ds/ads)*(((ads/60)/60)/24)
    elif output=='h':
        return (ds/ads)*((ads/60)/60)
    elif output=='m':
        return (ds/ads)*(ads/60)
    elif output=='full':
        return [(ds/ads)*(((ads/60)/60)/24)/7,(ds/ads)*(((ads/60)/60)/24),(ds/ads)*((ads/60)/60),(ds/ads)*(ads/60)- (ds/ads)*((ads/60)/60)*60]

def norm(a):
    return sqrt(sum([a[i]**2 for i in range(len(a))]))

def dist(a,b): # compute the distance in kilometers between a and b
    lat1=c_lat(a[0])   # convert minute decimal in rad
    lat2=c_lat(b[0])
    lon1=c_lat(a[1])
    lon2=c_lat(b[1])
    dz=(b[2]-a[1])*0.1*0.3048 # conversion niveau de vol (centaines de pieds) en km  # PROBLEMMMMMMMM
    #print lat1,lat2,lon1,lon2,cos(lat1)*cos(lat2)*cos(lon2-lon1)+sin(lat1)*sin(lat2)
    pouet=cos(lat1)*cos(lat2)*cos(lon2-lon1)+sin(lat1)*sin(lat2)
    if abs(pouet-1.)<10**(-8):
        arc=0.
    else:
        arc=6371*acos(pouet) # longueur de l'arc en kilometres
    return sqrt(arc**2+dz**2) #takes into account the altitude

def dist_flat_kms(a,b): # 
    """
    Compute the distance in kilometers between a and b.
    Coordinates must be in minute decimal.
    """
    a=np.array(a)
    b=np.array(b)
    lat1=c_lat(a[0])   # convert minute decimal in rad
    lat2=c_lat(b[0])
    lon1=c_lat(a[1])
    lon2=c_lat(b[1])
    #dz=(b[2]-a[1])*0.1*0.3048 # conversion niveau de vol (centaines de pieds) en km  # PROBLEMMMMMMMM
    #print lat1,lat2,lon1,lon2,cos(lat1)*cos(lat2)*cos(lon2-lon1)+sin(lat1)*sin(lat2)
    pouet=cos(lat1)*cos(lat2)*cos(lon2-lon1)+sin(lat1)*sin(lat2)
    if abs(pouet-1.)<10**(-8):
        arc=0.
    else:
        arc=6371*acos(pouet) # longueur de l'arc en kilometres
    return arc #sqrt(arc**2+dz**2) 

def gall_peters_projection(lat, lon):
    """
    lat and lon must be in degree. Gall Peters projection conserve areas.
    Return a projection in kilometers.
    """
    lat *= pi/180.
    lon *= pi/180.
    x = 6371.*lon 
    y = 2*6371.*sin(lat)

    return x, y
    
def dist_flat(a,b):
    a=np.array(a)
    b=np.array(b)
    return norm(a-b)
    
def dtw((a,b),f=dist,D_full=False):
    """
    Implementation of the Dynamic Time Warping algorithm.
    Returns either the full the distance matrix or the 
    the minimum distance.
    """
    a=np.array(a)
    b=np.array(b)
    N=len(a)
    M=len(b)
    D=[[0. for j in range(M)] for i in range(N)]
    for j in range(1,M):
        cost=f(a[0],b[j])
        D[0][j]=cost+D[0][j-1]
    for i in range(1,N):
        cost=f(a[i],b[0])
        D[i][0]=cost+D[i-1][0]
    for i in range(1,N):
        for j in range(1,M):
            cost=f(a[i],b[j])
            D[i][j]=cost+min(D[i-1][j],D[i][j-1],D[i-1][j-1])
    if not D_full:
        return D[N-1][M-1]
    else:
        return D

def dtw_path((a,b),D):
    """
    Based on distance matrix D, returns a list of tuples
    containing the indices of mapped nodes in each trajectories.
    This implementation begins from the end of the trajectory.
    """
    a=np.array(a)
    b=np.array(b)
    N=len(a)
    M=len(b)
    p=[(N-1,M-1)]
    while p[0]!=(0,0):
        n=p[0][0]
        m=p[0][1]
        if n==0:
            p.insert(0,(0,m-1))
        elif m==0:
            p.insert(0,(n-1,0))
        else:
            am=np.argmin([D[n-1][m-1],D[n-1][m],D[n][m-1]])
            if am==0:
                p.insert(0,(n-1,m-1))
            elif am==1:
                p.insert(0,(n-1,m))
            elif am==2:
                p.insert(0,(n,m-1))

    return p

def dtw_path2(a,b,D):
    """
    Based on distance matrix D, returns a list of tuples
    containing the indices of mapped nodes in each trajectories.
    This implementation begins from the start of the trajectory.
    Not used here.
    """
    a=np.array(a)
    b=np.array(b)
    N=len(a)
    M=len(b)
    p=[(0,0)]
    while p[-1]!=(N-1,M-1):
        n=p[-1][0]
        m=p[-1][1]
        if n==N-1:
            p.append((N-1,m+1))
        elif m==M-1:
            p.append((n+1,M-1))
        else:
            am=np.argmin([D[n+1][m+1],D[n+1][m],D[n][m+1]])
            if am==0:
                p.append((n+1,m+1))
            elif am==1:
                p.append((n+1,m))
            elif am==2:
                p.append((n,m+1))

    return p

def crossings(r1,r2,a,b,tt1,tt2,fid=0): # Insert point at crossings
    """
    r1, r2: list of names of navpoints m1/m3
    a, b: list of coordinates of navpoints m1/m3
    tt1, tt2: list of times (when reaching the navpoint).

    @return:
    c, d: list of coordinates of navpoints m1/m3 + crossings
    r3, r4: list of names of navpoints m1/m3 + crossings
    t3, t4: list of times (when reaching the navpoint) + crossings.
    """
    to_put=[]
    idx1,idx2=[],[]
    eps=10**(-6.)
    for i in range(len(a)-1):
        x1,y1,z1=a[i]
        x2,y2,z2=a[i+1]
        t1=delay(tt1[i][1])
        t2=delay(tt1[i+1][1])

        for j in range(len(b)-1):
            x3,y3=b[j][:2]
            x4,y4=b[j+1][:2]
            if (y2-y1)*(x4-x3)-(y4-y3)*(x2-x1)!=0.: # check if segments are not aligned.
                xc=((y3*x4-y4*x3)*(x2-x1)-(y1*x2-y2*x1)*(x4-x3))/((y2-y1)*(x4-x3)-(y4-y3)*(x2-x1)) 
                if x1+eps<xc<x2-eps and x3+eps<xc<x4-eps: # check if the x coordinate of crossing are within the two segments.
                    yc=((y2-y1)/(x2-x1))*xc + (y1*x2-y2*x1)/(x2-x1) # y coordinate of crossing
                    zc=((z2-z1)/(x2-x1))*xc + (z1*x2-z2*x1)/(x2-x1) # z coordinate of crossing
                    tc=date_st(((t2-t1)/(x2-x1))*xc + (t1*x2-t2*x1)/(x2-x1) -t1,starting_date=tt1[i][1]) # time of crossing
                    to_put.append(((xc,yc,zc),i+1,j+1,tc)) #coordinates of crossing, future index in m1, future index in m3, time of crossing.

    c,d = a[:],b[:]
    r3,r4 = r1[:],r2[:]
    t3,t4 = tt1[:],tt2[:]
    for i in range(len(to_put)-1,-1,-1):
        c.insert(to_put[i][1],to_put[i][0])
        r3.insert(to_put[i][1],('cross',to_put[i][0][2]))
        t3.insert(to_put[i][1],('cross',to_put[i][3]))
        d.insert(to_put[i][2],to_put[i][0])
        r4.insert(to_put[i][2],('cross',to_put[i][0][2]))
        t4.insert(to_put[i][2],('cross',to_put[i][3]))
    return (c,d),(r3,r4),(t3,t4)
    
def crossings_vert(r1,r2,a,b,rm1,rm2): # Insert point at crossing
    to_put=[]
    for i in range(len(a)-1):
        x1,z1=a[i]
        x2,z2=a[i+1]
        y1,y2=rm1[i][1],rm1[i+1][1]
        if x2-x1==0.:
            x2+=0.0000001
        if z2-z1==0.:
            z2+=0.0000001
        for j in range(len(b)-1):
            x3,z3=b[j]
            x4,z4=b[j+1]
            y3,y4=rm2[j][1],rm2[j+1][1]
            if x4-x3==0.:
                x4+=0.0000001
            if z4-z3==0.:
                z4+=0.0000001
            if (z2-z1)*(x4-x3)-(z4-z3)*(x2-x1)==0.:
                x4+=0.0000001
            xc=((z3*x4-z4*x3)*(x2-x1)-(z1*x2-z2*x1)*(x4-x3))/((z2-z1)*(x4-x3)-(z4-z3)*(x2-x1))
            if x1<xc<x2 and x3<xc<x4:
                zc=((z2-z1)/(x2-x1))*xc + (z1*x2-z2*x1)/(x2-x1)
                yc=((y2-y1)/(x2-x1))*xc + (y1*x2-y2*x1)/(x2-x1)
                to_put.append(([xc,yc,zc],i+1,j+1))
    c,d = a[:],b[:]
    r3,r4 = r1[:],r2[:]
    rm3,rm4 = rm1[:],rm2[:]
    #print 'in',len(rm3),len(c)
    to_put = sorted(to_put, key=lambda a:a[1])
    for i in range(len(to_put)-1,-1,-1): # ATTENTION INSERER A L'ENVERS !
        #print to_put[i][1]
        c.insert(to_put[i][1],[to_put[i][0][0],to_put[i][0][2]])
        r3.insert(to_put[i][1],('cross',to_put[i][0]))
        rm3.insert(to_put[i][1],to_put[i][0])
    #print 'in',len(rm3),len(c)
    to_put = sorted(to_put, key=lambda a:a[2])
    for i in range(len(to_put)-1,-1,-1): # ATTENTION INSERER A L'ENVERS !
        d.insert(to_put[i][2],[to_put[i][0][0],to_put[i][0][2]])
        r4.insert(to_put[i][2],('cross',to_put[i][0]))
        rm4.insert(to_put[i][2],to_put[i][0])

    return (r3,r4),(c,d),(rm3,rm4)

def sort_clock2(p):
    coin=sorted([a for a in p.keys() if a[0]=='a'],key=lambda b:b[1])
    sup_border=[p[a] for a in coin]
    coin=sorted([a for a in p.keys() if a[0]=='b'],key=lambda b:-b[1]) 
    inf_border=[p[a] for a in coin]
    return sup_border+inf_border

def local_area(a,b,p,k1,k2,debug=False):
    points={}
    points[('a',p[k1][0])]=a[p[k1][0]]
    points[('a',p[k2][0])]=a[p[k2][0]]
    points[('b',p[k2][1])]=b[p[k2][1]]
    for i in range(k2,k1-1,-1):
        points[('b',p[i][1])]=b[p[i][1]]
    if len(points)>2:
        polygon=Polygon(sort_clock2(points))
        if debug:
            print
            print abs(polygon.area),len(points)
            print p[k1][0],p[k2][0],p[k2][1],p[k1][1]
            print a[p[k1][0]], a[p[k2][0]], b[p[k2][1]],b[p[k1][1]]
            return polygon
        else:
            return abs(polygon.area)
    else:
        return 0.

def build_long(a):
    """
    Given a list of coordinates 3d a, compute the longitudinal distance since 
    beginning of trajectory. Compute also the cumulative distance
    """
    lonG = [(norm(np.array(a[i+1][:2]) - np.array(a[i][:2])), a[i+1][2]/32.8) for i in range(len(a)-1)] # Why do I divide by 32.8? Ot have altitude in meters?
    lonG.insert(0, (0., a[0][2]/32.8))
    long_cul, pouic = [], 0.
    for i in range(len(lonG)):
        pouic += lonG[i][0]
        long_cul.append(pouic)
    return lonG, long_cul

def build_long_2d(a):
    """
    2d version of previous function.
    """
    lonG = [norm(np.array(a[i+1][:2]) - np.array(a[i][:2])) for i in range(len(a)-1)]
    lonG.insert(0, 0.)
    long_cul, pouic = [], 0.
    for i in range(len(lonG)):
        pouic += lonG[i]
        long_cul.append(pouic)
    return lonG, long_cul

def point2(p,G):
    return [G.node[p[0]]['coord'][0],G.node[p[0]]['coord'][1],p[1]]
    
def area((a,b),pp, debug=False):
    p=[x for x in pp if x[0]<len(a) and x[1]<len(b)]
    i=0
    tot_cost=0.
    #print 'a,b',a,b
    pols=[]
    while i<len(p)-1 and p[i][0]<len(a) and p[i][1]<len(b):
        points={}
        points[('a',p[i][0])]=a[p[i][0]]
        if i==0 and norm(np.array(a[p[i][0]])-np.array(b[p[i][1]]))>10**(-6.):
            points[('b',p[i][1])]=b[p[i][1]]
        i+=1
        while i<len(p)-1 and norm(np.array(a[p[i][0]])-np.array(b[p[i][1]]))>10**(-6.):
            #print 'ii',i
            points[('a',p[i][0])]=a[p[i][0]]
            points[('b',p[i][1])]=b[p[i][1]]
            i+=1
        points[('a',p[i][0])]=a[p[i][0]]
        if norm(np.array(a[p[i][0]])-np.array(b[p[i][1]]))>10**(-6.):
            points[('b',p[i][1])]=b[p[i][1]]
        if len(points)>2:
            polygon=Polygon(sort_clock2(points))
            if debug:
                print 'debug2',sort_clock2(points),abs(polygon.area)
                pols.append(polygon)
            #print abs(polygon.area)
            tot_cost+=abs(polygon.area)
    if not debug:
        return float(tot_cost)
    else:
        return pols

def pouet(x,y):
    return abs(x-y)

def coin(x,y):
    return sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)

def LN(a,mu,sigma):
    ln=[]
    for x in a:
        ln.append(exp(-(log(x)-mu)**2/(2*sigma**2))/(x*sqrt(2*pi*sigma**2)))
    return np.array(ln)
    
def c_lat(lat): # convert minute decimal in radians
    return (lat/60.)*pi/180.
    
def get_data_results(ff):
    #print "Loading data from", ff
    compressed = False
    if not os.path.exists(ff):
        os.system('cd ' + split(ff,'/data')[0] + ' && tar -xzf ' + split(split(ff,'/')[-1],'pic')[0] + 'tar.gz')
        compressed = True
    with open(ff,'r') as f:
        seth = pickle.load(f)
    if compressed:
        os.system('rm ' + ff)
    #G, fl = seth.G, seth.flights
    #return G, fl

    return seth

def get_results(paras, save=True, redo_FxS=True, force=0, deviations=False, only_net=False, prefix='', from_version=None):
    """
    Used to fetch results, which can be a Set + deviations or a Set without deviations.
    """
    print
    if from_version == None:
        from_version = __version__
    rep = build_path(paras, from_version, full=False, prefix=prefix)
    print 'Searching in', rep

    if not deviations:
        seth=Set(**paras)
    else:
        seth=DevSet(paras,only_net=only_net)

    try:
        print 'I try to load the results (' + paras['mode'] + ') data_results...',
        assert not force
        seth = get_data_results(rep + '/data_results_' + paras['nodes'] + '_' + paras['type_of_deviation'] + '.pic')
        print 'done.'
    except (IOError, AssertionError):
        print 'fail.'
        print 'I try to load the set instead.'
        seth = get_set(paras, save=save, redo_FxS=redo_FxS,force=force, devset=deviations, only_net=only_net, prefix=prefix, from_version=from_version)

    return seth

def get_set(paras, save=True, redo_FxS=True, force=0, devset=False, only_net=False, prefix='', from_version=None):
    """
    This is to get a set without deviations results.
    """
    print
    # For legacy.
    if paras['data_version'] == None:
        if from_version == None:
            from_version = __version__
    else:
        from_version = paras['data_version']

    rep = build_path(paras, from_version, full=False,prefix=prefix)
    print 'Searching in', rep
    if not devset:
        seth=Set(**paras)
    else:
        seth=DevSet(paras,only_net=only_net)
    try:
        print 'I try to load the set (' + paras['mode'] + ') from pieces...',
        assert not force
        seth.import_pieces(from_version = from_version)
        print 'done.'
    except (IOError, AssertionError):
        print 'fail.'
        try :
            print 'I try to load the set (' + paras['mode'] + ') from data_init...',
            assert not force
            with open(rep + '/data_init_' + paras['nodes'] + '.pic','r') as f:
                seth_imported=pickle.load(f)
            
            # This way, we have the old object with the new methods.
            seth.G = seth_imported.G
            seth.flights = seth_imported.flights
            dic = {k:v for k,v in seth_imported.__dict__.items() if k!='G' and k!='flights'}
            for key in dic:
                setattr(seth, key, dic[key])

            del seth_imported
            print 'done.'

        except (IOError, AssertionError):
            print 'fail.'
            db=_mysql.connect(paras["address_db"],"root",paras['password_db'],"ElsaDB_A" + str(paras['airac']),conv=my_conv)
            seth.build(db,redo_FxS=redo_FxS,force=force>1)
            db.close()
            if save:
                seth.save_pieces()

    # Legacy
    if not hasattr(seth, "paras"):
        seth.paras=paras
        seth.direct = paras['direct']

    print
        
    return seth
    
def get_network(paras,save=True,my_company='',redo_FxS=True,force=False, prefix='', filtering=False, restrict=False):
    """
    To update.
    """
    rep=build_path(paras,'5.7',full=False,prefix=prefix)
    print 'Searching in', rep
    if paras['both']:
        suffix='_both'
    else:
        suffix='_' + paras['nodes']
    try:
        
        assert force==False
        print 'I try to load the network (' + paras['mode'] + ') from a stand-alone file...',

        with open(rep + '/network' + suffix + '.pic','r') as f:
            network=pickle.load(f)
        print 'done.'
    except (IOError, AssertionError):
        print 'fail.'
        try:
            print 'I try to load the network (' + paras['mode'] + ') from an old stand-alone file...',

            assert paras['mode']=='airports' or paras['mode']=='nav_sec'

            if paras['mode']=='airports':
                with open(rep + '/airport_network.pic','r') as f:
                    network=pickle.load(f)
            elif paras['mode']=='nav_sec':
                with open(rep + '/nav_sec_network.pic','r') as f:
                    network=pickle.load(f)
            # f=open(rep + '/airport_network' + suffix + '.pic','r')
            # network=pickle.load(f)
            # f.close()
            print 'done.'
        except(IOError, AssertionError):
            print 'fail'
            try :
                print 'I try to load the network (' + paras['mode'] + ') from data_init...',
                assert force==False
                #print rep + '/data_init_' + paras['nodes'] + '.pic'
                with open(rep + '/data_init_' + paras['nodes'] + '.pic','r') as f:
                    seth=pickle.load(f)
                print 'done.'
            except (IOError, AssertionError):
                print 'fail. I build the network.'
                db=_mysql.connect("localhost","root",paras['password_db'],"ElsaDB_A" + str(paras['airac']),conv=my_conv)
                seth=Set(**paras)
                seth.build(db,my_company=my_company,redo_FxS=redo_FxS)
                db.close()
                if save:
                    seth.save_pieces()
            except:
                raise
            network=seth.G
        except:
            raise
    except:
        raise

    if filtering:
        network=filter_graph(network, airports=paras['mode']=='airports')
        network.filter=True
    if paras['mode']=='navpoints' and restrict:
        network=restrict_to_ECAC(paras['airac'],network)
        network.restrict=True
        
    return network
    
def get_flights(paras,save=True,my_company='',redo_FxS=True,force=False, devset=False):
    """
    To update.
    """
    rep=build_path(paras,'5.7',full=False)

    try:
        print 'I try to load the flights from pieces (' + paras['mode'] + ')...'
        print 'from ', rep,
        assert force==False
        if paras['both']:
            suffix='_both'
        else:
            suffix='_' + paras['nodes']
        f=open(rep + '/flights' + suffix + '.pic','r')
        flights=pickle.load(f)
        f.close()
        print 'done.'
    except (IOError, AssertionError):
        print 'fail.'
        try :
            print 'I try to load the flights (' + paras['mode'] + ') from data_init...',
            assert force==False
            f=open(rep + '/data_init_' + paras['nodes'] + '.pic','r')
            seth=pickle.load(f)
            f.close()
        except (IOError, AssertionError):
            print 'fail.'
            db=_mysql.connect("localhost","root",paras['password_db'],"ElsaDB_A" + str(paras['airac']),conv=my_conv)
            if not devset:
                seth=Set(**paras)
            else:
                seth=DevSet(paras,only_net=only_net)
            seth.build(db,my_company=my_company,redo_FxS=redo_FxS)
            db.close()
            if save:
                seth.save_pieces()
        except:
            raise
        flights=seth.flights
    except:
        raise
        
    return flights

def print_out(a,text,out):
    #dep_delays=[delay(f['route_m3t'][0][1],f['route_m1t'][0][1]) for f in self.flights.values() if len(f['route_m3t'])>1 and len(f['route_m1t'])>1]
    if a!=[]:
        print >>out, text
        print >>out,'Min/Mean/StD/Max:',np.min(a),np.mean(a),np.std(a),np.max(a)

class Set(object):
    def __init__(self,**paras):
        self.paras=paras
        keys=['airac', 'nodes', 'zone', 'ext_area', 'cut_alt', 'direct', 'type_zone',\
            'timeStart', 'timeEnd', 'filtre', 'micromode', 'mode', 'verb', 'd', 'collapsed', 'gather_sectors', 'both', 'only_company']
        for k in keys:
            setattr(self, k, paras.get(k, None))
        self.version = __version__
        self.build_rep()
        self.zone_big=self.zone[:2]
        os.system('mkdir -p ' + self.rep)

    def build_rep(self):
        self.rep = build_path(self.paras, self.version, full=False) 

    def make_filter_sql(self,db):
        if self.type_zone=='EXT':
            type_sql=" "
        if self.type_zone=='SEMI':
            type_sql=""" and (left(f.icaoStart,2)='""" + self.zone_big + """'  or left(f.icaoEnd,2)='""" + self.zone_big + "') "
        elif self.type_zone=='INT':
            type_sql=""" and (left(f.icaoStart,2)='""" + self.zone_big + """' and left(f.icaoEnd,2)='""" + self.zone_big + "') "
    
        if not self.micromode:
            if date_db(self.timeStart,dateonly=True)==date_db(self.timeEnd,dateonly=True):
                time_sql= """and f.dateStartM1='""" + date_db(self.timeStart,dateonly=True) + "'"
            else:
                time_sql= """and f.dateStartM1>='""" + date_db(self.timeStart,dateonly=True) +  """' and date(f.dateStartM1)<='""" + date_db(self.timeEnd,dateonly=True) + "'"
        else:
            #if len(self.time)==1:
                time_sql= """and m1.datetimeStart>='""" + date_db(self.timeStart) + "' and m1.datetimeStart<'" + date_db(self.timeEnd) + "' and f.id=m1.flightTId "
        
        if self.only_company!=None:
            my_company = " and fg.company='" + self.only_company + "' "
        else:
            my_company = ''
        db.query("""DROP TABLE IF EXISTS FlightR""")
        if self.filtre == 'Strong':
            #query="""CREATE TEMPORARY TABLE FlightR
            query="""CREATE TABLE FlightR
                     SELECT fr.Id FROM
                         (SELECT f.id, CONCAT(f.dateStartM1, ' ', f.timeStartM1) as datetimeDep from FlightT as f, Aircraft as a, AircraftModel as am, FlightG as fg, CompanyIATACode as IATA""" + self.micro() + \
                            """WHERE f.aircraftId = a.Id and am.aircraftType = a.type and aircraftModel = 'L' and f.flightGId = fg.Id 
                            and fg.type = 'S' and fg.company!='ISS' and fg.company=IATA.company and IATA.IATACode!=''""" + time_sql + my_company + type_sql +"""  GROUP BY f.id) as fr, SegmentT_M1 as m1
            WHERE m1.heightEnd=0 and m1.flightTId=fr.id and time_to_sec(timediff(m1.datetimeEnd,fr.datetimeDep))>600"""
            db.query(query)
        elif self.filtre=='Medium':
            db.query("""CREATE TABLE FlightR
                     SELECT fr.Id FROM
                         (SELECT f.id, CONCAT(f.dateStartM1, ' ', f.timeStartM1) as datetimeDep from FlightT as f, Aircraft as a, AircraftModel as am, FlightG as fg""" + self.micro() + 
                            """WHERE f.aircraftId = a.Id and am.aircraftType = a.type and aircraftModel = 'L' and f.flightGId = fg.Id
                            and fg.type != 'M' and fg.company!='ISS'""" + time_sql + my_company + type_sql +"""  GROUP BY f.id) as fr, SegmentT_M1 as m1
            WHERE m1.heightEnd=0 and m1.flightTId=fr.id and time_to_sec(timediff(m1.datetimeEnd,fr.datetimeDep))>600""")
        elif self.filtre=='Weak':
            db.query("""CREATE TABLE FlightR
                     SELECT fr.Id FROM
                         (SELECT f.id, CONCAT(f.dateStartM1, ' ', f.timeStartM1) as datetimeDep from FlightT as f, Aircraft as a, AircraftModel as am, FlightG as fg""" + self.micro() + 
                            """WHERE f.aircraftId = a.Id and am.aircraftType = a.type and aircraftModel!= 'H' and aircraftModel!= 'G' and f.flightGId = fg.Id
                                 and fg.company!='ISS'""" + time_sql + my_company + type_sql +"""  GROUP BY f.id) as fr, SegmentT_M1 as m1
            WHERE m1.heightEnd=0 and m1.flightTId=fr.id and time_to_sec(timediff(m1.datetimeEnd,fr.datetimeDep))>600""")
        elif self.filtre=='All':
            db.query("""CREATE TABLE FlightR
                     SELECT fr.Id FROM
                         (SELECT f.id, CONCAT(f.dateStartM1, ' ', f.timeStartM1) as datetimeDep from FlightT as f, Aircraft as a, AircraftModel as am, FlightG as fg""" + self.micro() + 
                            """WHERE f.aircraftId = a.Id and am.aircraftType = a.type and aircraftModel!= 'H' and aircraftModel!= 'G' and f.flightGId = fg.Id
                                 """ + time_sql + my_company + type_sql +"""  GROUP BY f.id) as fr, SegmentT_M1 as m1
            WHERE m1.heightEnd=0 and m1.flightTId=fr.id""")
            
        db.query("""ALTER TABLE FlightR ADD INDEX pathid_index (Id)""")
    
    def micro(self):
        if self.micromode:
            return ", SegmentT_M1 as m1 "
        else:
            return " "
      
    def build_navpoints(self,db):
        if self.verb:
            print 'Selecting the navpoints (from DB)...'
        if not self.direct:
            self.Gf=nx.Graph()
        else:
            self.Gf=nx.DiGraph()
        
        # Tutti m1
        if self.nodes=='m1' or self.both:
            db.query("""DROP TABLE IF EXISTS PointR1""")
            if len(self.zone)==2 or self.zone=='ECAC':
                if self.ext_area:
                    db.query("""CREATE TEMPORARY TABLE Pouet SELECT EXTEND_BOUNDARY(0.005,boundary) as bd FROM CountryBoundary WHERE regionId='""" + self.zone + "'")
                else:
                    db.query("""CREATE TEMPORARY TABLE Pouet SELECT boundary as bd FROM CountryBoundary WHERE regionId='""" + self.zone + "'")
            elif len(self.zone)==4 or self.zone!='ECAC':
                if self.ext_area:
                    db.query("""CREATE TEMPORARY TABLE Pouet SELECT EXTEND_BOUNDARY(0.005,boundary) as bd FROM ACCBoundary WHERE accName='""" + self.zone + "'")
                else:
                    db.query("""CREATE TEMPORARY TABLE Pouet SELECT boundary as bd FROM ACCBoundary WHERE accName='""" + self.zone + "'")                
            else:
                if self.ext_area:
                    db.query("""CREATE TEMPORARY TABLE Pouet SELECT EXTEND_BOUNDARY(0.005,boundary) as bd FROM ASBoundary WHERE asName='""" + self.zone + "'")
                else:
                    db.query("""CREATE TEMPORARY TABLE Pouet SELECT boundary as bd FROM ASBoundary WHERE asName='""" + self.zone + "'")
                
            db.query("""CREATE TEMPORARY TABLE PointR1
                    SELECT distinct p.name, p.latitude, p.longitude, p.id
                     FROM FlightR as f, SegmentPoint as p, SegmentT_M1 as m1, Pouet as po
                     WHERE ((m1.pointStartId=p.id and ContainsExt(AsBinary(GeomFromText(CONCAT("LINESTRING(", p.latitude/60., " ", p.longitude/60., ")"))), AsBinary(po.bd))) \
                 or (m1.pointEndId=p.id and m1.heightEnd=0 and ContainsExt(AsBinary(GeomFromText(CONCAT("LINESTRING(", p.latitude/60., " ", p.longitude/60., ")"))), AsBinary(po.bd))))\
                 and f.id=m1.flightTId""")# and cb.regionId='""" + self.zone + "'")
            
            db.query("""ALTER TABLE PointR1 ADD INDEX pathid_index (id)""")
    
            # m1, non !
            db.query("""SELECT * FROM PointR1 where not name like 'Node%' """)
            r=db.store_result()
            nod=r.fetch_row(maxrows=0,how=1)
        
            for n in nod:
                self.Gf.add_node(n['name'],coord=[n['latitude'],n['longitude']])
                self.Gf.node[n['name']]['!']=False
                self.Gf.node[n['name']]['m1']=True
                self.Gf.node[n['name']]['m3']=False
            
            # m1, !
            db.query("""SELECT * FROM PointR1 where name like 'Node%' """)
            r=db.store_result()
            nod=r.fetch_row(maxrows=0,how=1)
            for n in nod:
                self.Gf.add_node(n['name'],coord=[n['latitude'],n['longitude']])
                self.Gf.node[n['name']]['!']=True
                self.Gf.node[n['name']]['m1']=True
                self.Gf.node[n['name']]['m3']=False
        
        # Tutti m3
        if self.nodes=='m3' or self.both:
            db.query("""DROP TABLE IF EXISTS PointR3""")
            if self.ext_area:
                db.query("""CREATE TEMPORARY TABLE PointR3
                        SELECT distinct p.name, p.latitude, p.longitude, p.id
                         FROM FlightR as f, SegmentPoint as p, SegmentT_M3 as m3, Pouet as po
                         WHERE ((m3.pointStartId=p.id and ContainsExt(AsBinary(GeomFromText(CONCAT("LINESTRING(", p.latitude/60., " ", p.longitude/60., ")"))), AsBinary(po.bd))) \
                     or (m3.pointEndId=p.id and m3.heightEnd=0 and ContainsExt(AsBinary(GeomFromText(CONCAT("LINESTRING(", p.latitude/60., " ", p.longitude/60., ")"))), AsBinary(po.bd))))\
                     and f.id=m3.flightTId""")
            else:
                db.query("""CREATE TEMPORARY TABLE PointR3
                    SELECT distinct p.name, p.latitude, p.longitude, p.id
                     FROM FlightR as f, SegmentPoint as p, SegmentT_M3 as m3, CountryBoundary as cb
                     WHERE ((m3.pointStartId=p.id and ContainsExt(AsBinary(GeomFromText(CONCAT("LINESTRING(", p.latitude/60., " ", p.longitude/60., ")"))), AsBinary(cb.boundary))) \
                 or (m3.pointEndId=p.id and m3.heightEnd=0 and ContainsExt(AsBinary(GeomFromText(CONCAT("LINESTRING(", p.latitude/60., " ", p.longitude/60., ")"))), AsBinary(cb.boundary))))\
                 and f.id=m3.flightTId and cb.regionId='""" + self.zone + "'")
            db.query("""ALTER TABLE PointR3 ADD INDEX pathid_index (id)""")
            
            # m3, non !  
            db.query("""SELECT * FROM PointR3 where not name like 'Node%' """)
            r=db.store_result()
            nod=r.fetch_row(maxrows=0,how=1)
            for n in nod:
                if not self.Gf.has_node(n['name']):
                    self.Gf.add_node(n['name'],coord=[n['latitude'],n['longitude']])
                    self.Gf.node[n['name']]['!']=False
                    self.Gf.node[n['name']]['m1']=False
                self.Gf.node[n['name']]['m3']=True
            
            # m3, !
            db.query("""SELECT * FROM PointR3 where name like 'Node%' """)
            r=db.store_result()
            nod=r.fetch_row(maxrows=0,how=1)
            for n in nod:
                if not self.Gf.has_node(n['name']):
                    self.Gf.add_node(n['name'],coord=[n['latitude'],n['longitude']])
                    self.Gf.node[n['name']]['!']=True
                    self.Gf.node[n['name']]['m1']=False
                self.Gf.node[n['name']]['m3']=True
            #print 'tot number:',len(G.nodes())
            #print 'm1 non !:',len([n for n in G.nodes() if not G.node[n]['!'] and G.node[n]['m1']])
            #print 'm1 !:',len([n for n in G.nodes() if G.node[n]['!'] and G.node[n]['m1']])
            #print 'm3 non ! non m1:',len([n for n in G.nodes() if not G.node[n]['!'] and G.node[n]['m3'] and not G.node[n]['m1']])
            #print 'm3 ! non m1:',len([n for n in G.nodes() if G.node[n]['!'] and G.node[n]['m3'] and not G.node[n]['m1']])
            #print 'Common to m1 and m3:',len([n for n in G.nodes() if G.node[n]['m3'] and G.node[n]['m1']])
        del nod
 
    def build_flights(self,db):
        if self.verb:
            print 'Fetching flights (navpoints)...'
        self.flights={}
        
        if self.nodes=='m1' or self.both:
            if self.verb:
                print 'Fetching m1 routes...'
            db.query("""DROP TABLE IF EXISTS PointR1_bis""")
            db.query("""CREATE TEMPORARY TABLE PointR1_bis
            SELECT * FROM PointR1""")
            db.query("""ALTER TABLE PointR1_bis ADD INDEX pointsid_index (name)""")
            
            query="""SELECT f.id, p.name, m1.heightStart as 'alt', m1.datetimeStart as t,\
                     m1.datetimeEnd as tt, m1.heightEnd as 'end', p2.name as name2
             FROM FlightR as f, PointR1 as p, SegmentT_M1 as m1, PointR1_bis as p2
             WHERE m1.pointStartId=p.id and  m1.pointEndId=p2.id and f.id=m1.flightTId"""
            if self.cut_alt!=0.:
                query = query + """ and m1.heightStart>""" + str(self.cut_alt) + """ and m1.heightEnd>""" + str(self.cut_alt)
            query = query + """ order by f.id,t"""
            db.query(query)
        
            r=db.store_result()
            M1=r.fetch_row(maxrows=0,how=1)

            f_out=[]
            i=0
            while i<len(M1):
                f=M1[i]
                id=f['id']
    
                route,time=[],[]
                out=False
                start=f['name']
                while i<len(M1) and M1[i]['id']==id:
                    f=M1[i]
                    if not self.Gf.has_node(f['name']):
                        out=True
                    else:
                        route.append((f['name'],f['alt']))
                        time.append((f['name'],from_file_to_date(f['t'])))
                        if M1[(i+1)%len(M1)]['id']!=id and self.Gf.has_node(f['name2']): #and float(f['end'])==0. 
                            route.append((f['name2'],f['end']))
                            time.append((f['name2'],from_file_to_date(f['tt'])))
                    i+=1
                f=M1[i-1]
                end=M1[i-1]['name']
                if out and self.Gf.has_node(start) and self.Gf.has_node(end):
                    f_out.append(id)
                else:
                    self.flights[id]={'route_m1':route,'route_m3':[],'route_m1t':time,'route_m3t':[]}
        
            del M1


        f_to_pop=[]
        if self.nodes=='m3' or self.both:
            if self.verb:
                print 'Fetching m3 routes...'
            db.query("""DROP TABLE IF EXISTS PointR3_bis""")
            db.query("""CREATE TEMPORARY TABLE PointR3_bis
            SELECT * FROM PointR3""")
            db.query("""ALTER TABLE PointR3_bis ADD INDEX pointsid_index (name)""")
            
            query="""SELECT f.id, p.name, m3.heightStart as 'alt', m3.datetimeStart as t,\
                     m3.datetimeEnd as tt, m3.heightEnd as 'end', p2.name as name2
             FROM FlightR as f, PointR3 as p, SegmentT_M3 as m3, PointR3_bis as p2
             WHERE m3.pointStartId=p.id and  m3.pointEndId=p2.id and f.id=m3.flightTId"""
            if self.cut_alt!=0.:
                query = query + """ and m3.heightStart>""" + str(self.cut_alt) + """ and m3.heightEnd>""" + str(self.cut_alt)
            query = query + """ order by f.id,t"""
            db.query(query)
            
            r=db.store_result()
            M3=r.fetch_row(maxrows=0,how=1)
            
            f_out=[]
            i=0
            pouic=0
            while i<len(M3):
                f=M3[i]
                id=f['id']
                if not id in f_out:
                    route,time=[],[]
                    out=False
                    start=f['name']
                    while i<len(M3) and M3[i]['id']==id:
                        f=M3[i]
                        if not self.Gf.has_node(f['name']):
                            out=True
                            pouic+=1
                        else:
                            route.append((f['name'],f['alt']))
                            time.append((f['name'],from_file_to_date(f['t'])))
                            if M3[(i+1)%len(M3)]['id']!=id and self.Gf.has_node(f['name2']): #and float(f['end'])==0. 
                                route.append((f['name2'],f['end']))
                                time.append((f['name2'],from_file_to_date(f['tt'])))
                        i+=1
                    end=M3[i-1]['name']
                    if out and self.Gf.has_node(start) and self.Gf.has_node(end):
                        f_to_pop.append(id)
                    else:
                        if not self.flights.has_key(id):
                            self.flights[id]={'route_m1':[],'route_m3':route,'route_m1t':[],'route_m3t':time}
                        else:
                            self.flights[id]['route_m3']=route
                            self.flights[id]['route_m3t']=time
                else:
                    i+=1
    
            del M3
        
        if self.verb:
            print 'Number of flights before cleaning:',len(self.flights)
        bl=open(jn(path_utilities, 'black_list_italy.txt'))
        bll = bl.readlines()
        bl.close()
        for l in bll:
            f_to_pop.append(int(split(l,"\n")[0]))
        for id in f_to_pop:
            if self.flights.has_key(id):
                self.flights.pop(id)
        coin=self.flights.keys()
        for fk in coin:# cleaning
            f=self.flights[fk]
            if self.both:
                if len(f['route_' + self.nodes])<2 or len(f['route_m3'])<2:
                    self.flights.pop(fk)
            else:
                if len(f['route_' + self.nodes])<2:
                    self.flights.pop(fk)
        if self.verb:
            print 'Number of flights after cleaning:',len(self.flights)
            
        if self.verb:
            print len(self.flights),'flights loaded.'
   
    def choose_d(self,d):
        if d==2:
            self.G=self.Gf
        elif d==3:
            if not self.direct:
                self.G=nx.Graph()
            else:
                self.G=nx.DiGraph()
            for f in self.flights.values():
                for nod in ['m1','m3']:
                    route=f['route_' + nod]
                    for i in range(len(route)):
                        p=route[i]
                        if not self.G.has_node(name(p)):
                            self.G.add_node(name(p),coord=self.Gf.node[p[0]]['coord'], alt=p[1], m1=nod=='m1', m3=nod=='m3')
                            self.G.node[name(p)]['!']=p[0][:4]=='Node'
                        else:
                            self.G.node[name(p)][nod]=True
                        route[i]=(name(p),p[1])
            del self.Gf
        
    def build_navpoints_network(self,d=2):#,filtre_sql,time_span_sql):
        self.choose_d(d)
        for f in self.flights.values():
            route=f['route_' + self.nodes]
            for i in range(len(route)-1):
                if not self.G.has_edge(route[i][0],route[i+1][0]):
                    self.G.add_edge(route[i][0],route[i+1][0],weight=1.)
                else:
                    self.G[route[i][0]][route[i+1][0]]['weight']+=1.

    def get_airports(self,db):
        db.query(""" select icaoId from Airport where left(icaoId,2)='""" + self.zone_big + """'""")
        r=db.store_result()
        airports1=r.fetch_row(maxrows=0,how=0)
        airports_full=[]
        for a in airports1:
            airports_full.append(a[0])
        self.airports=[a for a in airports_full if self.Gf.has_node(a)]
    
    def cleaning(self,cut=False):
        print 'Number of flights before cleaning:',len(self.flights)
        bl=open(jn(path_utilities, 'black_list_italy.txt'))
        bll = bl.readlines()
        bl.close()
        for l in bll:
            f_to_pop.append(int(split(l,"\n")[0]))
        for id in f_to_pop:
            if self.flights.has_key(id):
                self.flights.pop(id)
        coin=self.flights.keys()
        for fk in coin:# cleaning
            f=self.flights[fk]
            if len(f['route_m1'])<2 or len(f['route_m3'])<2:
                self.flights.pop(fk)
        
        print 'Number of flights after cleaning:',len(self.flights)
    
    def dis_to_border(self,airac):
        fille=open(jn(path_modules, self.zone + '_shape_' + str(airac) + '.pic'),'r')
        self.zone_shape=pickle.load(fille)
        fille.close()
        n,self.dis=0,0.
        for f in self.flights.values():
            if not f['route_m1'][0][0] in self.airports:
                p=Point(self.G.node[f['route_m1'][0][0]]['coord'][0],self.G.node[f['route_m1'][0][0]]['coord'][1])
                self.dis+=p.distance(self.zone_shape)
                n+=1
        self.dis=self.dis/float(n)
        
    # def build_sectors(self,db):
    #     self.SG = nx.Graph()

    def build_flights_sec(self,db,redo_FxS=True,experiment=False,cleaning=True):
        if self.verb:
            print 'Fetching flights (sectors)...'
        self.flights={}
       # print redo_FxS, not experiment, not self.zone=='ECAC'
       # if redo_FxS and not experiment and not self.zone=='ECAC':
       #     print 'pouet'
       #     db.query("""DROP TABLE IF EXISTS PxS""")
       #     query="""CREATE TEMPORARY TABLE PxS
       #             (SELECT p.id, asb.asName as asName, MIN(SS.minHeight) as minHeight, MAX(SS.maxHeight) as maxHeight
       #             FROM PointR1 as p, ASBoundary as asb, Sector As S, SectorSlice as SS
       #             WHERE ContainsExt(AsBinary(GeomFromText(CONCAT("LINESTRING(", (p.latitude + 0.001)/60., " ", (p.longitude + 0.001)/60., ")"))), AsBinary(asb.boundary))
       #             AND S.sectorId=asName AND SS.sectorUId=S.uniqueId
       #             GROUP BY p.id, asb.asName)
       #             UNION
       #             (SELECT p.id, asb.asName as asName, MIN(SS.minHeight) as minHeight, MAX(SS.maxHeight) as maxHeight
       #             FROM PointR3 as p, ASBoundary as asb, Sector As S, SectorSlice as SS
       #             WHERE ContainsExt(AsBinary(GeomFromText(CONCAT("LINESTRING(", (p.latitude + 0.001)/60., " ", (p.longitude + 0.001)/60., ")"))), AsBinary(asb.boundary))
       #             AND S.sectorId=asName AND SS.sectorUId=S.uniqueId
       #             GROUP BY p.id, asb.asName)
       #             ;"""
       #     db.query(query)
        #elif self.zone=='ECAC':
        print 'Using big table PxSTot1'
        db.query("""DROP TABLE IF EXISTS PxS""")
        query="""CREATE TEMPORARY TABLE PxS (SELECT * FROM PxSTot1);"""
        db.query(query)
        db.query("""ALTER TABLE PxS ADD INDEX id_index (id)""")
        #print query
        
        db.query("""SELECT PxS.asName as name, AVG(p.latitude) as latitude, AVG(p.longitude) as longitude FROM PxS, PointR1 as p WHERE p.id=PxS.id GROUP BY(PxS.asName)""")
        
        r=db.store_result()
        rr=r.fetch_row(maxrows=0,how=1)
        
        if not self.direct:
            self.Gf=nx.Graph()
        else:
            self.Gf=nx.DiGraph()
        
        for n in rr:
            self.Gf.add_node(n['name'],coord=[n['latitude'],n['longitude']])
            self.Gf.node[n['name']]['!']=False
            self.Gf.node[n['name']]['m1']=True
            self.Gf.node[n['name']]['m3']=True    # A CHANGER PLUS TARD ?
        
        if self.nodes=='m1' or self.both:
            
            redo_FxS=True
            if self.micromode or redo_FxS:
                print 'Building FxS1'
                db.query("""CALL ac_flightsM1_x_as('""" + date_db(self.timeStart) + """','""" + date_db(self.timeEnd) + """');""")
                db.query("""ALTER TABLE FxS1 ADD INDEX as_name (as_name)""")
                
           # db.query("""SELECT * FROM FxS1""")
           # r=db.store_result()
           # rr=r.fetch_row(maxrows=0,how=1)
            
            print 'Crossing FxS and PxS'
           # query="""SELECT id, ptStart, name, alt, t
           #         FROM (
           #         SELECT f.id as id, m1.pointStartId as ptStart, FxS.as_name as name, m1.heightStart as alt, m1.datetimeStart as t
           #         FROM PxS, FxS1 as FxS, FlightT as f, SegmentT_M1 as m1
           #         WHERE PxS.asName=FxS.as_name and m1.flightTId=f.id and m1.pointStartId=PxS.id AND FxS.flight_id=f.id
           #         AND m1.datetimeStart>='""" +  date_db(self.timeStart) + """' AND m1.datetimeStart<'""" + date_db(self.timeEnd) + \
           #         """' AND m1.heightStart>PxS.minHeight AND m1.heightStart<=PxS.maxHeight """
            query="""SELECT id, ptStart, name, alt, t
                    FROM (
                    SELECT FxS.flight_id as id, m1.pointStartId as ptStart, FxS.as_name as name, m1.heightStart as alt, m1.datetimeStart as t
                    FROM PxS, FxS1 as FxS, SegmentT_M1 as m1, FlightR as f
                    WHERE PxS.asName=FxS.as_name and m1.flightTId=FxS.flight_id and m1.pointStartId=PxS.id AND f.Id=FxS.flight_id
                    AND m1.datetimeStart>='""" +  date_db(self.timeStart) + """' AND m1.datetimeStart<'""" + date_db(self.timeEnd) + \
                    """' AND m1.heightStart>PxS.minHeight AND m1.heightStart<=PxS.maxHeight """
            
            if self.cut_alt!=0.:
                query = query + """ AND m1.heightStart>""" + str(self.cut_alt) + """ AND m1.heightEnd>""" + str(self.cut_alt)
                
            query=query + """ GROUP BY FxS.flight_id,m1.pointStartId ORDER BY FxS.flight_id, m1.datetimeStart) as tab GROUP BY id,name ORDER BY id, t;"""
            #print query
            db.query(query)
            #db.query("""SELECT f.id as id, FxS.as_name as name, m1.heightStart as 'alt', m1.datetimeStart as 't', m1.pointStartId
            #    FROM PxS, FxS, FlightT as f, SegmentT_M1 as m1
            #    WHERE PxS.asName=FxS.as_name and m1.flightTId=f.id and m1.pointStartId=PxS.id AND FxS.flight_id=f.id
            #    GROUP BY f.id, FxS.as_name
            #    ORDER BY f.id, m1.datetimeStart;""")
            print 'Crossing done, continuing...'
            r=db.store_result()
            M1=r.fetch_row(maxrows=0,how=1)
            
            f_out=[]
            i=0
            while i<len(M1):
                f=M1[i]
                id=f['id']
                route,time=[],[]
                out=False
                start=f['name']
                while i<len(M1) and M1[i]['id']==id:
                    f=M1[i]
                    if not self.Gf.has_node(f['name']):
                        out=True
                    else:
                        route.append((f['name'],f['alt']))
                        time.append((f['name'],from_file_to_date(f['t'])))
                        #if M1[(i+1)%len(M1)]['id']!=id and self.Gf.has_node(f['name2']): #and float(f['end'])==0. 
                        #    route.append((f['name2'],f['end']))
                        #    time.append((f['name2'],from_file_to_date(f['tt'])))
                    i+=1
                f=M1[i-1]
                end=M1[i-1]['name']
                if out and self.Gf.has_node(start) and self.Gf.has_node(end):
                    f_out.append(id)
                else:
                    self.flights[id]={'route_m1':route,'route_m3':[],'route_m1t':time,'route_m3t':[]}
        
            del M1
            print 'm1 done'
        
        f_to_pop=[]
        
        if self.nodes=='m3' or self.both:
            if self.micromode or redo_FxS:
                db.query("""CALL ac_flightsM3_x_as('""" + date_db(self.timeStart) + """','""" + date_db(self.timeEnd) + """');""")
                db.query("""ALTER TABLE FxS3 ADD INDEX as_name (as_name)""")
            
            query="""SELECT id, ptStart, name, alt, t
                    FROM (
                    SELECT f.id as id, m3.pointStartId as ptStart, FxS.as_name as name, m3.heightStart as alt, m3.datetimeStart as t
                    FROM PxS, FxS3 as FxS, FlightT as f, SegmentT_M3 as m3
                    WHERE PxS.asName=FxS.as_name and m3.flightTId=f.id and m3.pointStartId=PxS.id AND FxS.flight_id=f.id
                    AND m3.datetimeStart>=@d1 AND m3.datetimeStart<@d2
                    AND m3.heightStart >PxS.minHeight AND m3.heightStart<=PxS.maxHeight"""
            
            
            if self.cut_alt!=0.:
                query = query + """ AND m3.heightStart>""" + str(self.cut_alt) + """ AND m3.heightEnd>""" + str(self.cut_alt)
                
            query=query + """ GROUP BY f.id,m3.pointStartId ORDER BY f.id, m3.datetimeStart) as tab GROUP BY id,name ORDER BY id, t;"""
            #print query
            db.query(query)
            
            r=db.store_result()
            M3=r.fetch_row(maxrows=0,how=1)
            
            i=0
            pouic=0
            while i<len(M3):
                f=M3[i]
                id=f['id']
                if not id in f_out and self.flights.has_key(id):
                    route,time=[],[]
                    out=False
                    start=f['name']
                    while i<len(M3) and M3[i]['id']==id:
                        f=M3[i]
                        if not self.Gf.has_node(f['name']):
                            out=True
                            pouic+=1
                        else:
                            route.append((f['name'],f['alt']))
                            time.append((f['name'],from_file_to_date(f['t'])))
                            #if M3[(i+1)%len(M3)]['id']!=id and self.Gf.has_node(f['name2']): #and float(f['end'])==0. 
                            #    route.append((f['name2'],f['end']))
                            #    time.append((f['name2'],from_file_to_date(f['tt'])))
                        i+=1
                    end=M3[i-1]['name']
                    if out and self.Gf.has_node(start) and self.Gf.has_node(end):
                        f_to_pop.append(id)
                    else:
                        self.flights[id]['route_m3']=route
                        self.flights[id]['route_m3t']=time
                else:
                    i+=1
    
            del M3
        
        if cleaning:
            if self.verb:
                print 'Number of flights before cleaning:',len(self.flights)
            bl=open(jn(path_utilities , 'black_list_italy.txt'))
            bll = bl.readlines()
            bl.close()
            for l in bll:
                f_to_pop.append(int(split(l,"\n")[0]))
            for id in f_to_pop:
                if self.flights.has_key(id):
                    self.flights.pop(id)
            coin=self.flights.keys()
            for fk in coin:# cleaning
                f=self.flights[fk]
                if len(f['route_' + self.nodes])<2:# or (m3 and len(f['route_m3'])<2):
                    self.flights.pop(fk)
    
        
            if self.verb:
                print 'Number of flights after cleaning:',len(self.flights)
        else:
            print 'No cleaning!'

        if self.verb:
            print len(self.flights),'flights loaded.'
            
    def build_flights_airports(self, db):
        if self.verb:
            print 'Fetching flights (airports)...'
        db.query("""SELECT f.id as id, f.icaoStart as start, f.icaoEnd as 'end', 
                    CONCAT(f.dateStartM1, ' ' ,f.timeStartM1) as 'tS_m1', CONCAT(f.dateEndM1, ' ', f.timeEndM1) as 'tE_m1',
                    CONCAT(f.dateStartM3, ' ', f.timeStartM3) as 'tS_m3', CONCAT(f.dateEndM3, ' ', f.timeEndM3) as 'tE_m3'
                    FROM FlightR as fr, FlightT as f WHERE f.id=fr.id""")
    
        r=db.store_result()
        coin=r.fetch_row(maxrows=0,how=1)
        self.flights={n['id']:{'route':[n['start'], n['end']],'route_m1t':[n['tS_m1'], n['tE_m1']], 'route_m3t':[n['tS_m3'], n['tE_m3']]} for n in coin}
        if self.verb:
            print len(self.flights),'flights loaded.'
            
    def build_airports_network(self,db):
        if not self.direct:
            self.G=nx.Graph()
        else:
            self.G=nx.DiGraph()
        
        db.query("""DROP TABLE IF EXISTS FlightR_bis""")
        db.query("""CREATE TEMPORARY TABLE FlightR_bis SELECT * FROM FlightR""")
        db.query("""SELECT DISTINCT(ic.name) as'name', air.latitude as 'lat', air.longitude as 'lon' FROM
                    ((SELECT DISTINCT(f.icaoStart) as 'name' FROM FlightT as f, FlightR as fr WHERE f.id=fr.id)
                    UNION
                    (SELECT DISTINCT(f.icaoEnd) as 'name' FROM FlightT as f, FlightR_bis as fr WHERE f.id=fr.id)) as ic, Airport as air
                    WHERE air.icaoId=ic.name AND (ic.name like 'BI%' or ic.name like 'LF%' or ic.name like 'LG%' or ic.name like 'LD%' or ic.name like 'LE%' 
                    or ic.name like 'LB%' or ic.name like 'LC%' or ic.name like 'LA%' or ic.name like 'LM%' or ic.name like 'LJ%' 
                    or ic.name like 'LK%' or ic.name like 'LH%' or ic.name like 'LI%' or ic.name like 'LW%' or ic.name like 'LO%' 
                    or ic.name like 'LT%' or ic.name like 'LU%' or ic.name like 'LR%' or ic.name like 'LS%' or ic.name like 'LP%' 
                    or ic.name like 'LQ%' or ic.name like 'LZ%' or ic.name like 'LY%' or ic.name like 'EN%' or ic.name like 'EI%' 
                    or ic.name like 'EH%' or ic.name like 'EK%' or ic.name like 'EE%' or ic.name like 'ED%' or ic.name like 'EG%' 
                    or ic.name like 'EF%' or ic.name like 'EB%' or ic.name like 'EY%' or ic.name like 'EV%' or ic.name like 'EP%' 
                    or ic.name like 'ES%' or ic.name like 'UK%' or ic.name like 'UD%' or ic.name like 'UG%' or ic.name like 'EL%' 
                    or ic.name like 'ET%' or ic.name like 'LN%' or ic.name like 'UB%') """)  
        r=db.store_result()
        coin=r.fetch_row(maxrows=0,how=1)
        
        for a in coin:
            self.G.add_node(a['name'], coord=[a['lat']*60., a['lon']*60.])
        
        for f,v in self.flights.iteritems():
            if not self.G.has_edge(v['route'][0],v['route'][1]):
                if self.G.has_node(v['route'][0]) and self.G.has_node(v['route'][1]):
                    self.G.add_edge(v['route'][0],v['route'][1], weight=1.)
            else:
                self.G[v['route'][0]][v['route'][1]]['weight']+=1.
                
    def restrict_airports(self):
        self.G=self.G.subgraph([n for n in self.G.nodes() if n[:2]==self.zone])
            
    def build_nav_sec_network_and_flights(self,seth_nav,seth_sec):
        self.flights={}
        if self.direct:
            self.Gf=nx.DiGraph()
        else:
            self.Gf=nx.Graph()
        
        print 'Crossing navpoints and sectors...'
        i_not_found=0
        flights_not_returned=[]
        for i,f in seth_nav.flights.items():
            if seth_sec.flights.has_key(i):
                self.flights[i]={'route_m1':[], 'route_m1t':[]}
                route_nav=f['route_m1']
                route_sec=seth_sec.flights[i]['route_m1']
                time_nav=[p[1] for p in f['route_m1t']]
                time_sec=[p[1] for p in seth_sec.flights[i]['route_m1t']]
                
                for j in range(len(time_sec)):
                    t1=delay(time_sec[j])
                    if j==len(time_sec)-1:
                        t2=float("inf")
                    else:
                        t2=delay(time_sec[j+1])
                    navs=[(p,time_nav[k]) for k,p in enumerate(route_nav) if t1<=delay(time_nav[k])<t2]
                    for p,t in navs:
                        nn=p[0] + '_' + route_sec[j][0]
                        self.Gf.add_node(nn,coord=seth_nav.G.node[p[0]]['coord'],m1=seth_nav.G.node[p[0]]['m1'],\
                                    m3=seth_nav.G.node[p[0]]['m3'])
                        self.Gf.node[nn]['!']=seth_nav.G.node[p[0]]['!']
                        self.flights[i]['route_m1'].append((nn,p[1]))
                        self.flights[i]['route_m1t'].append((nn,t))

            else:
                #print 'Flight not found in sec:', i
                i_not_found+=1
                flights_not_returned.append(i)
                
        if self.verb:        
            print i_not_found, ' flights not found on ', len(seth_nav.flights)

        return flights_not_returned
                    
    def build(self,db,redo_FxS=True, save_flights=True, force=False):#,save=True,save_net=False):
        start_time = time()
        if self.verb:
            print "Building " + self.mode + " set..."
        self.make_filter_sql(db)
        
        if self.mode=='airports':
            self.build_flights_airports(db)
            self.build_airports_network(db)
          
        elif self.mode=='navpoints' or self.mode=='sectors':
            self.build_navpoints(db)
            if self.verb:
                print 'Number of navpoints',len(self.Gf)
            if self.mode=='sectors':    
                self.build_flights_sec(db,redo_FxS=redo_FxS,experiment=False)
                assert self.d==2
            else:
                self.build_flights(db)
            self.get_airports(db)
            self.build_navpoints_network(d=self.d)                        

            self.nodes_to_compute=self.G.nodes() 

        else:
            paras_nav=self.paras.copy()
            paras_nav['mode']='navpoints'

            paras_sec=self.paras.copy()
            paras_sec['cut_alt']=0.
            paras_sec['mode']='sectors'

            seth_nav=get_set(paras_nav, save=True, redo_FxS=redo_FxS, force=force)
            seth_sec=get_set(paras_sec, save=True, redo_FxS=redo_FxS, force=force)
                    
            self.flights_not_used=self.build_nav_sec_network_and_flights(seth_nav,seth_sec)
            
            self.build_navpoints_network()

            if save_flights:
                f=open(self.rep + '/nav_sec_flights_restricted.pic','w')
                pickle.dump(self.flights_restricted(),f)
                f.close()
        
        if self.verb:
            print 'Everything has been loaded in ', time() - start_time, 's'
            
    def save_pieces(self):
        if self.verb:
            print "Saving set in pieces..."
        if self.both:
            suffix='_both'
        else:
            suffix='_' + self.nodes
            
        f=open(self.rep + '/network' + suffix + '.pic', 'w')
        pickle.dump(self.G,f)
        f.close()
        
        f=open(self.rep + '/flights' + suffix + '.pic', 'w')
        pickle.dump(self.flights,f)
        f.close()
            
        f=open(self.rep + '/Set_attributes.pic', 'w')
        pickle.dump(self.get_paras(),f)
        f.close()      
        
    def get_paras(self):
        return {k:v for k,v in self.__dict__.items() if k!='G' and k!='flights'}
        
    def import_pieces(self, from_version = None):
        if from_version != None:
            self.version = from_version
            self.build_rep()

        if self.verb:
           print "Loading set in pieces..."
        if self.both:
            suffix='_both'
        else:
            suffix='_' + self.nodes
            
        #print 'Loading network...'
        f=open(self.rep + '/network' + suffix + '.pic', 'r')
        self.G=pickle.load(f)
        f.close()
        
        #print 'Loading flights...'
        f=open(self.rep + '/flights' + suffix + '.pic', 'r')
        self.flights=pickle.load(f)
        f.close()
        
        #print 'Loading attributes...'
        f=open(self.rep + '/Set_attributes.pic', 'r')
        dic=pickle.load(f)
        f.close()  
        for key in dic:
            setattr(self, key, dic[key])
        
       # if self.mode=='nav_sec':
       #     f=open(self.rep + '/Set_other_attr' + suffix + '.pic', 'r')
       #     dic=pickle.load(f)
       #     f.close()
       #     self.flights_not_used=dic['flights_not_used']
            
    def flights_restricted(self):
        return {fk:fv for fk,fv in self.flights.items() if fk not in self.flights_not_used}
    
    def point(self,p):
        return [self.G.node[p[0]]['coord'][0],self.G.node[p[0]]['coord'][1],p[1]]
        
    def basic_stats(self):
        #self.dis_to_border(self.airac)
        #if verb:
        #    print "distance to closest border for first points of the trajectories:", self.dis
        out=open(self.rep + '/basic_stats.txt','w')
        if (hasattr(self, "verb") and self.verb) or not hasattr(self, "verb"):
            print 'Computing basic stats...'
        for f in self.flights.values():
            d=0.
            pp=f['route_m1']
            if pp!=[]:
                for i in range(1,len(pp)):
                    d+=dist(self.point(pp[i]),self.point(pp[i-1]))
            f['l_m1']=d + 0.001
            d=0.
            pp=f['route_m3']
            if pp!=[]:
                for i in range(1,len(pp)):
                    d+=dist(self.point(pp[i]),self.point(pp[i-1]))
            f['l_m3']=d + 0.001
            f['geo']=dist(self.point(f['route_m1'][0]),self.point(f['route_m1'][-1]))

        l_m1=[f['l_m1'] for f in self.flights.values()]
        print >>out,"Length in m1:"
        print >>out,'Min/Mean/StD/Max:', np.min(l_m1),np.mean(l_m1),np.std(l_m1),np.max(l_m1)
        l_m3=[f['l_m3'] for f in self.flights.values()]
        print >>out,"Length in m3:"
        print >>out,'Min/Mean/StD/Max:', np.min(l_m3),np.mean(l_m3),np.std(l_m3),np.max(l_m3)
        l_m1m3=[(f['l_m3'] - f['l_m1'])/f['l_m1'] for f in self.flights.values()]
        print >>out,"Difference of length between m1 and m3 trajectories over length of m1:"
        print >>out,'Min/Mean/StD/Max:', np.min(l_m1m3),np.mean(l_m1m3),np.std(l_m1m3),np.max(l_m1m3)
        l_m1m3_abs=[abs(f['l_m3'] - f['l_m1'])/f['l_m1'] for f in self.flights.values()]
        print >>out,"Absolute difference of length between m1 and m3 trajectories over length of m1:"
        print >>out,'Min/Mean/StD/Max:', np.min(l_m1m3_abs),np.mean(l_m1m3_abs),np.std(l_m1m3_abs),np.max(l_m1m3_abs)
        l_m1geo=[(f['l_m1']-f['geo'])/f['l_m1'] for f in self.flights.values()]
        print >>out,"Difference of length between m1 and geodesic trajectories over length of m1:"
        print >>out,'Min/Mean/StD/Max:', np.min(l_m1geo),np.mean(l_m1geo),np.std(l_m1geo),np.max(l_m1geo)
        nb_seg_m1=[len(f['route_m1'])-1 for f in self.flights.values()]
        print >>out,"Number of segments in m1 trajectories:"
        print >>out,'Min/Mean/StD/Max:', np.min(nb_seg_m1),np.mean(nb_seg_m1),np.std(nb_seg_m1),np.max(nb_seg_m1)
        nb_seg_m3=[len(f['route_m3'])-1 for f in self.flights.values()]
        print >>out,"Number of segments in m3 trajectories:"
        print >>out,'Min/Mean/StD/Max:', np.min(nb_seg_m3),np.mean(nb_seg_m3),np.std(nb_seg_m3),np.max(nb_seg_m3)
        diff_seg_m3m1=[len(f['route_m3']) - len(f['route_m1']) for f in self.flights.values()]
        print >>out,"Difference of segments between m3  and m1 trajectories:"
        print >>out,'Min/Mean/StD/Max:', np.min(diff_seg_m3m1),np.mean(diff_seg_m3m1),np.std(diff_seg_m3m1),np.max(diff_seg_m3m1)
        alt_m1=[np.mean([(f['route_m1'][i][1] + f['route_m1'][i+1][1])/2.*(dist(self.point(f['route_m1'][i]),self.point(f['route_m1'][i+1]))/f['l_m1']) for i in range(len(f['route_m1'])-1)]) for f in self.flights.values()]
        print >>out,"Mean altitude in m1:"
        print >>out,'Min/Mean/StD/Max:', np.min(alt_m1),np.mean(alt_m1),np.std(alt_m1),np.max(alt_m1)
        alt_m3=[np.mean([(f['route_m3'][i][1] + f['route_m3'][i+1][1])/2.*(dist(self.point(f['route_m3'][i]),self.point(f['route_m3'][i+1]))/f['l_m3']) for i in range(len(f['route_m3'])-1)]) for f in self.flights.values()]
        print >>out,"Mean altitude in m3:"
        print >>out,'Min/Mean/StD/Max:', np.min(alt_m3),np.mean(alt_m3),np.std(alt_m3),np.max(alt_m3)
        alt_m3m1=[abs(alt_m3[i] - alt_m1[i])/alt_m1[i] for i in range(len(alt_m3))]
        print >>out,"Absolute difference of altitude bet. m3 and m1 over m1:"
        print >>out,'Min/Mean/StD/Max:', np.min(alt_m3m1),np.mean(alt_m3m1),np.std(alt_m3m1),np.max(alt_m3m1)
        
        alt_m1=[]
        for f in self.flights.values():
            for i in range(len(f['route_m1'])-1):
                alt_m1.append((f['route_m1'][i][1] + f['route_m1'][i+1][1])/2.)
        print >>out,"Altitude of segment in  m1:"
        print >>out,'Min/Mean/StD/Max:', np.min(alt_m1),np.mean(alt_m1),np.std(alt_m1),np.max(alt_m1)
        alt_m3=[]
        for f in self.flights.values():
            for i in range(len(f['route_m3'])-1):
                alt_m3.append((f['route_m3'][i][1] + f['route_m3'][i+1][1])/2.)
        if alt_m3!=[]:
            print >>out,"Altitude of segment in  m3:"
            print >>out,'Min/Mean/StD/Max:', np.min(alt_m3),np.mean(alt_m3),np.std(alt_m3),np.max(alt_m3)
        
        
        # --------------- Delays ----------------
        
        duration_flight=[delay(f['route_m1t'][-1][1],f['route_m1t'][0][1]) for f in self.flights.values() if len(f['route_m1t'])>1]
        print >>out,"Duration of flight:"
        print >>out,'Min/Mean/StD/Max:',np.min(duration_flight),np.mean(duration_flight),np.std(duration_flight),np.max(duration_flight)
        
        dep_delays=[delay(f['route_m3t'][0][1],f['route_m1t'][0][1]) for f in self.flights.values() if len(f['route_m3t'])>1 and len(f['route_m1t'])>1]
        print_out(dep_delays,"Departure delays:",out)

        ER_delays=[delay(f['route_m3t'][-1][1],f['route_m3t'][0][1]) - delay(f['route_m1t'][-1][1],f['route_m1t'][0][1]) for f in self.flights.values()\
            if len(f['route_m3t'])>1 and len(f['route_m1t'])>1]
        print_out(ER_delays,"En route delays:",out)    
        
        arr_delays=[delay(f['route_m3t'][-1][1],f['route_m1t'][-1][1]) for f in self.flights.values() if len(f['route_m3t'])>0 and len(f['route_m1t'])>0]
        print_out(arr_delays,"Arrival delays:",out) 
        
        # --------------- Network ----------------
        H = self.G.subgraph([n for n in self.G.nodes() if self.G.node[n][self.nodes]])
        degs = [H.degree(n) for n in H.nodes()]
        print >>out, "Degree"
        print >>out, "Min/Mean/Median/StD/Max:", np.min(degs),np.mean(degs),np.median(degs),np.std(degs),np.max(degs)
        strs = [H.degree(n,weight='weight') for n in H.nodes()]
        print >>out, 'Strength'
        print >>out, "Min/Mean/Median/StD/Max:", np.min(strs),np.mean(strs),np.median(strs),np.std(strs),np.max(strs)
        print >>out, 'Number of nodes'
        print >>out, len([n for n in self.nodes_to_compute if self.G.node[n][self.nodes]])
        print >>out, 'Number of edges'
        print >>out, len([e for e in self.G.edges() if e[0] in self.nodes_to_compute and e[1] in self.nodes_to_compute \
                          and self.G.node[e[0]][self.nodes] and self.G.node[e[1]][self.nodes]])
        out.close()
             
class DevSet(Set):
    def __init__(self, paras,only_net=True,q=0.):#airac,rep,extended,cut_alt,filtre,type_zone,zone,timeStart,timeEnd,n_days,nodes,type_of_deviation,period,micromode,direct=False,d=2,only_net=True,q=0.):
        super(DevSet, self).__init__(**paras)
        
        self.type_of_deviation=paras['type_of_deviation']
        self.big_filtre=paras['filtre'] + '_' + paras['type_zone'] + paras['zone']
        #self.period=paras['period']
        self.cut_alt=paras['cut_alt']
        self.only_net=only_net
        self.d=paras['d']
        self.q=q
        self.collapsed=False
        self.met=['dis','vert_dis','pos_dis','neg_dis','pos_vert_dis','neg_vert_dis','frac','delay','fork','fork2','forktobe','antifork','alt','pos_delay','neg_delay']
        
    def deviation(self,verb,do_local=True):
        def point(p):
            return [self.G.node[p[0]]['coord'][0],self.G.node[p[0]]['coord'][1],p[1]]
        if do_local:
            for n in self.G.nodes():
                self.G.node[n]['dis']=0.
                self.G.node[n]['vert_dis']=0.
                self.G.node[n]['nn']=0
                self.G.node[n]['pos_dis']=0.
                self.G.node[n]['neg_dis']=0.
                self.G.node[n]['pos_dis_nn']=0
                self.G.node[n]['neg_dis_nn']=0
                self.G.node[n]['pos_vert_dis']=0.
                self.G.node[n]['neg_vert_dis']=0.
                self.G.node[n]['pos_vert_dis_nn']=0
                self.G.node[n]['neg_vert_dis_nn']=0
                self.G.node[n]['frac']=0.
                self.G.node[n]['delay']=0.
                self.G.node[n]['fork']=0
                self.G.node[n]['fork2']=0
                self.G.node[n]['fork2_shorter']=0
                self.G.node[n]['fork2_longer']=0
                self.G.node[n]['forktobe']=0
                self.G.node[n]['antifork']=0
                self.G.node[n]['alt']=0.
                self.G.node[n]['pos_delay']=0.
                self.G.node[n]['neg_delay']=0.
                self.G.node[n]['pos_de_nn']=0
                self.G.node[n]['neg_de_nn']=0
                self.G.node[n]['nn_de']=0
        
        # --------------- Computing deviations --------------
        if verb:
            print 'Computing deviations...'
        self.type_of_dev={'direct':[],'rerouting':[],'traffic':[], 'airport':[], 'other':[],'all':[]}
        self.type_of_traj={'normal':[], 'traffic':[], 'rerouted':[],'other':[],'rare':[]}
        self.length_dev=[]
        self.areas=[]
        
        for fk in self.flights.keys(): #deviations
            f=self.flights[fk]
            

            if do_local: # Frac: Detect if the node in m1 is present in m3
                nodes_m3 = [ppp[0] for ppp in f['route_m3']]
                for pp in f['route_m1']:
                    if not pp[0] in nodes_m3:
                        self.G.node[pp[0]]['frac']+=1.
            
            if do_local: # Fork: Detect if the node in m1 is the last common node between m1 and m3.
                min_len = min(len(f['route_m1']),len(f['route_m3']))
                i = 0
                while f['route_m1'][i][0] == f['route_m3'][i][0] and i<min_len-1:
                    i += 1
                if i<min_len-1: # Should not take any action if all points are the same.
                    self.G.node[f['route_m1'][i-1][0]]['fork'] += 1 
            
            f['areas']=[[],[],[]]
            for k in self.type_of_dev.keys():
                f[k]=0
            if f['route_m1']==[]:
                print 'prob'
            if f['route_m3']!=[] and f['route_m1']!=[]:
                # Coordinates of m1 and m3 trajectories
                rm1=[point(n) for n in f['route_m1']]
                rm3=[point(n) for n in f['route_m3']]

                 # Compute the number of changes in altitude.
                ch_alt_m1, ch_alt_m3 = 0., 0.
                for i in range(len(rm1)-1):
                    if rm1[i][2]!=rm1[i+1][2]:
                        ch_alt_m1+=1

                for i in range(len(rm3)-1):
                    if rm3[i][2]!=rm3[i+1][2]:
                        ch_alt_m3+=1
                f['ch_alt'] = float(ch_alt_m3)/float(len(rm3)) - float(ch_alt_m1)/float(len(rm1))

                # ----- Horizontal deviations ---- 
                # Add so some navpoints at crossings of trajectories
                (rm1,rm3),(route_m1,route_m3),(route_m1t,route_m3t) = crossings(f['route_m1'],f['route_m3'],rm1,rm3,f['route_m1t'],f['route_m3t'])
                # Find the best mapping
                D=dtw((rm1,rm3),D_full=True)
                # Compute the list of mapped nodes
                p=dtw_path((rm1,rm3),D)
                # Compute total (horizontal) area generated by the difference between trajectories
                tip=area((rm1,rm3),p)
                
                # Area is rescaled by the length of the trajectory.
                if f['l_m1']!=0.:
                    f['dev']=tip/f['l_m1']
                else:
                    f['dev']=0.

                if do_local and self.nodes=='m1':
                    # Compute longitudinal distance and cumulative distance
                    long1, long1_cul = build_long(rm1)
                    long3, long3_cul = build_long(rm3)
                    # Delay at the beginning of the trajectory.
                    init_delay = delay(route_m3t[0][1],route_m1t[0][1])

                    i, j, i_p, previous_area = 0, 0, 0, 0.
                    finished=False
                    self.areas.append([[],[]])
                    while i<len(p)-1 and not finished:
                        while j<len(p)-1 and p[j][0]==p[i][0]: # look for next pairs of mapped nodes.
                            j+=1
                        if j==len(p)-1:
                            finished=True
                        dd = dist(rm1[p[j][0]],rm1[p[i][0]])
                        areaa=local_area(rm1,rm3,p,i,j)
                        if self.type_of_deviation=='derivative': # can compute derivative of generated area.
                            amount = (areaa - previous_area)/dd**2
                        else:
                            amount = areaa/dd
                        if long1_cul[-1]!=0:
                            f['areas'][0].append(float(long1_cul[p[i][0]])/float(long1_cul[-1])) # Fractional distance to destination
                            f['areas'][1].append(areaa)
                            f['areas'][2].append((p[i][0],p[j][0],p[i][1],p[j][1])) # Points generating the area.
                        else:
                            f['areas'][0].append(0.)
                            f['areas'][1].append(0.)
                            f['areas'][2].append((p[i][0],p[j][0],p[i][1],p[j][1]))

                        if route_m1[p[i][0]][0]!='cross': # Don't do anything if it was a point generated by the dtw.
                            self.G.node[route_m1[p[i][0]][0]]['dis']+=amount # Amount of area generated
                            self.G.node[route_m1[p[i][0]][0]]['nn']+=1
                            self.G.node[route_m1[p[i][0]][0]]['alt']+=abs(route_m1[p[i][0]][1]-route_m3[p[i][1]][1]) # Difference between altitudes in m1 and m3.
                            if amount>=0.:
                                self.G.node[route_m1[p[i][0]][0]]['pos_dis']+=amount
                                self.G.node[route_m1[p[i][0]][0]]['pos_dis_nn']+=1
                            if amount<0.:
                                self.G.node[route_m1[p[i][0]][0]]['neg_dis']+=amount
                                self.G.node[route_m1[p[i][0]][0]]['neg_dis_nn']+=1
                        previous_area=areaa
                        if self.type_of_deviation=='derivative':
                            delays = (delay(route_m3t[p[i][1]][1],route_m1t[p[i][0]][1]) - delay(route_m3t[p[i_p][1]][1],route_m1t[p[i_p][0]][1]))/dd
                        else:
                            delays = (delay(route_m3t[p[i][1]][1],route_m1t[p[i][0]][1]) - init_delay)
                        if route_m1[p[i][0]][0]!='cross':
                            self.G.node[route_m1[p[i][0]][0]]['delay']+=delays
                            self.G.node[route_m1[p[i][0]][0]]['nn_de']+=1
                            if delays>=0.:
                                self.G.node[route_m1[p[i][0]][0]]['pos_delay']+=delays
                                self.G.node[route_m1[p[i][0]][0]]['pos_de_nn']+=1
                            if delays<0.:
                                self.G.node[route_m1[p[i][0]][0]]['neg_delay']+=delays
                                self.G.node[route_m1[p[i][0]][0]]['neg_de_nn']+=1
                        i_p=i
                        i=j
                    if self.type_of_deviation=='derivative':
                        delays = (delay(route_m3t[p[i][1]][1],route_m1t[p[i][0]][1]) - delay(route_m3t[p[i_p][1]][1],route_m1t[p[i_p][0]][1]))/dd
                    else:
                        delays = delay(route_m3t[p[i][1]][1],route_m1t[p[i][0]][1]) - init_delay
                    if route_m1[p[i][0]][0]!='cross':# and delays>0.:
                        self.G.node[route_m1[p[i][0]][0]]['delay']+=delays
                        self.G.node[route_m1[p[i][0]][0]]['nn_de']+=1
                        if delays>=0.:
                            self.G.node[route_m1[p[i][0]][0]]['pos_delay']+=delays
                            self.G.node[route_m1[p[i][0]][0]]['pos_de_nn']+=1
                        if delays<0.:
                            self.G.node[route_m1[p[i][0]][0]]['neg_delay']+=delays
                            self.G.node[route_m1[p[i][0]][0]]['neg_de_nn']+=1
                 
                if do_local:
                    a = f['areas']
                    for k in self.type_of_dev.keys(): # metrics linked to trajectories
                        f[k] = 0
                    lonG,i,start = 0., 0, 0
                    aa = 0.
                    began = a[1][0]>10**(-6.) # Deviation (non-zero generated area) has already begun before the beginning of the trajectory
                    # try:
                    #     assert len(long1_cul)==len(route_m1)
                    # except:
                    #     print "List of long1_cul do not have the same length than the route in m1:", len(long1_cul), len(route_m1)
                    #     raise

                    # CAUTION here: route_m1 and f['areas'] have different length, but long1_cul has the same length than route_m1

                    while i<len(a[0])-1:
                        lonG = 0. # longitudinal distance
                        aa += a[1][i]
                        if a[1][i]==0. and a[1][i+1]>10**(-6.): # Deviation begins.
                            start=i
                            aa=0.
                            began = True
                            idx_m1 = a[2][i][0]
                            idx_m3 = a[2][i][2]
                            m1_pt = route_m1[idx_m1][0]
                            m3_pt = route_m3[idx_m3][0]
                            if m1_pt!='cross': # Don't do anything if this point has been generated by the dtw.
                                # fork2 : last point which is not generating an area.
                                self.G.node[m1_pt]['fork2'] += 1 # Why did I put this here? Maybe because it's a bit better than the way I compute fork.
                                #m3_pt = p[i][0], p[i][1]
                                if (long1_cul[-1] - long1_cul[idx_m1]) > (long3_cul[-1] - long3_cul[idx_m3]): # direct: m3 trajectory is shorter after deviation
                                    self.G.node[m1_pt]['fork2_shorter'] += 1
                                else: # rerouting: m3 trajectory is longer after deviation.
                                    self.G.node[m1_pt]['fork2_longer'] += 1

                                if i>0 and a[1][i-1]==0.:
                                    # forktobe : point before the fork.
                                    self.G.node[m1_pt]['forktobe'] += 1


                        if began and a[1][i+1]==0.: # Deviation ends
                            idx_m1 = a[2][i][0]
                            m1_pt = route_m1[idx_m1][0]
                            dis_m1, dis_m3 = 0., 0.
                            for ii in range(a[2][i][0],a[2][i][1]):
                                dis_m1 += dist(rm1[ii],rm1[ii+1])
                            for ii in range(a[2][i][2],a[2][i][3]):
                                dis_m3 += dist(rm3[ii],rm3[ii+1])
                            
                            lonG = a[0][i+1]-a[0][start]
                            self.length_dev.append(lonG)
                            for k in self.type_of_dev.keys():
                                if cond(k,start,i,f,lonG,dis_m3-dis_m1):
                                    f[k] += 1
                                    self.type_of_dev[k].append([aa,dis_m3-dis_m1])
                            self.type_of_dev['all'].append([aa,dis_m1,dis_m3-dis_m1])
                            began=False
                            if m1_pt!='cross':
                                # antifork: point which finishes a deviations.
                                self.G.node[m1_pt]['antifork'] += 1
                        i+=1
                    categorize(f)
                    self.type_of_traj[f['flag']].append(a)
    
    
                #Vertical
                rm1=[point(n) for n in f['route_m1']]
                rm3=[point(n) for n in f['route_m3']]
                long1,long1_cul=build_long(rm1)
                long3,long3_cul=build_long(rm3)
        
                (route_m1,route_m3),(long1,long3),(rm1,rm3) = crossings_vert(f['route_m1'],f['route_m3'],long1,long3,rm1,rm3)
                D=dtw((rm1,rm3),D_full=True)
                p=dtw_path((rm1,rm3),D)
                top=area((long1,long3),p)
                if f['l_m1']!=0.:
                    f['vert_dev']=top/f['l_m1']
                else:
                    f['vert_dev']=0.
        
                if do_local and self.nodes=='m1':
                    i, j, i_p, previous_vert_area = 0, 0, 0, 0.
                    finished=False
                    self.areas.append([[],[]])
                    while i<len(p) and not finished:
                        while j<len(p) and p[j][0]==p[i][0]:
                            #print j
                            j+=1
                        if j==len(p):
                            finished=True
                            j+=-1
                        dd=(dist(rm1[p[j][0]],rm1[p[i][0]]) + dist(rm1[p[i][0]],rm1[p[i_p][0]]))/2.
                        vert_area=local_area(long1,long3,p,i,j)
                        if self.type_of_deviation=='derivative':
                            amount = (vert_area - previous_vert_area)/dd**2
                        else:
                            amount = vert_area/dd
                        previous_vert_area=vert_area
                        if route_m1[p[i][0]][0]!='cross':
                            self.G.node[route_m1[p[i][0]][0]]['vert_dis']+=amount
                            if amount>=0.:
                                self.G.node[route_m1[p[i][0]][0]]['pos_vert_dis']+=amount
                                self.G.node[route_m1[p[i][0]][0]]['pos_vert_dis_nn']+=1
                            if amount<0.:
                                self.G.node[route_m1[p[i][0]][0]]['neg_vert_dis']+=amount
                                self.G.node[route_m1[p[i][0]][0]]['neg_vert_dis_nn']+=1
                        i_p=i
                        i=j 
        
        if do_local:
            for n in self.G.nodes():
                if self.G.degree(n,weight='weight')!=0.:
                    self.G.node[n]['frac']=self.G.node[n]['frac']/self.G.degree(n,weight='weight')
                    self.G.node[n]['fork']=self.G.node[n]['fork']/self.G.degree(n,weight='weight')
                    self.G.node[n]['fork2']=self.G.node[n]['fork2']/self.G.degree(n,weight='weight')
                    self.G.node[n]['fork2_shorter']=self.G.node[n]['fork2_shorter']/self.G.degree(n,weight='weight')
                    self.G.node[n]['fork2_longer']=self.G.node[n]['fork2_longer']/self.G.degree(n,weight='weight')
                    self.G.node[n]['forktobe']=self.G.node[n]['forktobe']/self.G.degree(n,weight='weight')
                    self.G.node[n]['antifork']=self.G.node[n]['antifork']/self.G.degree(n,weight='weight')
                if self.G.node[n]['nn']!=0:
                    self.G.node[n]['dis']=self.G.node[n]['dis']/float(self.G.node[n]['nn'])
                    self.G.node[n]['vert_dis']=self.G.node[n]['vert_dis']/float(self.G.node[n]['nn'])
                    
                    self.G.node[n]['alt'] = self.G.node[n]['alt']/float(self.G.node[n]['nn'])
    
                    self.G.node[n]['pos_dis']=self.G.node[n]['pos_dis']/float(self.G.node[n]['nn'])
                    self.G.node[n]['neg_dis']=abs(self.G.node[n]['neg_dis'])/float(self.G.node[n]['nn'])
                    self.G.node[n]['pos_vert_dis']=self.G.node[n]['pos_vert_dis']/float(self.G.node[n]['nn'])
                    self.G.node[n]['neg_vert_dis']=abs(self.G.node[n]['neg_vert_dis'])/float(self.G.node[n]['nn'])
                if self.G.node[n]['nn_de']!=0:
                    self.G.node[n]['delay']=self.G.node[n]['delay']/float(self.G.node[n]['nn_de'])
                    self.G.node[n]['pos_delay']=self.G.node[n]['pos_delay']/float(self.G.node[n]['nn_de'])
                    self.G.node[n]['neg_delay']=abs(self.G.node[n]['neg_delay'])/float(self.G.node[n]['nn_de'])
    
    def build_routes(self,flights):
        return [f['route_m1'] for f in flights.values()],[f['route_m3'] for f in flights.values()]

    def save(self, verb, compress=True):  # To save basic and advanced results.
        if verb:
            print 'Saving Data...'
        self.rep_save = build_path(self.paras, __version__, full=False)
        stri='data_results_' + self.nodes + '_' + self.type_of_deviation
        #if self.only_company!=None:
        #    stri += self.only_company
        with open(self.rep_save + '/' + stri + '.pic','w') as res:
            pickle.dump(self,res)

        #if verb:
        #    print "Saved in", self.rep_save + '/' + stri + '.pic'

        if compress:
            os.system('cd ' + self.rep_save +\
                      " && tar -czf " + stri +'.tar.gz ' + stri +'.pic')
            os.system('rm ' + self.rep_save + '/' + stri +'.pic')
        
    def import_net_metrics(self,Dev2):
        if self.d!=Dev2.d:
            'ERROR: not the same dimension'
        self.nodes_to_compute=self.G.nodes()
        self.G=Dev2.G
    
    def global_metrics(self,do_local=True):
        if do_local:
            for m in self.met:
                self.G.graph[m] = np.mean([self.G.node[n][m] for n in self.G.nodes() if self.G.degree(n,weight='weight')!=0])
        
        self.G.graph['str'] = np.mean([self.G.degree(n,weight='weight') for n in self.G.nodes()])
        self.G.graph['deg'] = np.mean([self.G.degree(n) for n in self.G.nodes()])
        #self.G.graph['nb_nav'] = len(self.G.nodes())
        #self.G.graph['f'] = np.mean([self.G.degree(n) for n in self.G.nodes()])

    def collapse(self,list_of_files,compressed=False):
        self.G.remove_nodes_from(self.G.nodes())
        self.collapsed=True
        i=0
        for f in list_of_files:
            if not os.path.exists(f):
                os.system('cd ' + split(f,'/data')[0] + ' && tar -xzf ' + split(split(f,'/')[-1],'pic')[0] + 'tar.gz && cd ../../..')
            ff=open(f,'r')
            s=pickle.load(ff)
            ff.close()
            if compressed:
                os.system('rm ' + f)
            g,fl=s.G,s.flights
            print 'nodes:',len(g.nodes())
            bc = bet(g,self.nodes,fl,self.d,False)
            for n in g.nodes():
                self.G.add_node(n + str(i))
                self.G.node[n + str(i)]=g.node[n]
                self.G.node[n + str(i)]['bc']=bc[n]
            for e in g.edges():
                self.G[e[0] + str(i)][e[1] + str(i)] = g[e[0]][e[1]]
            i+=1
    
    def collapse_bidon(self,list_of_files,compressed=False):
        self.G.remove_nodes_from(self.G.nodes())
        self.collapsed=True
        i=0
        for f in list_of_files:
            if not os.path.exists(f):
                print f
                os.system('cd ' + split(f,'/data')[0] + ' && tar -xvzf ' + split(split(f,'/')[-1],'pic')[0] + 'tar.gz && cd ../../..')
            ff=open(f,'r')
            #s=pickle.load(ff)
            ff.close()
            if compressed:
                os.system('rm ' + f)
            #g,fl=s.G,s.flights
            #bc = bet(g,self.nodes,fl,self.d,False)
            #for n in g.nodes():
            #    self.G.add_node(n + str(i))
            #    self.G.node[n + str(i)]=g.node[n]
            #    self.G.node[n + str(i)]['bc']=bc[n]
            #for e in g.edges():
            #    self.G[e[0] + str(i)][e[1] + str(i)] = g[e[0]][e[1]]
            i+=1
                
# def find_mapped_node(m1_pt, dtw_tuples):
#     i=0
#     while i<len(dtw_tuples) and dtw_tuples[i][0]!=m1_pt:
#         i+=1
#     try:
#         assert i<len(dtw_tuples)
#     except:
#         print "Could not find point:", m1_pt, "in the mapping:", dtw_tuples
#         raise
#     return dtw_tuples[i][1]

def name((nav,alt)):
    return nav + str(int(float(alt)/10.)) + '0'

def adapt_shape_to_map(shape,m):
    coords=list(shape.exterior.coords)
    #print 'b',[(c[1],c[0]) for c in coords][:10]
    #print
    #print m([c[1]/60. for c in coords],[c[0]/60. for c in coords])
    x,y=m([c[1] for c in coords],[c[0] for c in coords])
    #print 'a',x[:10],y[:10]
    return Polygon([(x[i],y[i]) for i in range(len(x)-1)])
    
def adapt_shape_to_map2(shape,m):
    coords=list(shape.exterior.coords)
    #print 'b',[(c[1],c[0]) for c in coords][:10]
    #print
    #print m([c[1]/60. for c in coords],[c[0]/60. for c in coords])
    x,y=m([c[1]/60. for c in coords],[c[0]/60. for c in coords])
    #print 'a',x[:10],y[:10]
    return Polygon([(x[i],y[i]) for i in range(len(x)-1)])

def cond(category,start,i,f,long,type):
    a=f['areas']
    #thre=100.
    #if category=='slight':
    #    j_max=np.argmax([a[1][j] for j in range(start,i+1)])
    #    return a[1][j_max]/(a[0][j_max+1] - a[0][j_max])<=thre and long>0.66
    if category=='traffic':
        return i-start<=1# and f['l_m3'] > f['l_m1']
    if category=='direct':
        return type<0
    if category=='rerouting':
        #j_max=np.argmax([a[1][j] for j in range(start,i+1)])
        #return a[1][j_max]/(a[0][j_max+1] - a[0][j_max])>thre and long>0.66# and f['l_m3'] < f['l_m1']
        return long>0.66 and not cond('traffic',start,i,f,long,type)# and f['l_m3'] < f['l_m1']
    if category=='airport':
        return i-start<=1 and i==len(a[0])-2
    if category=='other':
        return not (cond('traffic',start,i,f,long,type) or cond('rerouting',start,i,f,long,type))

def categorize(f):
    if f['dev']>25. or delay(f['route_m1t'][-1][1],f['route_m1t'][0][1]) > 5000. or delay(f['route_m1t'][-1][1],f['route_m1t'][0][1]) < -1000.\
    or f['l_m3'] - f['l_m1'] >2000 or delay(f['route_m3t'][0][1],f['route_m1t'][0][1])>4000:
        f['flag']='rare'
    else:
        if f['rerouting']!=0:
            f['flag']='rerouted'
        elif f['rerouting']==0 and f['other']!=0:
            f['flag']='other'
        elif f['rerouting']==0 and f['other']==0 and f['traffic']!=0:
            f['flag']='traffic'
        elif f['rerouting']==0 and f['other']==0 and f['traffic']==0:# and f['direct']==0:
            f['flag']='normal'
        #else:
        #    f['flag']='other'
    
def do_colormap(a):
    q = mquantiles(sorted(a),prob=np.arange(0,1.,0.1))
    max_d=np.max(a)
    min_d=np.min(a)
    q=np.insert(q,0,min_d)
    q=np.append(q,max_d)
    q_back = q[:]
    q=(q-min_d)/(max_d-min_d)
    cdict = {'red':[],'green':[],'blue':[]}
    c_p=(0.,0.,0.)
    for i in range(0,len(q)):
        #c=cc.to_rgb(color2[i])
        c=(float(i)/float(len(q)-1),0.,1.-float(i)/float(len(q)-1))
        cdict['red'].append((q[i],c_p[0],c[0]))
        cdict['green'].append((q[i],c_p[1],c[1]))
        cdict['blue'].append((q[i],c_p[2],c[2]))
        c_p=c[:]
    return q_back,lsc('my_colormap',cdict,256)

def expand(shape,l): #WRONG ! TODO: Why ? Seems to work fine...
    border = list(shape.exterior.coords)[:-1]
    #c=np.array(list(shape.centroid.coords)[0])
    new_border = []
    l=l*sqrt(shape.area)
    for i in range(len(border)):
        if i==len(border)-1:
            j=0
        else:
            j=i+1
        #u1 = np.array([border[i-1][0] - border[i][0],border[i-1][1] - border[i][1]]) # Vecteur pointant du point vers celui d'avant
        #u1 = u1/norm(u1)
        #u2 = np.array([border[j][0] - border[i][0],border[j][1] - border[i][1]]) # Vecteur poitant du point vers celui d'apres
        #u2 = u2/norm(u2)
        ab = np.array(border[j]) - np.array(border[i-1])
        ab=ab/norm(ab)
        n = np.array((-ab[1],ab[0]))
        #u = np.array(b) - c
        #u = u/norm(u)
        #
        #if u1[1]+u2[1]!=0.: # Si les trois points ne sont pas alignes avec la l'horizontale
        #    u3 = np.array([(u1[0]+u2[0])/(u1[1]+u2[1]),1.]) #calcul du vecteur rentrant (bissectrice)
        #else:
        #    u3 = np.array([1.,0.])
        #u3 = u3/norm(u3)
        new_border.append(np.array(border[i]) + n*l) # Nouveau point decale vers l'exterieur selon la bissectrice
    return Polygon(new_border)

def length(lat1,lon1,lat2,lon2): # in degrees
    """
    Great circle (without altitude)
    """
    lat1=lat1*3.141593/180.
    lat2=lat2*3.141593/180.
    lon1=lon1*3.141593/180.
    lon2=lon2*3.141593/180.
    a=sin((lat2-lat1)/2.)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2.)**2
    return 6371*2*atan(sqrt(a), sqrt(1 - a))

def bet(G,nodes,flights,d,all_pairs):
    if all_pairs:
        return nx.betweenness_centrality(G)
    else:
        bc={n:0 for n in G.nodes()}
        pairs_entry_exit=[(f['route_' + nodes][0][0],f['route_' + nodes][-1][0]) for f in flights.values()]
        if d==2:
            shortest_paths={p:list(nx.all_shortest_paths(G,source=p[0],target=p[1])) for p in pairs_entry_exit}
        else:
            for e in G.edges():
                G[e[0]][e[1]]['wei']=(1000 - (float(e[0][-3:]) + float(e[1][-3:])))
            shortest_paths={p:list(nx.all_shortest_paths(G,source=p[0],target=p[1],weight='wei')) for p in pairs_entry_exit}
        
        for n in bc.keys():
            sig=0.
            for p in shortest_paths.values():
                if len(p)!=0:
                    sig+=float(len([1 for pp in p if n in pp]))/float(len(p))
            bc[n]=sig/float(len(pairs_entry_exit))
        return bc

def bet_OD(G, OD=None):
    if OD==None:
        return nx.betweenness_centrality(G)
    else:
        shortest_paths = {p:list(nx.all_shortest_paths(G, source=p[0], target=p[1])) for p in OD}

        bc={n:0 for n in G.nodes()}
        for n in bc.keys():
            sig = 0.
            for p in shortest_paths.values():
                if len(p)!=0:
                    sig += float(len([1 for pp in p if n in pp]))/float(len(p))
            bc[n] = sig/float(len(OD))
        return bc

def build_path(paras,version,full=True, prefix=None, suffix=None):
    if prefix == None:
        prefix = paras['main_rep']
    mode=paras['mode']
    big_filtre=paras['filtre'] + '_' + paras['type_zone'] + paras['zone']
    if not paras['micromode']:
        period= str(paras['starting_date'][0]) + '-' + str(paras['starting_date'][1]) + '-' + str(paras['starting_date'][2]) + '+' + str(paras['n_days']-1)
        if paras['collapsed']:
            period=period + '_step' + str(paras['step'])
    else:
        period=date_human(paras['timeStart']) + '--' + date_human(paras['timeEnd'])
    
    rep = prefix
    if  len(paras['zone'])==2:
        rep += paras['zone'] + '_v' + version + "/" + big_filtre + "/" + period + '_d' + str(paras['d'])
    else:
        rep += paras['zone'][:2] + '_v' + version + "/" + big_filtre + "/" + paras['zone'] + '_' + period + '_d' + str(paras['d'])
        
    if mode!='airports':
        if paras['use_base_net']:
            rep = rep + '_net'
        if paras['cut_alt']!=0.:
            rep = rep + '_cut' + str(paras['cut_alt'])
            
    if paras['direct']:
        rep = rep + '_directed'
        
    if mode=='sectors':
        rep = rep + '_sectors'
        
    if not mode=='airports':
        if not paras['do_local']:
            rep = rep + '_reduced'
    else:
        rep = rep + '_airports'
        
    if mode=='nav_sec':
        rep+='_nav_sec'

    if 'only_company' in paras.keys() and paras['only_company']!=None:
        rep += '_' + paras['only_company']
    if suffix!=None:
        rep+=suffix
        
    if full:
        return big_filtre, period, rep
    else:
        return rep
    
# def loading_network(rep,airports=False,nav_sec=False):
#     assert not (airports and nav_sec)
#     #big_filtre, period, rep = build_path(paras,version,airports=airports)
#     print 'Loading Network...'
    
#     if not airports and not nav_sec:
#         stri='data_results_m1_absolute'
#         compress=False
#         if not os.path.exists(rep + '/' + stri + '.pic'):
#             compress=True
#             os.system('cd ' + rep + ' && tar -xzf ' + stri + '.tar.gz')
        
#         f=open(rep + '/' + stri + '.pic','r')
#         seth=pickle.load(f)
#         f.close()
        
#         if compress:
#             os.system('rm ' + rep + '/' + stri + '.pic')
#         G=seth.G.copy()
#     elif airports:
#         f=open(rep + '/airport_network.pic','r')
#         G=pickle.load(f)
#         f.close()
#     elif nav_sec:
#         f=open(rep + '/nav_sec_network.pic','r')
#         G=pickle.load(f)
#         f.close()
#         #G=G.subgraph([n for n in G.nodes() if G.degree(n)!=0])
#     return G
    
def filter_graph(G, airports=False, m1=True, m3=False, nonzero=True):
    if not airports:
        return G.subgraph([n for n in G.nodes() if (G.degree(n)!=0 or not nonzero) and (G.node[n]['m1'] or not m1) and (G.node[n]['m3'] or not m3)])
    else:
        return G.subgraph([n for n in G.nodes() if (G.degree(n)!=0 or not nonzero)])
        
def gather_network_metrics(paras,rep,reps,mode,metrics=['n_nodes', 'n_edges','str','deg','bc','diameter'],name='data_init_m1',do=['avg','std','full'], restrict=False):
    print 'Gathering network metrics...'
    results={m:{d:[] for d in do} for m in metrics}
    for r in reps:
        print 'Fetching stuff in', r
        f=open(r + '/' + name + '.pic','r')
        if mode=='sectors' or mode=='navpoints':
            G=pickle.load(f).G
            G=restrict_to_ECAC(paras['airac'],G)
        else:
            G=pickle.load(f)
        f.close()

        print 'loaded'
        if 'n_nodes' in metrics:
            print 'doing number of nodes...'
            if results['n_nodes'].has_key('avg'):
                results['n_nodes']['avg'].append(len(G.nodes()))
        if 'n_edges' in metrics:
            print 'doing number of edges...'
            if results['n_edges'].has_key('avg'):
                results['n_edges']['avg'].append(len(G.edges()))
            
        if 'deg' in metrics:
            print 'doing degree...'
            if results['deg'].has_key('avg'):
                results['deg']['avg'].append(np.mean([G.degree(n) for n in G.nodes()]))
            if results['deg'].has_key('std'):
                results['deg']['std'].append(np.std([G.degree(n) for n in G.nodes()]))
            if results['deg'].has_key('full'):
                results['deg']['full'].append([G.degree(n) for n in G.nodes()])
        if 'str' in metrics:
            print 'doing strength...'
            if results['str'].has_key('avg'):
                results['str']['avg'].append(np.mean([G.degree(n, weight='weight') for n in G.nodes()]))
            if results['str'].has_key('std'):
                results['str']['std'].append(np.std([G.degree(n, weight='weight') for n in G.nodes()]))
            if results['str'].has_key('full'):
                results['str']['full'].append([G.degree(n, weight='weight') for n in G.nodes()])
        if 'bc' in metrics:
            print 'doing bc...'
            if results['bc'].has_key('avg'):
                results['bc']['avg'].append(np.mean(nx.betweenness_centrality(G).values()))
            if results['bc'].has_key('std'):
                results['bc']['std'].append(np.std(nx.betweenness_centrality(G).values()))
            if results['bc'].has_key('full'):
                results['bc']['full'].append(nx.betweenness_centrality(G).values())
                
        if 'diameter' in metrics:
            print 'doing diameter...'
            if results['diameter'].has_key('avg'):
                results['diameter']['avg'].append(nx.diameter(G))
            
    f=open(rep + '/results_net_metrics.pic','w')
    pickle.dump(results,f)
    f.close()
    
    f=open(rep + '/results_net_metrics.txt','w')
    for m,r in results.items():
        for k,v in r.items():
            if k!='full':
                print >>f, 'Mean of', m, k, np.mean(v)
                print >>f, 'Std of', m, k, np.std(v)
    f.close()
    return results
    
def make_plots_network_metrics(rep,results, fig_id=100000):
    i=0
    for m,v in results.items():
        if v.has_key('avg'):
            plt.figure(fig_id + i)
            plt.suptitle(m + 'avg')
            i+=1
            plt.plot(v['avg'],'ro-')
            plt.savefig(rep + '/network_metrics_' + m + '_' + 'avg.png')
        if v.has_key('std'):
            plt.figure(fig_id + i)
            plt.suptitle(m + 'std')
            i+=1
            plt.plot(v['std'], 'ro-')
            plt.savefig(rep + '/network_metrics_' + m + '_' + 'std.png')
        if v.has_key('full'):
            plt.figure(fig_id + i)
            plt.suptitle(m + 'full')
            i+=1
            plt.xlabel(m)
            plt.ylabel('Cumulative probability')
            for l in v['full']:
                cul=cumulative(l)
                plt.plot(cul[0],cul[1],'-')
            
            plt.savefig(rep + '/network_metrics_' + m + '_' + 'full.png')
            
            plt.figure(fig_id + i)
            plt.suptitle(m + 'full')
            i+=1
            plt.xlabel(m)
            plt.ylabel('Cumulative probability')

            if len(v['full'])>0:
                def f_fit(x, b):#, b, c):
                    #return a*np.exp(-b*x) + c
                    #return 1 + a*(np.exp(-b*x) - 1)   
                    return np.exp(-b*x)

                def f_fit2(x,b):
                    return - x*b


                popt=[]
                for l in v['full']:
                    cul=cumulative(l)
                    #popt.append(curve_fit(f_fit,np.array(cul[0]),np.array(cul[1]))[0])
                    popt.append(curve_fit(f_fit2, np.array(cul[0]), np.log(np.array(cul[1])))[0])
                    plt.semilogy(cul[0],cul[1],'-')

                popt =[np.mean([day[j] for day in popt]) for j in range(len(popt[0]))]
                #print 'popt', popt
                plt.semilogy(cul[0], np.exp(f_fit2(np.array(cul[0]),*popt)), 'k--')
                #plt.semilogy(cul[0], f_fit(np.array(cul[0]),*popt), 'k--')
                
                #plt.semilogy(cul[0], f_fit(np.array(cul[0]),0.0081), 'r--')


            plt.savefig(rep + '/network_metrics_' + m + '_' + 'full_semilog.png')
            
            plt.figure(fig_id + i)
            plt.suptitle(m + 'full')
            i+=1
            plt.xlabel(m)
            plt.ylabel('Cumulative probability')
            for l in v['full']:
                cul=cumulative(l)
                plt.loglog(cul[0],cul[1],'-')
            
            plt.savefig(rep + '/network_metrics_' + m + '_' + 'full_loglog.png')
            
def restrict_to_ECAC(airac,G):
    f=open(jn(path_modules, 'All_shapes_' + str(airac) + '.pic'),'r')
    shape=pickle.load(f)['ECAC']['boundary'][0]
    f.close()
    return nx.subgraph(G,[n for n in G.nodes() if \
    Point((G.node[n]['coord'][0]/60.,G.node[n]['coord'][1]/60.)).within(shape) and (not G.node[n].has_key('m1') or G.node[n]['m1'])])
    
def get_paras():
    #if len(sys.argv)>1:
    #    fil=sys.argv[1]    
    #else:
    fil='para'
    mod = __import__(fil,fromlist='paras')
    reload(mod)
    paras = __import__(fil,fromlist='paras').paras
    
    return paras

def extract_flows_from_data(paras, nodes, pairs = []):
    """
    Extract the number of flight between entries and exits
    (First and last points of trajectories within nodes)
    plus times of entries.
    """
    flights=get_flights(paras)

    flows = {}
    times = {}
    flights_selected = []
    for f in flights.values():
        route=f['route_m1']
        i=0
        found_entry=False
        while i<len(route) and not found_entry: #Find the first point which is in the nodes given.
            p=route[i][0]
            if p in nodes:
                idx_entry = i
                entry=p
                found_entry=True
            i+=1
        i=len(route) - 1
        found_exit=False
        while i>=0 and not found_exit: #find the last point which is in the list of nodes.
            p=route[i][0]
            if p in nodes:
                idx_exit = i
                exit=p
                found_exit=True
            i+=-1
        
        if found_exit and found_entry and entry!=exit and ((entry,exit) in pairs or pairs ==[]): 
            #check that everypoints between the entry and the exit are in the list of nodes.
            trajectory_in_nodes = True
            j = idx_entry
            while trajectory_in_nodes and j<idx_exit:
                if not route[j][0] in nodes:
                    trajectory_in_nodes = False
                j+=1

            if trajectory_in_nodes:
                flows[(entry, exit)] = flows.get((entry, exit),0) + 1
                times[(entry, exit)] = times.get((entry, exit),[]) + [f['route_m1t'][idx_entry][1]]
                #flights_selected[(entry, exit)] = flights_selected.get((entry, exit),[]) + [f]
                f['route_m1']=f['route_m1'][idx_entry:idx_exit+1]
                f['route_m1t']=f['route_m1t'][idx_entry:idx_exit+1]
                flights_selected.append(f)

    return flows, times, flights_selected

def _OLD_select_layer_sector(password_db, airac, zone, level = 250.):
    db=_mysql.connect("localhost","root", password_db,"ElsaDB_A" + str(airac), conv=my_conv)

    query = """SELECT S.sectorId#, SS.airblockUId#,  REPLACE(REPLACE(REPLACE(AsText(A.boundary), '(',''),'POLYGON',''),')','') as boundary
            FROM SectorSlice as SS, Sector as S, Airblock as A
            WHERE minHeight <""" + str(int(level))  + """ AND maxHeight >""" + str(int(level)) + \
            """ AND SS.sectorUId=S.uniqueId AND S.sectorId like '""" + zone + \
            """%' AND A.uniqueId=SS.airblockUId AND S.type='ES' AND S.airspaceCategory='_'
            ORDER BY S.sectorId"""

    db.query(query)
    r=db.store_result()
    rr=r.fetch_row(maxrows=0, how=0)

    db.close()

    return [rrr[0] for rrr in rr]

def build_network_based_on_shapes(password_db, airac, zone, layer):
    # Finding the max height of sectors in this area
    db=_mysql.connect("localhost","root", password_db,"ElsaDB_A" + str(airac), conv=my_conv)
    query="""SELECT max(maxHeight)
    FROM SectorSlice as SS, Sector as S, Airblock as A
    WHERE SS.sectorUId=S.uniqueId AND S.sectorId like '""" + zone + """%' 
    AND A.uniqueId=SS.airblockUId AND S.type='ES' AND S.airspaceCategory='_' """

    db.query(query)
    r=db.store_result()
    max_height = float(r.fetch_row(maxrows=0,how=0)[0][0])
    db.close()

    if max_height<layer:
        print "There is no sectors at the requested layer (", layer, "),"
        layer=max_height-5
        print "so I set it to the maximum height - 5FL (", layer, ")."

    sectors = select_layer_sector(password_db, airac, zone, layer)
    with open(jn(path_modules, 'All_shapes_334.pic'),'r') as f:
        all_shapes = pickle.load(f)

    shapes = {s:all_shapes[s]['boundary'][0] for s in sectors}

    # Remove overlapping sectors.
    to_remove = []
    items = shapes.items()
    for i, (sec1, shape1) in enumerate(items):
        for j in range(i+1, len(items)):
            sec2, shape2 = items[j]
            if shape1.intersects(shape2) and not shape1.touches(shape2): #overlap
                #overlapping_sectors.append(sorted([sec1, sec2], key:lambda x:x.area))
                #to_remove.append([sec1, sec2][np.argmin([shape1, shape2])]) #Flag the smallest to be removed
                cut_shape = shape1.difference(shape2)
                if not cut_shape.is_empty:
                    shapes[sec1] = cut_shape
                else:
                    del shapes[sec1]                    
    # print "Removing sectors", set(to_remove)
    # for s in set(to_remove):
    #     del shapes[s]

    # Make the network detecting which sectors are touching which others.
    G = nx.Graph()
    items = shapes.items()
    for i, (sec1, shape1) in enumerate(items):
        G.add_node(sec1, coord=list(shape1.representative_point().coords)[0], shape=shape1)
        for j in range(i+1, len(items)):
            sec2, shape2 = items[j]
            if shape1.touches(shape2):
                G.add_edge(sec1, sec2)

    return G, shapes

def select_layer_sector(password_db, airac, zone, layer):
    """
    Taken from Module/build_one_layer_sector.py
    """
    print 'Extracting the layer...'

    db=_mysql.connect("localhost","root", password_db,"ElsaDB_A" + str(airac), conv=my_conv)
    
    query=  """SELECT S.sectorId,  REPLACE(REPLACE(REPLACE(AsText(A.boundary), '(',''),'POLYGON',''),')','') as boundary
    FROM SectorSlice as SS, Sector as S, Airblock as A
    WHERE minHeight <""" +str(layer) + """ AND maxHeight >""" + str(layer) + """ AND SS.sectorUId=S.uniqueId AND S.sectorId like '""" + zone + \
    """%' AND A.uniqueId=SS.airblockUId AND S.type='ES' AND S.airspaceCategory='_' ORDER BY S.sectorId"""

    db.query(query)
        
    r=db.store_result()
    sects=r.fetch_row(maxrows=0,how=1)

    try:
        assert len(sects)>0
    except AssertionError:
        print "Could not find any sector in zone", zone, "at altitude", layer, "in database."
        print "Corresponding query:"
        print query
        raise

    sector=sects[0]['sectorId']
    bounds={}
    i=0
    while i<len(sects):
        bounds[sector]=[]
        while i<len(sects) and sects[i]['sectorId']==sector:
            bd=[(float(split(p,' ')[0]), float(split(p,' ')[1])) for p in split(sects[i]['boundary'],',')]
            bounds[sector].append(Polygon(bd))
            i+=1
        if len(bounds[sector])>1:
            bounds[sector]=cascaded_union(bounds[sector])
        else:
            bounds[sector]=bounds[sector][0]
        if i<len(sects):
            sector=sects[i]['sectorId']

    db.close()

    return bounds

def map_of_net(G, colors='r', num=0, limits=(0,0,0,0), title='', size_nodes=1., size_edges=2., nodes=[], zone_geo=[], edges=True, fmt='svg', dpi=100, \
    save_file = None, show=True, figsize=(9,6), background_color='white', key_word_weight='weight', z_order_nodes=6, diff_edges=False):
    """
    Draw a net. TODO: maximum width.
    """
    restrict_nodes = True
    if limits==(0,0,0,0):
        limits = (min([G.node[n]['coord'][0]/60. for n in nodes]) - 0.2,
                min([G.node[n]['coord'][1]/60. for n in nodes]) - 0.2,
                max([G.node[n]['coord'][0]/60. for n in nodes]) + 0.2,
                max([G.node[n]['coord'][1]/60. for n in nodes]) + 0.2)
        restrict_nodes = False

    if nodes==[]:
        if restrict_nodes:
            # Restrict nodes to geometrical extent of the zone.
            nodes = [n for n in G.nodes() if limits[0]-0.2<=G.node[n]['coord'][0]/60.<=limits[2]+0.2 and limits[1]-0.2<=G.node[n]['coord'][1]/60.<=limits[3]+0.2]
        else:
            nodes = G.nodes()

    if type(colors)==type({}):
        colors = [colors[n] for n in nodes]

    if type(z_order_nodes)==type({}):
        z_order_nodes = [z_order_nodes[n] for n in nodes]

    if type(size_nodes)==type(1) or type(size_nodes)== type(1.):
        size_nodes = [size_nodes for n in nodes]
    elif size_nodes==[]:
        size_nodes = [1 for n in nodes]
    elif type(size_nodes)==type((0,1)):
        if size_nodes[0]=='strength':
            size_nodes=[G.degree(n, weight=key_word_weight)*size_nodes[1] for n in nodes]
        elif size_nodes[0]=='degree':
            size_nodes=[G.degree(n)*size_nodes[1] for n in nodes]
        else:
            Exception("The following size function is not implemented:" + size_nodes)
    elif type(size_nodes)==type({}):
        size_nodes=[size_nodes[n] for n in nodes]

    x_min,y_min,x_max,y_max = limits
    fig = plt.figure(num, figsize=figsize)
    gs = gridspec.GridSpec(1, 2, width_ratios=[6.,1.])
    ax = plt.subplot(gs[0])
    #ax.set_aspect(1./0.8)
    ax.set_aspect(figsize[0]/figsize[1])
    m = draw_zonemap(x_min,y_min,x_max,y_max,'i', sea_color=background_color, continents_color=background_color, lake_color=background_color)
    x,y = split_coords(G,nodes, r=0.1)
    x,y = m(y,x)
    ax.set_title(title)
    sca = ax.scatter(x, y, marker='o', zorder=z_order_nodes, s=size_nodes, lw=0, c=colors)#,cmap=my_cmap)
    max_wei = max([abs(G[e[0]][e[1]][key_word_weight]) for e in G.edges() if e[0] in nodes and e[1] in nodes])
    if edges:
        for e in G.edges():
            if e[0] in nodes and e[1] in nodes:
                #print e,width(G[e[0]][e[1]]['weight'])
                if diff_edges:
                    color = 'r' if G[e[0]][e[1]][key_word_weight]>0 else 'b'
                else:
                    color = 'k'
                xe1,ye1 = m(G.node[e[0]]['coord'][1]/60.,G.node[e[0]]['coord'][0]/60.)
                xe2,ye2 = m(G.node[e[1]]['coord'][1]/60.,G.node[e[1]]['coord'][0]/60.)
                plt.plot([xe1,xe2],[ye1,ye2], '-', lw=width(G[e[0]][e[1]][key_word_weight], max_wei, scale=size_edges), color=color, zorder=4)

    if zone_geo!=[]:
        patch=PolygonPatch(adapt_shape_to_map(zone_geo,m), facecolor='grey', edgecolor='grey', alpha=0.08, zorder=3)#edgecolor='grey', alpha=0.08,zorder=3)
        ax.add_patch(patch)

    if save_file!=None:
        plt.savefig(save_file + '.' + fmt, dpi = dpi)
        print 'Figure saved as', save_file + '.' + fmt
    if show:
        plt.show()

def width(x,maxx, scale=2.):
    return scale*x/maxx#+0.02#0.2
    
def split_coords(G,nodes, r=0.04):
    lines=[]
    for n in G.nodes():
        if n in nodes:
            added=False
            for l in lines:
                if dist_flat(G.node[n]['coord'],G.node[l[0]]['coord'])<1.: #nodes closer than 0.1 degree
                    l.append(n)
                    added=True
            if not added:
                lines.append([n])
    
    for l in lines[:]:
        if len(l)==1:
            lines.remove(l)

    pouet={}
    for l in lines:
        for n in l:
            pouet[n]=l
    x,y=[],[]
    for n in nodes:
        if not n in pouet.keys():
            x.append(G.node[n]['coord'][0]/60.)
            y.append(G.node[n]['coord'][1]/60.)
        else:
            l=pouet[n]
            #r=0.04
            theta=2.*pi*float(l.index(n))/float(len(l))
            x.append(G.node[n]['coord'][0]/60. + r*cos(theta))
            y.append(G.node[n]['coord'][1]/60. + r*sin(theta))
    return x,y

def numberize_nodes(G):
    G.mapping = {n:i for i, n in enumerate(G.nodes())}
    nx.relabel_nodes(G, G.mapping, copy=False)

def numberize_trajs(trajs, mapping, fmt='(n, z), t'):
    if fmt=='(n, z), t':
        geom_trajs, start_dates = zip(*trajs)
        geom_trajs = list(geom_trajs)

        for i, traj in enumerate(geom_trajs):
            for j, pouet in enumerate(traj):
                pouet = list(pouet)
                pouet[0] = mapping[pouet[0]]
                traj[j] = tuple(pouet)
            geom_trajs[i] = traj 
        #print trajs[:2]
        trajs = list(zip(geom_trajs, start_dates))
    else:
        raise Exception("Format", fmt, "not implemented yet")

def infer_navpoints_from_trajectories(trajectories, fmt='(n, z), t'):
    pass 

def build_traffic_network(trajectories, fmt_in='(x, y, z, t)'):
    """
    Build a networkx object which has the points of the trajectories as nodes,
    an edge between one point and another if there is at least one flight 
    flying between them. The edges are weighted with the number of flights going from
    one node to the other.

    The input format contains some labels for the trajectories, they will be used 
    for the nodes of the network as they are. If the format contains coordinates
    only, the function will detect the closest points and make them nodes, using a
    threshold. This is done by the trajectory converter in the general_tools library.
    """ 
    
    accepted_formats = ['(x, y, z, t)', '(x, y, z, t, s)', '(n, z), t', '(n), t']
    
    #for 
# def _test_build_network_based_on_shapes():
#     G, shapes = build_network_based_on_shapes('4ksut79f', 334, 'LF', 350.)

#     draw_network_and_patches(G, None, shapes, draw_navpoints_edges=False, \
#     draw_sectors_edges=True, rep='.', save=False, name='network', \
#     show=True, flip_axes=True, trajectories=[], \
#     trajectories_type='navpoints', dpi = 100, figsize = None)

#     # fig = plt.figure()
#     # ax = fig.add_subplot(111)
#     # for pol in shapes.values():
#     #     patch = PolygonPatch(pol,alpha=0.5, zorder=2)
#     #     ax.add_patch(patch) 
#     # plt.plot([46.],[1.])
#     # plt.show()

def _test_select_layer_sector():
    sectors = select_layer_sector('4ksut79f', 334, 'LF', 350.)
    #print sectors
    with open(jn(path_modules, 'All_shapes_334.pic'),'r') as f:
        all_shapes = pickle.load(f)

    shapes = {s:all_shapes[s]['boundary'][0] for s in sectors}

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for pol in shapes.values():
        patch = PolygonPatch(pol,alpha=0.5, zorder=2)
        ax.add_patch(patch) 
    plt.plot([46.],[1.])
    #plt.show()

    #print shapes


if __name__=='__main__':
    # paras = get_paras()

    # paras_nav=paras.copy()
    # paras_nav['mode']='navpoints'
    # paras_sec=paras.copy()
    # paras_sec['cut_alt']=0.
    # paras_sec['mode']='sectors'

    # seth_nav=get_set(paras_nav)
    # seth_sec=get_set(paras_sec)
    # seth=get_set(paras)

    _test_select_layer_sector()
    _test_build_network_based_on_shapes()



        
            
