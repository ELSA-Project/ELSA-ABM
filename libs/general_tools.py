#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This is a general tool file.
"""

import numpy as np
#from string import split
import os
import sys
#from math import *
from time import gmtime, strftime
from scipy.misc import comb
from scipy.optimize import curve_fit
from scipy.stats.mstats import mquantiles
from scipy.stats import pearsonr, gaussian_kde, rv_continuous
from mpl_toolkits.basemap import Basemap
import pickle
from MySQLdb.constants import FIELD_TYPE
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from descartes.patch import PolygonPatch
from shapely.geometry import Polygon
import contextlib
import sys
import subprocess
import datetime
from numpy.random import randint
from math import *
import networkx as nx
import matplotlib.pyplot as plt

from math import sqrt

days_per_month=np.array([31,28,31,30,31,30,31,31,30,31,30,31])

legend_location={'ur':1, 'ul':2, 'll':3, 'lr':4, 'r':5, 'cl':6, 'cr':7, 'lc':8, 'uc':9, 'c':10}

my_conv = { FIELD_TYPE.LONG: int, FIELD_TYPE.FLOAT: float, FIELD_TYPE.DOUBLE: float }

_colors=('Blue','BlueViolet','Brown','CadetBlue','Crimson','DarkMagenta','DarkRed','DeepPink','Gold','Green','OrangeRed')

def flip_polygon(pol):
    """
    Flip x and y axis for a polygon.
    """
    return Polygon([(p[1], p[0]) for p in list(pol.exterior.coords)])


def draw_network_and_patches(G, G_nav, polygons, draw_navpoints_edges=True, \
    draw_sectors_edges=False, rep='.', save=True, name='network', \
    show=True, flip_axes=False, trajectories=[], \
    trajectories_type='navpoints', dpi = 100, figsize = None):
    """
    Quite general functions used to draw navigations points, sectors, and trajectories.
    That's for the ABM mainly. TODO: move it?
    """

    # print "trajectories:"
    # for t in trajectories:
    #     print t
    # print
    # for i,pairs in enumerate(trajectories):
    #     for j, t in enumerate(pairs):
    #         trajectories[i][j] = tuple(t)

    if flip_axes:
        if G:
            for n in G.nodes():
                G.node[n]['coord']=(G.node[n]['coord'][1], G.node[n]['coord'][0])
        if G_nav:
            for n in G_nav.nodes():
                G_nav.node[n]['coord']=(G_nav.node[n]['coord'][1], G_nav.node[n]['coord'][0])
        if polygons!=None:
            polygons={k:flip_polygon(pol) for k,pol in polygons.items()}

    print  'Drawing networks and patches...'
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(111)
    if polygons!=None:
        for pol in polygons.values():
            patch = PolygonPatch(pol,alpha=0.5, zorder=2)
            ax.add_patch(patch) 
    if G:
        ax.scatter([G.node[n]['coord'][0] for n in G.nodes()], [G.node[n]['coord'][1] for n in G.nodes()],c='r', marker='s')
        if draw_sectors_edges:
            for e in G.edges():
                plt.plot([G.node[e[0]]['coord'][0],G.node[e[1]]['coord'][0]],[G.node[e[0]]['coord'][1],G.node[e[1]]['coord'][1]],'k-',lw=0.2)#,lw=width(G[e[0]][e[1]]['weight'],max_wei),zorder=4)
        
    if G_nav:
        ax.scatter([G_nav.node[n]['coord'][0] for n in G_nav.nodes()], [G_nav.node[n]['coord'][1] for n in G_nav.nodes()],c=[_colors[G_nav.node[n]['sec']%len(_colors)]\
            for n in G_nav.nodes()],marker='o', zorder=6, s=20)
        if draw_navpoints_edges:
            for e in G_nav.edges():
                plt.plot([G_nav.node[e[0]]['coord'][0],G_nav.node[e[1]]['coord'][0]],[G_nav.node[e[0]]['coord'][1],G_nav.node[e[1]]['coord'][1]],'k-',lw=0.2)#,lw=width(G[e[0]][e[1]]['weight'],max_wei),zorder=4)
    
    if trajectories!=[]:
        if trajectories_type=='navpoints':
            H=G_nav
        else:
            H=G
        #weights={n:{v:0. for v in H.neighbors(n)} for n in H.nodes()}
        weights={n:{} for n in H.nodes()}
        for path in trajectories:
            try:
                #path=f.FPs[[fpp.accepted for fpp in f.FPs].index(True)].p
                for i in range(0,len(path)-1):
                    #print path[i], path[i+1]
                    if path[i] in weights.keys():
                        weights[path[i]][path[i+1]] = weights[path[i]].get(path[i+1],0.) + 1.
            except ValueError:
                paras_to_display

        max_w=np.max([w for vois in weights.values() for w in vois.values()])
        
        for n,vois in weights.items():
            if n in H.nodes():
                for v,w in vois.items():
                    if v in H.nodes():
                        plt.plot([H.node[n]['coord'][0],H.node[v]['coord'][0]],[H.node[n]['coord'][1],H.node[v]['coord'][1]],'r-',lw=w/max_w*4.)
       
    if save:
        plt.savefig(rep + '/' + name +'.png', dpi = dpi)
        print 'Figure saved in', rep + '/' + name +'.png'
    if show:
        plt.show()

    return fig

def range_congru(start,end,cong):
    """
    Create a list from start to end with a congruence. TODO: could be much quicker.
    """
    a=[]
    i=start
    while i<end:
        a.append(i%cong)
        i+=1
    return a

def delay(date,starting_date=np.array([2010,1,1,0,0,0])): 
    """
    Convert date into time elapsed (in seconds) since starting date. DO NOT take into 
    account bissextile years. 
    """
    date=np.array(date)
    starting_date=np.array(starting_date)
    da=date[0]-starting_date[0]
    #dmo=date[1]-starting_date[1]+12*da
    dd=date[2]-starting_date[2]+ sum([days_per_month[i] for i in range_congru(min(date[1],starting_date[1])-1, max(date[1],starting_date[1]) -1 + 12*da,12)])#days_per_month[min(date[1],starting_date[1])-1]*dmo  
    dh=date[3]-starting_date[3]+24*dd
    dmi=date[4]-starting_date[4]+60*dh
    ds=date[5]-starting_date[5]+60*dmi
    return ds

def time_from_day(date): #converts date into time elapsed (in seconds) since start of the day
    return delay(date,starting_date = np.array([date[0],date[1],date[2],0.,0.,0.]))

def date_st(delay,starting_date=[2010,1,1,0,0,0]): #contrary of delay. Positive delays only. NOT WORKING.
    delay=int(delay)
    starting_date=np.array(starting_date)
    #print 'delay=',delay 
    d = delay%(3600*24)
    h = delay%3600 - d*24
    mi = delay%60 - h*60 - d*60*24
    s = delay - mi*60 - h*3600 - d*3600*24 
    
    s2 = (starting_date[-1] + s)%60
    rs=(starting_date[-1] + s)/60
    mi2 = (starting_date[-2] + mi + rs)%60
    rmi = (starting_date[-2] + mi + rs)/60
    h2 = (starting_date[-3] + h + rmi)%24
    rh= (starting_date[-3] + h + rmi)/24
    #print  starting_date[-4], d, rh
    d2 = starting_date[-4] + d + rh
    m2 = starting_date[1]
    y2 = starting_date[0]
    
    #print y2, m2, d2
    #rd = (starting_date[-4] + d + rh)/days_per_month[starting_date[-4]]
    #m2 = (starting_date[-5] + m + rd)%12
    #rm = (starting_date[-5] + m + rd)/12
    #y2 = starting_date[-5] + rm
    
    while d2 - days_per_month[m2-1]>0:
        d2 = d2 - days_per_month[m2-1]
        if m2==12:
            y2+=1
            m2=1
        else:
            m2+=1
    
    return y2,m2,d2,h2,mi2,s2


#def date_st2(delay,starting_date=np.array([2010,1,1,0,0,0])): #Finer version of the previous one.
#    delay=int(delay)
#    starting_date=np.array(starting_date)
#
#    d = (((delay/60)/60)/24)
#    h = delay/3600 - n_days*24
#    mi = delay/60 - (delay/3600)*60
#    s = delay%60
#    i=0
#    y,m,d = starting_date[:3]
#    date = y,m,d,h,mi,s
#    
#    while i<n_days:
#        while m<13 and i<n_days:
#            while d<days_per_month[m-1]+1 and i<n_days:
#                d+=1
#                i+=1
#            if d==days_per_month[m-1]+1:
#                m+=1
#                d=1
#        if m==13:
#            y+=1
#            m=1
#    return y,m,d,h,mi,s

def date_generation(first_date,n_days):
    """"
    DO NOT take into account bissextile years. 
    """
    i=0
    y,m,d=first_date
    dates=[]
    while i<n_days:
        while m<13 and i<n_days:
            while d<days_per_month[m-1]+1 and i<n_days:
                dates.append([y, m, d])
                d+=1
                i+=1
            m+=1
            d=1
        y+=1
        m=1
    return dates


#def date_generation2(first_date,end_date):
#    ys,ms,ds,hs,mis,ss=first_date
#    ye,me,de,he,mie,se=end_date
#    y,m,d,h,mi,s=ys,ms,ds,hs,mis,ss
#    dates=[]
#    while y<=ye:
#        while m<=me:
#            while m<=me:
#            while d<days_per_month[m-1]+1 and i<n_days:
#                dates.append([y, m, d])
#                d+=1
#                i+=1
#            m+=1
#            d=1
#        y+=1
#        m=1
#    return dates

def date_db(date,dateonly=False): # y,m,d,h,m,
    """
    For mysql database.
    """
    if dateonly:
        return str(date[0]) + '-' + str(date[1])  + '-' + str(date[2])
    else:
        return str(date[0]) + '-' + str(date[1])  + '-' + str(date[2]) + ' ' + str(date[3]) + ':' + str(date[4]) + ':' + str(date[5])

def date_human(date,dateonly=False): # y,m,d,h,m,
    """
    For humans. Don't use it you're a bot.
    """
    if dateonly:
        return str(date[0]) + '-' + str(date[1])  + '-' + str(date[2])
    else:
        return str(date[0]) + '-' + str(date[1])  + '-' + str(date[2]) + '_' + str(date[3]) + ':' + str(date[4]) + ':' + str(date[5])
        
def header(paras,name_prog,version,paras_to_display=[]):
    """
    Generic header of a program, displaying some parameters and the 
    starting time.
    """
    if paras_to_display==[]:
        paras_to_display=paras.keys()
    first_line='------------------------ Program ' + name_prog + ' version ' + version + ' ------------------------'
    l=len(first_line)
    trait='-'*l
    head=trait + '\n' + first_line + '\n' + trait + '\n'
    line=''
    for k in paras_to_display:
        v=paras[k]
        pouic=k + ': ' + str(v) +  ' ; '
        if len(line) < l - len(pouic):
            line = line + pouic
        else:
            head = head + line + '\n' 
            line=pouic
    head +=line + '\n' + trait +'\n'
    head += 'Started on ' + strftime("%a, %d %b %Y %H:%M:%S", gmtime()) + '\n'
    head = head + trait + '\n'
    return head
    
def make_rep(paras, prefix='',suffix='', par=[]):
    if par==[]:
        par=paras.keys()
    rep=prefix
    for p in par:
        if type(paras[p])!=type('s'):
            rep+=p + str(paras[p]) + '_'
        else:
            rep+=p + '_' + paras[p] + '_' 
    if suffix=='':
        rep=rep[:-1] # Pour eviter d'avoir le _ a la fin
    else:
        rep+=suffix    
    return rep 
    
def cumulative(a):
    """
    Used to go from pdf to cdf.
    """
    b=sorted(a)
    cdf=[[],[]]
    n=0
    for i in range(len(b)-1,-1,-1):
        n+=1
        if i<len(b)-1 and b[i]!=b[i+1]:
            cdf[0].insert(0,b[i])
            cdf[1].insert(0,n)
    if len(cdf[1])!=0:
        cdf[1]=np.array(cdf[1])/float(cdf[1][0])
    return cdf
    
def hyper_test(X, K, n, N):
    """
    Hypergeometric test.
    """
    return (1 - sum([comb(K,x)*comb(N-K,n-x)/float(comb(N,n)) for x in range(X)]))
    
def draw_zonemap(x_min,y_min,x_max,y_max,res, continents_color='white', lake_color='white', sea_color='white'):
    m = Basemap(projection='gall',lon_0=0.,llcrnrlon=y_min,llcrnrlat=x_min,urcrnrlon=y_max,urcrnrlat=x_max,resolution=res)
    m.drawmapboundary(fill_color=sea_color) #set a background colour
    m.fillcontinents(color=continents_color,lake_color=lake_color)  # #85A6D9')
    m.drawcoastlines(color='#6D5F47', linewidth=0.8)
    m.drawcountries(color='#6D5F47', linewidth=0.8)
    m.drawmeridians(np.arange(-180, 180, 5), color='#bbbbbb')
    m.drawparallels(np.arange(-90, 90, 5), color='#bbbbbb')
    return m
    
def save(stuff, name='stuff.pic', rep='.'): # why did I make this function? I am so lazy.
    with open(rep + '/' + name,'w') as f:
        pickle.dump(stuff,f)
    return rep + '/' + name
    
def load(name, rep='.'): # same remark.
    with open(rep + '/' + name,'r') as f:
        stuff=pickle.load(f)
    return stuff
    
def loading(func):   #decorator
    """
    This is useful when you want to compute something, but you might already have
    it on the disk, in which case you want to load it from it.
    """
    def wrapper(*args, **kwargs):
        if kwargs.has_key('path'):
            if type(kwargs['path'])==type('p'):
                path=kwargs['path']
                rep=''
            elif type(kwargs['path'])==type(['p','p']):
                assert len(kwargs['path'])==2
                path= kwargs['path'][0] + '/' + kwargs['path'][1]
                rep=kwargs['path'][0]
        else:
            path='pouet' # A changer (mettre le nom de la fonction). TODO
        if kwargs.has_key('force'):
            force=kwargs['force']
        else:
            force=False
        if kwargs.has_key('save'):
            save=kwargs['save']
        else:
            save=True
        if kwargs.has_key('verbose_load'):
            verbose_load=kwargs['verbose_load']
        else:
            verbose_load=False
        kwargs.pop('path',None)
        kwargs.pop('save',None)
        kwargs.pop('force',None)
        kwargs.pop('verbose_load',None)
        
        if os.path.exists(path) and not force:
            if verbose_load:
                print 'Loading from disk.'
            with open(path,'r') as f:
                something=pickle.load(f)
            
            return something
        else:
            if verbose_load:
                print 'Computing from scratch.'
            something= func(*args,**kwargs)
            if save:
                if rep!='':
                    subprocess.check_output('mkdir -p ' + rep, shell=True)
                with open(path,'w') as f:
                    pickle.dump(something,f)
                if verbose_load:
                    print 'Saved in ', path
            return something
    return wrapper

class DummyFile(object): # used for silence
    def write(self, x): pass

@contextlib.contextmanager
def silence(silent):
    """
    Silence your program :).
    """
    if silent:
        save_stdout = sys.stdout
        sys.stdout = DummyFile()
    yield
    if silent:
        sys.stdout = save_stdout

@contextlib.contextmanager
def clock_time():
    """
    Time your program.
    """
    start=datetime.datetime.now()
    yield
    elapsed = datetime.datetime.now() - start
    print 'Executed in ', str(elapsed), 's.'

def counter(i, end, start=0, message=''):
    """
    Count the number of iterations, that's all.
    """
    sys.stdout.write('\r' + message + str(int(100*(abs(i-start)+1)/float(abs(end-start)))) + '%')
    sys.stdout.flush() 
    if i==end-1:
        print

@contextlib.contextmanager
def write_on_file(name_file):
    """
    Useful  for logs.
    """
    if name_file!=None:
        with open(name_file, 'w') as f:
            save_stdout = sys.stdout
            sys.stdout = f
            yield
            sys.stdout = save_stdout
    else:
        yield

#decorator
def save_fig(plot):
    """
    Automatic saving for plots.
    """
    def wrapper(*args, **kwargs):
        rep = kwargs['rep']
        kwargs.pop('rep', None)
        suffix = kwargs['suffix']
        kwargs.pop('suffix', None)
        ret = plot(*args, **kwargs)
        if rep!='':
            plt.savefig(rep + '/' + plot.func_name + '_' + suffix + '.png')
            print 'Saved in', rep + '/' + plot.func_name + '_' + suffix + '.png'
        return ret
    return wrapper

def make_union_interval(intervals):
    """
    Used to make union of intervals.
    """
    list_debut = [ interv[0] for interv in intervals ]
    list_fin = [ interv[1] for interv in intervals ]
    list_debut.sort()
    list_fin.sort()
    list_final = []
    nb_superposition = 0

 
    while list_debut:
        ordre_debut_fin = cmp(list_debut[0], list_fin[0])
        if ordre_debut_fin == -1:
            pos_debut = list_debut.pop(0)
            nb_superposition += 1
            list_final.append((pos_debut,nb_superposition))
        elif ordre_debut_fin == +1:
            pos_fin = list_fin.pop(0)
            nb_superposition -= 1
            list_final.append((pos_fin,nb_superposition))
        else:
            list_debut.pop(0)
            list_fin.pop(0)
    
    while list_fin:
        pos_fin = list_fin.pop(0)
        nb_superposition -= 1
        list_final.append((pos_fin,nb_superposition))
 
    return list_final

def sort_lists(list1, list2):
    """
    Sort ith respect to values in list1
    """
    return zip(*sorted(zip(list1, list2), key = lambda pair: pair[0]))

def fit(x, y, first_point = 0, last_point = -1, f_fit = None, p0 = None):
    """
    Simple function for linear fit.
    """
    if f_fit ==None:
        def f_fit(x,a,b):
            return a + b*x

    x, y = sort_lists(x,y)

    to_remove = []
    for i in range(len(x)):
        if np.isnan(x[i]) or np.isnan(x[i]):
            to_remove.append(i)
    x = [x[i] for i in range(len(x)) if not i in to_remove]
    y = [y[i] for i in range(len(x)) if not i in to_remove]

    x = np.array(x)
    y = np.array(y)

    #print first_point, last_point   
    #print x[first_point:last_point]

    popt, pcov = curve_fit(f_fit, x[first_point:last_point], y[first_point:last_point], p0=p0)
    #print popt, pcov
    def f_fit_opt(x):
        return f_fit(x,*popt)

    return popt, pcov, f_fit_opt

@save_fig
def plot_scatter(a, b, xlabel = '', ylabel = '', title = '', do_fit = True):
    plt.figure()
    plt.title(title)

    if do_fit:
        popt, pcov, f_fit_opt = fit(a, b)
        print 'Optimal oo/slope:', popt
        print 'Covariance matrix:', pcov
        plt.plot(sorted(a), [f_fit_opt(x) for x in sorted(a)], 'b-')

        print 'Pearson correlation coefficient:', pearsonr(a,b)[0]

    plt.plot(a, b, 'ro')


    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

@save_fig
def plot_quantiles(a, b, xlabel = '', ylabel = '', title = '', n_quantiles = 8, errorbars = True, fmt = 'o--', col = 'b'):

    a, b = sort_lists(a, b)

    qs = mquantiles(a, prob = np.arange(min(a)/float(max(a)), 1., (1. - min(a)/float(max(a)))/float(n_quantiles)))
    b_avg = [np.mean([s for j,s in enumerate(b) if qs[i]<=a[j]<qs[i+1]]) for i in range(len(qs)-1)]
    b_std = [np.std([s for j,s in enumerate(b) if qs[i]<=a[j]<qs[i+1]])/sqrt(len([s for j,s in enumerate(b) if qs[i]<=a[j]<qs[i+1]])) for i in range(len(qs)-1)]
    a_avg = [np.mean([s for j,s in enumerate(a) if qs[i]<=a[j]<qs[i+1]]) for i in range(len(qs)-1)]

    plt.figure()
    plt.title(title)

    if errorbars:
        plt.errorbar(a_avg, b_avg, b_std, fmt=fmt, c=col)
    else:
        plt.plot(a_avg, b_avg, col + fmt)

    plt.xlabel(xlabel + '(' + str(n_quantiles) + '-quantiles)')
    plt.ylabel(ylabel)

@save_fig
def plot_hist(a, xlabel = '', title = '', bins = 20):
    plt.figure()
    plt.title(title)

    plt.hist(a, bins = bins)

    plt.xlabel(xlabel)
    plt.ylabel('Counts')

#decorator
def logged(thing_to_do):
    def wrapper(*args, **kwargs):
        if kwargs.has_key('log'):
            log = kwargs['log']
            kwargs.pop('log', None)
        else:
            log = True
        if kwargs.has_key('rep'):
            rep = kwargs['rep'] + '/'
        else:
            rep = ''

        if log:
            if kwargs.has_key('log_file'):
                log_file = kwargs['log_file']
                kwargs.pop('log_file', None)
            else:
                log_file = rep + 'log_ '+ thing_to_do.func_name + '.txt'
        else:
            log_file = None

        with write_on_file(log_file):
            ret = thing_to_do(*args, **kwargs)

        return ret
    return wrapper

def merge_dict(a, b):
    """
    First version of merge_dict. Used to merge dictionnaries with non-overlapping keys.
    """
    d = {}
    for k, v in a.items() + b.items():
        d[k] = v

    return d

def yes(question):
    ans=''
    while not ans in ['Y','y','yes','Yes','N','n','No','no']:
        ans=raw_input(question + ' (y/n)\n')
    return ans in ['Y','y','yes','Yes']

def getDistribution(data):
    """
    Get scipy continuous distribution built from data.
    Taken from http://stackoverflow.com/questions/10678546/creating-new-distributions-in-scipy
    """
    kernel = gaussian_kde(data)
    class rv(rv_continuous):
        def _rvs(self, *x, **y):
            return kernel.resample(int(self._size)) 
        def _cdf(self, x):
            return kernel.integrate_box_1d(-numpy.Inf, x)
        def _pdf(self, x):
            return kernel.evaluate(x)
    return rv(name='kdedist')

def bootstrap_test(sample1, sample2, k = 1000, p_value = 0.05, two_tailed = True):
    """
    Test the null hypothesis that the two samples are independent from each other 
    thanks to pearson coefficients.
    Note that we keep nan values during the resampling (and eliminate them to compute 
    the pearson coefficient). 
    """
    # eliminate all entries which have a nan in one of the sample. 
    
    sample1_bis, sample2_bis = zip(*[zz for zz in zip(sample1, sample2) if not np.isnan(zz[0]) and not np.isnan(zz[1])])
    r_sample = pearsonr(sample1_bis, sample2_bis)[0]
    
    n = len(sample1)
    try:
        assert n == len(sample2)
    except:
        Exception("Samples must have same sizes.")

    r_resample = np.zeros(k)
    for i in xrange(k):
        s1_rand = sample1[randint(0, n, n)] # Resampling with the same size
        s2_rand = sample2[randint(0, n, n)] 
        s1_rand_bis, s2_rand_bis = zip(*[zz for zz in zip(s1_rand, s2_rand) if not np.isnan(zz[0]) and not np.isnan(zz[1])])
        r_resample[i] = pearsonr(s1_rand_bis, s2_rand_bis)[0]
        
    ci = np.percentile(r_resample, [p_value/2., 1.-p_value/2.])
    
    #print "Percentiles:", ci
    
    return  ci[0]<r_sample<ci[1]

def get_sorted_indices(a):
    # Note: can also use argsort from numpy.
    return [i[0] for i in sorted(enumerate(a), key=lambda x:x[1])]

def insert_list_in_list(l1, l2, list_indices):
    """
    Insert l2 in l1 given the list of indices AFTER which 
    each element in l2 must be put in l1.
    Example:
    >>> l1 = [0, 4, 6]
    >>> l2 = [1, 2, 3, 5]
    >>> list_indices = [0, 0, 0, 1]
    returns [0, 1, 2, 3, 4, 5, 6]
    """
    assert len(list_indices)==len(l2)
    for i in range(len(l2)-1, -1, -1):
        l1.insert(list_indices[i]+1, l2[i])
    return l1

def build_triangular(N):
    """
    build a triangular lattice in a rectangle with N nodes along the abscissa (so 4*N**2 in total)
    
    argument:
        N: number of nodes along the abscissa
    output:
        G: Networkx Graph.
    """
    eps=10e-6
    G=nx.Graph()   
    a=1./float(N+0.5) - eps
    n=0
    j=0
    while j*sqrt(3.)*a <= 1.:
        i=0
        while i*a <= 1.:
            G.add_node(n, coords=(i*a, j*sqrt(3.)*a)) #LUCA: node capacity added.
            n+=1
            if i*a + a/2 < 1. and  j*sqrt(3.)*a + (sqrt(3.)/2.)*a < 1.:
                G.add_node(n, coords=(i*a + a/2., j*sqrt(3.)*a + (sqrt(3.)/2.)*a)) #LUCA: node capacity added.
                n+=1
            i+=1
        j+=1
            
    for n in G.nodes():
        for m in G.nodes():
            if n!=m and abs(sqrt((G.node[n]['coords'][0] - G.node[m]['coords'][0])**2\
            + (G.node[n]['coords'][1]- G.node[m]['coords'][1])**2) - a) <eps:
                G.add_edge(n,m)
    print(len(G.nodes()))
    
    return G

def _test_build_triangular():
    G=build_triangular(8.01783725737,5)
    
    plt.plot([G.node[n]['coords'][0] for n in G.nodes()], [G.node[n]['coords'][1] for n in G.nodes()], 'ro')
    for e in G.edges():
        plt.plot([G.node[e[0]]['coords'][0], G.node[e[1]]['coords'][0]], [G.node[e[0]]['coords'][1], G.node[e[1]]['coords'][1]], 'r-')
    plt.show()

def clean_network(G):
    """
    Reomve nodes with 0 degree:
    """
    for n in G.nodes()[:]:
        if G.degree(n)==0:
            G.remove_node(n)

    return G

if __name__=='__main__':
    #Tests-
    print 'n_days=', (delay([2011,3,1,0,0,0]) - delay([2010,12,1,0,0,0]))/(24*3600)
    print date_st(delay([2011,3,1,0,0,0]))
    print
    print 
    print date_st(delay([2010,12,1,0,0,0]))

    with clock_time():
        n_iter=10000
        for i in range(n_iter):
            j=i**2
        print 'coin'
        