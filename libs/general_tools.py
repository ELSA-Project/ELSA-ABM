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

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    rgb_255 = tuple(int(value[i:i+lv/3], 16) for i in range(0, lv, lv/3))
    return tuple(a/255. for a in rgb_255)

nice_colors = ['#348ABD',  '#7A68A6',  '#A60628',  '#467821',  '#CF4457',  '#188487',  '#E24A33']
nice_colors = [hex_to_rgb(v) for v in nice_colors]

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

@contextlib.contextmanager
def stdout_redirected(to=os.devnull):
    '''
    found here:
    http://stackoverflow.com/questions/5081657/how-do-i-prevent-a-c-shared-library-to-print-on-stdout-in-python
    
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''
    fd = sys.stdout.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w') # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout) # restore stdout.
                                            # buffering and flags such as
                                            # CLOEXEC may be different
            #print "log written in", to

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

# Taken from https://gist.github.com/pv/8036995

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """
 
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")
 
    new_regions = []
    new_vertices = vor.vertices.tolist()
 
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2
 
    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))
 
    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]
 
        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue
 
        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]
 
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue
 
            # Compute the missing endpoint of an infinite ridge
 
            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal
 
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius
 
            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())
 
        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]
 
        # finish
        new_regions.append(new_region.tolist())
 
    return new_regions, np.asarray(new_vertices)

class TrajConverter(object):
    """
    General Trajectory converter.
    The formats can essentially be:
     -- (x, y, z, t) : trajectories are made of 4-tuples with lat, lon, altitude and 
    time
     -- (x, y, z, t, s) : same with sector as fifth element.
     -- (n), t : trajectories are made ONE 2-tuple. The first one is a list of labels
    of nodes, the second one is the time of entrance, i.e. the time of the first 
    point
     -- (n, z), t : same with altitude attached to each point.

    The format of time can be either:
     -- t : a float representing the number of minutes elapsed since the beginning of 
    the day (which is stored somewhere else).
     -- tt : a tuple (yy, mm, dd, h, m , s).
    """

    def __init__(self):
        pass

    def set_G(self, G):
        self.G = G

    def check_trajectories(self, trajectories):
        """
        Check that the trajectories are compliant with the nodes of the network.
        """
        # TODO
        pass

    def convert(self, trajs, fmt_in, fmt_out, **kwargs):
        """
        General converter. Some information can be lost in the 
        conversion, depending on the input/output format.
        """

        accepted_fmts = ['(x, y, z, t)', '(x, y, z, t, s)', '(n), t', '(n, z), t']

        try:
            assert fmt_in in accepted_fmts and fmt_out in accepted_fmts
        except AssertionError:
            print "Unrecognized format." 
            raise

        if fmt_in==fmt_out:
            return trajs

        if fmt_in in ['(n), t', '(n, z), t']:
            if fmt_out in ['(x, y, z, t)', '(x, y, z, t, s)']:
                if fmt_out=='(x, y, z, t)':
                    kwargs['put_sectors'] = False
                elif fmt_out=='(x, y, z, t, s)':
                    kwargs['put_sectors'] = True

                return self._convert_trajectories_n_to_x(trajs, fmt_in=fmt_in, **kwargs)
            else:
                raise Exception("Not implemented yet.") #TODO
                if fmt_in=='(n), t' and fmt_in=='(n, z), t':
                    print "I will just add 0 for altitudes..."
                    
        elif fmt_in in ['(x, y, z, t)', '(x, y, z, t, s)']:
            if fmt_out in ['(n), t', '(n, z), t']:
                print "Warning: Converting coordinate-based to nav-based trajectories can result in errors."
                print "Check the value of the threshold."

                # Beware, this function returns also the network !
                return self._convert_trajectories_x_to_n(trajs, fmt_in=fmt_in, **kwargs)

            elif fmt_out in ['(x, y, z, t)', '(x, y, z, t, s)']:
                if fmt_in=='(x, y, z, t)':
                    print "I am just adding dummy sectors (0)"
                    raise Exception("Not implemented yet.")
                else:
                    x, y, z, t, s = tuple(zip(*trajs))
                    return list(zip(x, y, z, t))

    def _convert_trajectories_x_to_n(self, trajs, fmt_in='(x, y, z, t)', fmt_out='(n), t',\
        keyword_times='weight', keyword_coord='coord', **kwargs):
        """
        Convert coordinate-based to nav-based trajectories using spatial proximity with 
        nodes of the network self.G
        """

        # Make the list of all the coordinates
        #coords_list = [(point[0], point[1]) for traj in trajs for point in traj]
        points_list = [point for traj in trajs for point in traj]
        
        assigments = self._cluster_points([(p[0], p[1]) for p in points_list], **kwargs)
        print "Found", len(set(assigments.values())), "nodes."

        # Generate '(n), t' and compute the times of flight between nodes
        times = {}
        new_trajs = []
        idx = 0
        for traj in trajs:
            new_traj = []
            for i, point in enumerate(traj):
                if fmt_out=='(n), t':
                    new_traj.append(assigments[idx])
                elif fmt_out=='(n, z), t':
                    new_traj.append((assigments[idx], point[2]))
                if i<len(traj)-1:
                    key = (assigments[idx], assigments[idx+1])
                    times[key] = times.get(key, []) + [delay(traj[i+1][3]) - delay(point[3])]
                idx += 1
            new_trajs.append((new_traj, traj[0][3]))

        # Do the average of times.
        for k, list_v in times.items():
            times[k] = np.mean(list_v)

        # Build the edges of the network
        G = nx.Graph()
        for (n1, n2), t in times.items():
            G.add_edge(n1, n2)
            G[n1][n2][keyword_times] = t/60.

        # Compute the coordinates of the nodes
        nodes_coords = {}
        nodes_pop = {}
        for idx, node in assigments.items():
            nodes_coords[node] = nodes_coords.get(node, np.array((0., 0.))) + np.array([points_list[idx][0], points_list[idx][1]])
            nodes_pop[node] = nodes_pop.get(node, 0.) + 1.
        for node, coords in nodes_coords.items():
            G.node[node][keyword_coord] = tuple(coords/nodes_pop[node])

        self.G = G

        return new_trajs

    def _cluster_points(self, coords_list, thr=10**(-4.)):
        """
        This procedure is a quick hack. It might fail completely if the points 
        are too homogeneously distributed or if the threshold is too high.
        """

        n_nodes = 0
        assignements = {}
        for i, cc1 in enumerate(coords_list):
            for j, cc2 in enumerate(coords_list):
                if j>i:
                    if np.linalg.norm((np.array(cc1) - np.array(cc2)))<thr:
                        if not i in assignements.keys() and not j in assignements.keys():
                            assignements[i] = n_nodes
                            assignements[j] = n_nodes
                            n_nodes += 1
                        elif i in assignements.keys() and not j in assignements.keys():
                            assignements[j] = assignements[i]
                        elif not i in assignements.keys() and j in assignements.keys():
                            assignements[i] = assignements[j]

        for i in range(len(coords_list)):
            if not i in assignements.keys():
                assignements[i] = n_nodes
                n_nodes += 1

        return assignements

    def _convert_trajectories_n_to_x(self, trajectories, fmt_in='(n), t', **kwargs):
        """
        General converter of format of trajectories. The output is meant to be 
        compliant with the tactical ABM, i.e either (x, y, z, t) or (x, y, z, t, s).

        Parameters
        ----------
        G : Net object
            Needed in order to compute travel times between nodes.
        trajectories : list
            of trajectories in diverse format
        fmt_in : string
            Format of input.
        kwargs : additional parameters
            passed to other methods.

        Returns
        -------
        trajectories : list
            of converted trajectories with signature (x, y, z, t) or (x, y, z, t, s)
        Notes
        -----
        Needs expansion to support other conversion. Maybe make a class.
        Needs to specify format of output.

        """
        
        if fmt_in=='(n), t':
            return self._convert_trajectories_no_alt(trajectories, **kwargs)
        elif fmt_in=='(n, z), t':
            return self._convert_trajectories_alt(trajectories, **kwargs)
        else:
            raise Exception("format", fmt, "is not implemented")

    def _convert_trajectories_no_alt(self, trajectories, put_sectors=False, input_minutes=False,
        remove_flights_after_midnight=False, starting_date=[2010, 5, 6, 0, 0, 0]):
        """
        Convert trajectories with navpoint names into trajectories with coordinate and time stamps.

        trajectories signature in input:
        (n), t
        trajectories signature in output:
        (x, y, 0, t) or (x, y, 0, t, s)

        Altitudes in output are all set to 0.

        Parameters
        ----------
        G : Net object
            Used to have the coordinates of points and the travel times between nodes.
        trajectories : list
        put_sectors : boolean, optional
            If True, output format is (x, y, 0, t, s)
        input_minutes : boolean, optional
            Used to cope with the fact that the coordinates stored in the network can be in 
            degree or minutes of degree.
        remove_flights_after_midnight : boolean, True
            if True, remove from the list all flights landing ther day after starting_date
        starting_date : list of tuple 
            of format [yy, mm, dd, h, m , s]

        Returns
        -------
        trajectories_coords : list
            of trajectories with format (x, y, 0, t) or (x, y, 0, t, s)
        
        """ 

        trajectories_coords = []
        for i, (trajectory, d_t) in enumerate(trajectories):
            traj_coords = []
            for j, n in enumerate(trajectory):
                if not input_minutes:
                    x = self.G.node[n]['coord'][0]
                    y = self.G.node[n]['coord'][1]
                else:
                    x = self.G.node[n]['coord'][0]/60.
                    y = self.G.node[n]['coord'][1]/60.
                t = d_t if j==0 else  list(date_st(delay(t) + 60.*self.G[n][trajectory[j-1]]['weight']))
                if remove_flights_after_midnight and list(t[:3])!=list(starting_date[:3]):
                    break
                if not put_sectors:
                    traj_coords.append([x, y, 0., t])
                else:
                    traj_coords.append([x, y, 0., t, self.G.node[n]['sec']])
            if not remove_flights_after_midnight or list(t[:3])==list(starting_date[:3]):
                trajectories_coords.append(traj_coords)

        if remove_flights_after_midnight:
            print "Dropped", len(trajectories) - len(trajectories_coords), "flights because they arrive after midnight."
        return trajectories_coords

    def _convert_trajectories_alt(self, trajectories, put_sectors=False, input_minutes=False,
        remove_flights_after_midnight=False, starting_date=[2010, 5, 6, 0, 0, 0]):
        """
        Convert trajectories with navpoint names into trajectories with coordinate and time stamps.
        
        trajectories signature in input:
        (n, z), t
        trajectories signature in output:
        (x, y, z, t) or (x, y, z, t, s)

        Parameters
        ----------
        G : Net object
            Used to have the coordinates of points and the travel times between nodes.
        trajectories : list
        put_sectors : boolean, optional
            If True, output format is (x, y, z, t, s)
        input_minutes : boolean, optional
            Used to cope with the fact that the coordinates stored in the network can be in 
            degree or minutes of degree.
        remove_flights_after_midnight : boolean, True
            if True, remove from the list all flights landing ther day after starting_date
        starting_date : list of tuple 
            of format [yy, mm, dd, h, m , s]

        Returns
        -------
        trajectories_coords : list
            of trajectories with format (x, y, z, t) or (x, y, z, t, s)

        """ 

        trajectories_coords = []
        for i, (trajectory, d_t) in enumerate(trajectories):
            traj_coords = []
            for j, (n, z) in enumerate(trajectory):
                if not input_minutes:
                    x = self.G.node[n]['coord'][0]
                    y = self.G.node[n]['coord'][1]
                else:
                    x = self.G.node[n]['coord'][0]/60.
                    y = self.G.node[n]['coord'][1]/60.
                t = d_t if j==0 else date_st(delay(t) + 60.*self.G[n][trajectory[j-1][0]]['weight'])
                if remove_flights_after_midnight and list(t[:3])!=list(starting_date[:3]):
                    break
                if not put_sectors:
                    traj_coords.append([x, y, z, list(t)])
                else:
                    if 'sec' in self.G.node[n].keys():
                        sec = self.G.node[n]['sec']
                    else:
                        sec = 0
                    traj_coords.append([x, y, z, list(t), sec])
            if not remove_flights_after_midnight or list(t[:3])==list(starting_date[:3]):
                trajectories_coords.append(traj_coords)

        if remove_flights_after_midnight:
            print "Dropped", len(trajectories) - len(trajectories_coords), "flights because they arrive after midnight."
        return trajectories_coords


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
        