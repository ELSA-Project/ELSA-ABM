#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===========================================================================
Advanced plots displaying different metrics against each other.
===========================================================================
"""

import sys
sys.path.insert(1,'..')
import pickle
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
from string import split
import os
from matplotlib import rc, rcParams
import pandas as pd

from utilities import read_paras
from iter_simO import build_path_average
from simulationO import build_path as build_path_single

pd.options.display.mpl_style = 'default'
#rcParams['mathtext.fontset'] = 'cm'
#rcParams['font.family'] = 'serif'
#rcParams['font.serif'] = ['Tahoma', 'Bitstream Vera Sans', 'Lucida Grande', 'Verdana']

version = '2.9.1'
main_version = split(version,'.')[0] + '.' + split(version,'.')[1]
loc = {'ur':1, 'ul':2, 'll':3, 'lr':4, 'r':5, 'cl':6, 'cr':7, 'lc':8, 'uc':9, 'c':10}

nice_colors = ['#348ABD',  '#7A68A6',  '#A60628',  '#467821',  '#CF4457',  '#188487',  '#E24A33']

colors = ('DarkRed','Blue','BlueViolet','Brown','CadetBlue','Crimson','DarkMagenta','DeepPink','Gold','Green','OrangeRed')

def rename_variable(var):
    if var=='nA':
        return r"$f_S$"
    elif var=='Delta_t':
        return r'$\Delta t$'
    else:
        return var

def build_path(paras, vers=main_version):
    rep=build_path_single(paras, vers=main_version, in_title=[n for n in ['Nfp', 'tau', 'par', 'ACtot', 'nA', 'departure_times'] if not n in paras['paras_to_loop']])
    for n in paras['paras_to_loop']:
        rep+='_loop_on_' + n
    return rep

# ---------- Plots with double loops ------------------ #

def global_met_vs_par(results, paras, met='satisfaction', name='', rep='.', labelx='', labely='', loc=3, norm_pairs_of_airports=False, norm_initial_values=False,\
                        figsize=(10, 7), xlim=None, ylim=None, anchor_box=(0., 0.), reduce_plot_x=0., labelpad=-5, put_first_last=False, zorders={}, fmts={}):
    """
    To have a global metric 'met' vs the par on which you loop. Gives several plots, corresponding to the outer loop.
    """
    
    if norm_pairs_of_airports and (paras['paras_to_loop'][1]!='density' or paras['paras_to_loop'][1]!='ACtot' or paras['paras_to_loop'][1]!='ACsperwave'):
        print 'You should not use norm_pairs of airports if the abscissa is not a number of flights !'
        #raise
    
    if name=='':
        name=met + '_vs_' + paras['paras_to_loop'][1]
        if norm_pairs_of_airports:
            name+= '_norm_pairs'
        if norm_initial_values:
            name+='_norm_initial_values'

    if labelx=='':
        labelx=rename_variable(paras['paras_to_loop'][-1])
        if norm_pairs_of_airports:
            labelx+= ' normalized by number of pairs of airports'
    if labely=='':
        labely='Global ' + met
        if norm_initial_values:
            labely+= ' normalized by initial values'

    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    
    plt.xlabel(labelx, fontsize=24, labelpad=labelpad)
    plt.ylabel(labely, fontsize=24)
    plt.tick_params(labelsize = 18)
    leg=[]
    i=0
    coin=sorted(results.keys())
    if put_first_last:
        coin.append(coin[0])
        coin.remove(coin[0])

    if norm_pairs_of_airports:
        len_pairs={G.name:len(G.pairs) for G in paras['G_iter']}
        overlaps={G.name:overlap_network(G) for G in paras['G_iter']}
        
    for p1 in coin:
        r1=results[p1]
        x=sorted(r1.keys())
        y=[r1[v][met]['avg'] for v in x]
        ey=[r1[v][met]['std'] for v in x]
        if paras['paras_to_loop'][1]=='par':
            x=np.log10(sorted([p[0][2]/p[0][0] for p in x]))
        if norm_pairs_of_airports:
            x=np.array(x)/float(len_pairs[p1.name]*overlaps[p1.name])
        if norm_initial_values:
            y=np.array(y)/np.array(y[0])

        if p1 in zorders.keys():
            zorder = zorders[p1]
        else:
            zorder = 1

        if p1 in fmts.keys():
            fmt = fmts[p1]
        else:
            fmt = 'o--'
        print p1
        print y
        plt.errorbar(x,y,ey,fmt=fmt,color=[i/float(len(results.keys())),0.,1.-i/float(len(results.keys()))], zorder=zorder)
        if paras['paras_to_loop'][0]=='par':
            #leg.append(r'$\alpha$' + '=' + str(p1[0][0]) + '; ' + r'$\beta$' + '=' + str(p1[0][2]))    
            if p1[0][0]!=0:
                leg.append(r'$\beta/\alpha=$' + str(p1[0][2]/p1[0][0]))    
            else:
                leg.append(r'$\beta/\alpha=\infty$')
        else:
            leg.append(rename_variable(paras['paras_to_loop'][0]) + '=' + str(p1))
        i+=1
    #plt.xlim([0.,100.])

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * (1-reduce_plot_x), box.height])
    if loc!=0:
        plt.legend(leg, loc=loc, fontsize=16, fancybox=True, shadow=True)  
    else:
        plt.legend(leg, fontsize=16, fancybox=True, shadow=True, bbox_to_anchor=anchor_box)  
    plt.xlim(xlim)
    plt.ylim(ylim)   
    plt.savefig(rep + '/' +name + ".png", close=False) 
    plt.savefig(rep + '/' +name + ".svg")  

def contour_global_met_vs_par(results, paras, met='satisfaction', name='', rep='.', labelx='', labely='', labelz='', loc=3, norm_pairs_of_airports=False, norm_initial_values=False,\
                        figsize=(10, 7), xlim=None, ylim=None, anchor_box=(0., 0.), reduce_plot_x=0., labelpad=-5, put_first_last=False, zorders={}, fmts={}, n_levels=10):
    """
    To have a global metric 'met' vs the par on which you loop. Gives several plots, corresponding to the outer loop.
    Contour version.
    """
    
    if norm_pairs_of_airports and (paras['paras_to_loop'][1]!='density' or paras['paras_to_loop'][1]!='ACtot' or paras['paras_to_loop'][1]!='ACsperwave'):
        print 'You should not use norm_pairs of airports if the abscissa is not a number of flights !'
        #raise
    
    if name=='':
        name=met + '_vs_' + paras['paras_to_loop'][1]
        if norm_pairs_of_airports:
            name+= '_norm_pairs'
        if norm_initial_values:
            name+='_norm_initial_values'
    name += '_contour'

    if labelx=='':
        labelx=rename_variable(paras['paras_to_loop'][-1])
        if norm_pairs_of_airports:
            labelx+= ' normalized by number of pairs of airports'
    if labely=='':
        labely='Global ' + met
        if norm_initial_values:
            labely+= ' normalized by initial values'

    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    
    plt.xlabel(labelx, fontsize=24, labelpad=labelpad)
    plt.ylabel(labely, fontsize=24)
    plt.tick_params(labelsize = 18)
    leg=[]
    i=0
    coin=sorted(results.keys())
    if put_first_last:
        coin.append(coin[0])
        coin.remove(coin[0])

    if norm_pairs_of_airports:
        len_pairs={G.name:len(G.pairs) for G in paras['G_iter']}
        overlaps={G.name:overlap_network(G) for G in paras['G_iter']}
    


    matrix = []
    y = []
    for p1 in coin:
        y.append(p1)
        r1=results[p1]
        x=sorted(r1.keys())
        z=[r1[v][met]['avg'] for v in x]
        #ey=[r1[v][met]['std'] for v in x]
        if paras['paras_to_loop'][1]=='par':
            x=np.log10(sorted([p[0][2]/p[0][0] for p in x]))
        if norm_pairs_of_airports:
            x=np.array(x)/float(len_pairs[p1.name]*overlaps[p1.name])
        if norm_initial_values:
            z=np.array(z)/np.array(z[0])

        matrix.append(z)
        # if p1 in zorders.keys():
        #     zorder = zorders[p1]
        # else:
        #     zorder = 1

        # if p1 in fmts.keys():
        #     fmt = fmts[p1]
        # else:
        #     fmt = 'o--'
        # print p1
        # print y
        #plt.errorbar(x,y,ey,fmt=fmt,color=[i/float(len(results.keys())),0.,1.-i/float(len(results.keys()))], zorder=zorder)
        # if paras['paras_to_loop'][0]=='par':
        #     #leg.append(r'$\alpha$' + '=' + str(p1[0][0]) + '; ' + r'$\beta$' + '=' + str(p1[0][2]))    
        #     if p1[0][0]!=0:
        #         leg.append(r'$\beta/\alpha=$' + str(p1[0][2]/p1[0][0]))    
        #     else:
        #         leg.append(r'$\beta/\alpha=\infty$')
        # else:
        #     leg.append(rename_variable(paras['paras_to_loop'][0]) + '=' + str(p1))
        i+=1
    #plt.xlim([0.,100.])
    print 
    plt.contourf(x, y, matrix, n_levels, # [-1, -0.1, 0, 0.1],
                    #alpha=0.5,
                    cmap=plt.cm.afmhot,
                    #origin=origin
                    )

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * (1-reduce_plot_x), box.height])
    # if loc!=0:
    #     plt.legend(leg, loc=loc, fontsize=16, fancybox=True, shadow=True)  
    # else:
    #     plt.legend(leg, fontsize=16, fancybox=True, shadow=True, bbox_to_anchor=anchor_box)  
    clb = plt.colorbar()
    clb.set_label(labelz, fontsize=20)
    clb.ax.tick_params(labelsize = 18)
    plt.xlim(xlim)
    plt.ylim(ylim)   
    plt.savefig(rep + '/' +name + ".png", close=False) 
    plt.savefig(rep + '/' +name + ".svg")
    
def global_met1_VS_met2(results, paras, met1='regulated_FPs', met2='satisfaction',name='', labelx='', labely='', rep='.', loc=3):
    """
    To have a global metric 'met2' vs 'met1'. Gives several plots, corresponding to the outer loop.
    """
    if name=='':
         name=met1 + '_vs_' + met2
         
    if labelx=='':
        labelx='Global ' + met2
    if labely=='':
        labely='Global ' + met1
    plt.figure()
    plt.xlabel(labelx)
    plt.ylabel(labely)
    leg=[]
    i=0
    coin=sorted(results.keys())
    for p1 in coin:
        r1=results[p1]
        x=[v[met2]['avg'] for v in r1.values()]
        y=[v[met1]['avg'] for v in r1.values()]
        ey=[v[met1]['std'] for v in r1.values()]
        plt.errorbar(x,y,ey,fmt='o',color=[i/float(len(results.keys())),0.,1.-i/float(len(results.keys()))])
        #leg.append(u'\u03b1' + '=' + str(p1[0][0]) + '; ' + u'\u03b2' + '=' + str(p1[0][2])) 
        leg.append(rename_variable(paras['paras_to_loop'][0]) + '=' + str(p1)) 
        i+=1
    plt.legend(leg, loc=loc)    
    plt.savefig(rep + '/' + name + ".png")    
    
def AC_met_vs_par(list_results_r, paras, met='satisfaction', name='', rep='.',AC=[],  labelx='', labely='', loc=3, figsize=(10, 7), xlim=None, ylim=None,\
                    anchor_box=(0., 0.), reduce_plot_x=0.):
    """
    To have the metric 'met' per AC vs the par on which you loop. Gives several plots, corresponding to the outer loop.
    """
    
    print AC
    
    if AC==(1.,0.,1000.):
        pop=r'AO$_R$'
    elif AC==(1.,0.,0.001):
        pop=r'AO$_S$'
    else:
        print "I did not recognize the company, I put pop='A'"
        pop=r'AO_$A$'

    if name=='':
         name=met + '_vs_' + paras['paras_to_loop'][1] + '_' + pop
    if labelx=='':
        labelx=rename_variable(paras['paras_to_loop'][-1])
    if labely=='':
        labely=met + ' of ' + pop
    plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    plt.xlabel(labelx, fontsize=24, labelpad=-10)
    plt.ylabel(labely, fontsize=24)
    plt.tick_params(labelsize=18)
    leg=[]
    i=0
    coin=sorted(list_results_r.items(), key=lambda a: a[0])
    for parconf,r1 in coin:
        x=sorted(r1.keys())
        y=[r1[v][met][AC]['avg'] for v in x]
        ey=[r1[v][met][AC]['std'] for v in x]
        plt.errorbar(x,y,ey,fmt='o--',color=[i/float(len(list_results_r.keys())),0.,1.-i/float(len(list_results_r.keys()))])
        #leg.append(u'\u03b1' + '=' + str(parconf[0][0]) + '; ' + u'\u03b2' + '=' + str(parconf[0][2]))    
        leg.append(rename_variable(paras['paras_to_loop'][0]) + '=' + str(parconf)) 
        i+=1
        
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * (1-reduce_plot_x), box.height])
    if loc!=0:
        plt.legend(leg, loc=loc, fontsize=16, fancybox=True, shadow=True)  
    else:
        plt.legend(leg, fontsize=16, fancybox=True, shadow=True, bbox_to_anchor=anchor_box)  
    
    plt.xlim(xlim)
    plt.ylim(ylim)  
    plt.savefig(rep + '/' + name + ".png", close=False) 
    plt.savefig(rep + '/' + name + ".svg")   
    
def ratio_AC_met_vs_par(list_results_r, paras, met='satisfaction', name='', rep='.',AC=[],  labelx='', labely='', loc=3):
    """
    To have the metric 'met' per AC vs the par on which you loop. Gives several plots, corresponding to the outer loop.
    """
    
    if AC==[]:
        AC=[paras['par'][0],paras['par'][1]]
    
    if AC[0]==(1.,0.,1000.):
        popA='R'
    elif AC[0]==(1.,0.,0.001):
        popA='S'
    else:
        print "I did not recognize the first company, I put popA='A'"
        popA='A'
    if AC[1]==(1.,0.,1000.):
        popB='R'
    elif AC[1]==(1.,0.,0.001):
        popB='S'
    else:
        print "I did not recognize the second company, I put popB='B'"
        popB='B'

    if name=='':
         name=met + '_vs_flights_AC_diff_' + popA + '_' + popB
    if labelx=='':
        labelx=rename_variable(paras['paras_to_loop'][-1])
    if labely=='':
        labely='Difference of ' + met + ' of AC (' + popA + '-' + popB + ')'
    plt.figure()
    plt.xlabel(labelx)
    plt.ylabel(labely)
    leg=[]
    i=0
    coin=sorted(list_results_r.items(), key=lambda a: a[0])
    for parconf,r1 in coin:
        x=sorted(r1.keys())
        y=[r1[v][met][AC[0]]['avg']/r1[v][met][AC[1]]['avg'] for v in x]
        ey=[(r1[v][met][AC[0]]['std']/r1[v][met][AC[0]]['avg'] + r1[v][met][AC[1]]['std']/r1[v][met][AC[1]]['avg'])*r1[v][met][AC[0]]['avg']/r1[v][met][AC[1]]['avg'] for v in x]
        #plt.errorbar(x,y,ey,fmt='o--',color=[i/float(len(list_results_r.keys())),0.,1.-i/float(len(list_results_r.keys()))])
        plt.plot(x,y,'--o',color=[i/float(len(list_results_r.keys())),0.,1.-i/float(len(list_results_r.keys()))])
        #leg.append(u'\u03b1' + '=' + str(parconf[0][0]) + '; ' + u'\u03b2' + '=' + str(parconf[0][2]))    
        leg.append(paras['paras_to_loop'][0] + '=' + str(parconf)) 
        i+=1
        
    plt.plot([x[0],x[-1]],[1.,1.],'k--')
    plt.legend(leg, loc=loc)    
    plt.savefig(rep + '/' + name + ".png")   
    
def difference_AC_met_vs_par(list_results_r, paras, met='satisfaction', name='', rep='.',AC=[],  labelx='', labely='', loc=3, norm=False, figsize=(10, 7), \
    xlim=None, ylim=None, anchor_box=(0., 0.), reduce_plot_x=0.):
    """
    To have the metric 'met' per AC vs the par on which you loop. Gives several plots, corresponding to the outer loop.
    """
    
    if AC==[]:
        AC=[paras['par'][0],paras['par'][1]]
    
    if AC[0]==(1.,0.,1000.):
        popA='R'
    elif AC[0]==(1.,0.,0.001):
        popA='S'
    else:
        print "I did not recognize the first company, I put popA='A'"
        popA='A'
    if AC[1]==(1.,0.,1000.):
        popB='R'
    elif AC[1]==(1.,0.,0.001):
        popB='S'
    else:
        print "I did not recognize the second company(", AC[1], "), I put popB='B'"
        popB='B'

    if name=='':
         name=met + '_vs_flights_AC_diff_' + popA + '_' + popB
    if labelx=='':
        labelx=rename_variable(paras['paras_to_loop'][-1])
    if labely=='':
        labely='Difference of ' + met + ' of AC (' + popA + '-' + popB + ')'
    plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    plt.xlabel(labelx, fontsize=24, labelpad=-10)
    plt.ylabel(labely, fontsize=24)
    plt.tick_params(labelsize=18)
    leg=[]
    i=0
    coin=sorted(list_results_r.items(), key=lambda a: a[0])
    for parconf,r1 in coin:
        x=sorted(r1.keys())
        if norm:
            y=[r1[v][met][AC[0]]['avg']/r1[1.][met][AC[0]]['avg'] - r1[v][met][AC[1]]['avg']/r1[0.][met][AC[1]]['avg'] for v in x]
            ey=[(r1[v][met][AC[0]]['std']/r1[1.][met][AC[0]]['avg'] + r1[v][met][AC[1]]['std']/r1[0.][met][AC[1]]['avg']) for v in x]
        else:
            y=[r1[v][met][AC[0]]['avg'] - r1[v][met][AC[1]]['avg'] for v in x]
            ey=[(r1[v][met][AC[0]]['std'] + r1[v][met][AC[1]]['std']) for v in x]

        old_y = y[:]
        x = [x[j] for j in range(len(ey)) if old_y[j]>-0.99]
        y = [y[j] for j in range(len(ey)) if old_y[j]>-0.99]
        ey = [ey[j] for j in range(len(ey)) if old_y[j]>-0.99]
        plt.errorbar(x, y, ey,fmt='o--',color=[i/float(len(list_results_r.keys())),0.,1.-i/float(len(list_results_r.keys()))])
        #plt.plot(x,y,'--o',color=[i/float(len(list_results_r.keys())),0.,1.-i/float(len(list_results_r.keys()))])
        #leg.append(u'\u03b1' + '=' + str(parconf[0][0]) + '; ' + u'\u03b2' + '=' + str(parconf[0][2]))    
        leg.append(rename_variable(paras['paras_to_loop'][0]) + '=' + str(parconf)) 
        i+=1
        
    plt.plot([x[0],x[-1]],[0.,0.],'k--')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * (1-reduce_plot_x), box.height])
    if loc!=0:
        plt.legend(leg, loc=loc, fontsize=16, fancybox=True, shadow=True)  
    else:
        plt.legend(leg, fontsize=16, fancybox=True, shadow=True, bbox_to_anchor=anchor_box)  
    
    plt.xlim(xlim)
    plt.ylim(ylim)    
    plt.savefig(rep + '/' + name + ".png", close=False)   
    plt.savefig(rep + '/' + name + ".svg")   
    
def just_legend(list_results_r, paras, met='satisfaction', name='', rep='.'):
    if name=='':
         name=met + '_just_legend'
    plt.figure()
    leg=[]
    i=0
    coin=sorted(list_results_r.items(), key=lambda a: a[0])
    for parconf,r1 in coin:
        plt.plot([0,0],[0,1],'o--',color=[i/float(len(list_results_r.keys())),0.,1.-i/float(len(list_results_r.keys()))])
        leg.append(rename_variable(paras['paras_to_loop'][0]) + '=' + str(parconf)) 
        i+=1
    plt.legend(leg)
    plt.savefig(rep + '/' + name + ".png")   
#def tranversal_global_met_vs_par(list_results_r, met='satisfaction', name='', rep=''):
#    """
#    Same as global_met_vs_par, inverting the two levels of loop.
#    """
#    if name=='':
#         name=met + '_vs_flights'
#    plt.figure()
#    plt.xlabel('nA')
#    plt.ylabel('Global ' + met)
#    leg=[]
#    i=0
#    #coin=sorted(list_results_r.items(), key=lambda a: a[0])
#    
#    keys=sorted(list_results_r.values()[0].keys())
#    for k in keys:
#        x=list_results_r.keys()
#        y=[v[k]['global'][met]['avg'] for v in list_results_r.values()]
#        ey=[v[k]['global'][met]['std'] for v in list_results_r.values()]
#        plt.errorbar(x,y,ey,fmt='o',color=[i/float(len(list_results_r.keys())),0.,1.-i/float(len(list_results_r.keys()))])
#        #leg.append(u'\u03b1' + '=' + str(parconf[0][0]) + '; ' + u'\u03b2' + '=' + str(parconf[0][2]))    
#        leg.append('ACtot=' + str(k)) 
#        i+=1
#        
#    plt.legend(leg)    
#    plt.savefig(rep + name + ".png")   
    
def normalized_met_vs_par(results, paras, met='satisfaction', name='', rep='.', labelx='', labely='', loc=3, xlim=None, ylim=None,\
                        figsize=(10, 7), reduce_plot_x=0.):
    """
    Same as previous one, for normalized satisfaction.
    """
    try:
        assert paras['paras_to_loop'][-1]=='nA'
    except:
        print 'The inner loop should be over nA! Try to swap the order in paras_to_loop.'
        raise
    
    if name=='':
         name='normalized_' + met + '_vs_flights'
             
    if labelx=='':
        labelx=rename_variable('nA')
    if labely=='':
        #labely='Normalized ' + met
        labely=r'$\tilde{\mathcal{S}}^{TOT}$'
        
    plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    plt.xlabel(labelx, fontsize=24, labelpad=-5)
    plt.ylabel(labely, fontsize=24)
    plt.tick_params(labelsize = 18)
    leg=[]
    i=0

    max_nA = 2.
    min_nA = -1.
    print results.keys()
    ACs=results.values()[0].values()[0].values()[0].keys()
    print "ACs:", ACs
    coin=sorted(results.keys())
    for p1 in coin:
        r1=results[p1]
        x=sorted(r1.keys())
        y=[((r1[nA][met][ACs[1]]['avg']/r1[max_nA][met][ACs[1]]['avg'])*nA + \
            (r1[nA][met][ACs[0]]['avg']/r1[min_nA][met][ACs[0]]['avg'])*(1.-nA)) for nA in x] #loop sur les nA
        
        ey=[(r1[nA][met][ACs[1]]['std']/r1[nA][met][ACs[1]]['avg']+r1[max_nA][met][ACs[1]]['std']/r1[max_nA][met][ACs[1]]['avg'])*(r1[nA][met][ACs[1]]['avg']/r1[max_nA][met][ACs[1]]['avg'])*nA\
        +(r1[nA][met][ACs[0]]['std']/r1[nA][met][ACs[0]]['avg']+r1[min_nA][met][ACs[0]]['std']/r1[min_nA][met][ACs[0]]['avg'])*(r1[nA][met][ACs[0]]['avg']/r1[min_nA][met][ACs[0]]['avg'])*(1.-nA)\
        for nA in x]
        
        plt.plot(x,y,'o--',color=[i/float(len(results.keys())),0.,1.-i/float(len(results.keys()))])
        leg.append(rename_variable(paras['paras_to_loop'][0]) + '=' + str(p1)) 
        i+=1
        
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * (1-reduce_plot_x), box.height])
    if loc!=0:
        plt.legend(leg, loc=loc, fontsize=16, fancybox=True, shadow=True)  
    else:
        plt.legend(leg, fontsize=16, fancybox=True, shadow=True, bbox_to_anchor=anchor_box)  

    plt.xlim(xlim)
    plt.ylim(ylim) 
    plt.savefig(rep + '/' + name + ".png", close=False)
    plt.savefig(rep + '/' + name + ".svg")      
    
def loads_vs_par(results, paras, norm=False,  name='', rep='.', labelx='', labely=''):
    """
    Average loads and normalized loads of the sectors.
    NEED AN UPDATE.
    """
    if norm:
        met='loads_norm'
    else:
        met='loads'
    if name=='':
        name=met + '_vs_flights'
    plt.figure()
    plt.xlabel('ACtot')
    plt.ylabel(met)
    leg=[]
    i=0
    #coin=sorted(list_results_r.items(), key=lambda a: a[0])
    
    keys=sorted(results.keys())
    for k in keys: # Loop sur les ACtot
        results=results[k]
        x=sorted(results.keys())
        y=[results[xx][met]['avg'] for xx in x]
        ey=[results[xx][met]['std'] for xx in x]

        plt.errorbar(x,y,ey,fmt='o--',color=[i/float(len(results.keys())),0.,1.-i/float(len(list_results_r.keys()))])
        #leg.append(u'\u03b1' + '=' + str(parconf[0][0]) + '; ' + u'\u03b2' + '=' + str(parconf[0][2]))    
        leg.append(rename_variable('nA') + '=' + str(k)) 
        i+=1
        
    plt.legend(leg)    
    plt.savefig(rep + '/' + name + ".png",dpi=100)   


# ------------- Single loop plots -------------------- #

def global_vs_main_loop(results, paras, met='satisfaction', name='', rep='.', err='big'):
    if name=='':
        name='/' + met + '_vs_' + paras['paras_to_loop'][0]
    plt.figure()
    plt.xlabel(rename_variable(paras['paras_to_loop'][0]))
    plt.ylabel(met)
    
    x=sorted(results.keys())
    y=[results[xx][met]['avg'] for xx in x]
    ey=[results[xx][met]['std'] for xx in x]

    if paras['paras_to_loop'][0]=='par':
        x=np.log10(sorted([p[0][2]/p[0][0] for p in x]))
	if err=='small':
        ey=np.array(ey)/sqrt(float(paras['n_iter']))

    plt.errorbar(x,y,ey,fmt='o--',color='r')
    
    plt.savefig(rep + '/' + name + ".png",dpi=100)
    
def per_AC_vs_main_loop(results, paras, met='satisfaction', name='', rep='.', ACs=[],\
 pop='A', labelx='', labely=''):

    """
    To have the metric 'met' per AC vs the par on which you loop.
    """

	print 'Doing plots', met, 'for company', AC

    plt.figure()

    if labelx=='':
        labelx=rename_variable(paras['paras_to_loop'][0])
    if labely=='':
        labely=met + ' of '
        for AC in ACs:
            if AC==(1.,0.,1000.):
                pop=r'AO$_R$'
            elif AC==(1.,0.,0.001):
                pop=r'AO$_S$'
            else:
                print "I did not recognize the company, I put pop='A'"
                pop=r'AO_$A$'

            labely+=pop + ', '
        labely = labely[:-2]

    if name=='':
        name = met + '_per_AC_vs_' + paras['paras_to_loop'][0]

    for i, AC in enumerate(ACs):
        if AC==(1.,0.,1000.):
            pop=r'AO$_R$'
        elif AC==(1.,0.,0.001):
            pop=r'AO$_S$'
        else:
            print "I did not recognize the company, I put pop='A'"
            pop=r'AO_$A$'
        
        x=sorted(results.keys())
        y=[results[xx][met][AC]['avg'] for xx in x]
        ey=[results[xx][met][AC]['std'] for xx in x]
        plt.errorbar(x, y, ey, fmt='o--', color=colors[i], label = pop)

    plt.xlabel(labelx)
    plt.ylabel(labely)
    plt.legend(loc = loc['ul'])
    
    plt.savefig(rep + '/' + name + ".png",dpi=100)
    
    
# --------------------------------------------------------------------- #
    
def get_results(paras, vers='2.9'):
    results, results_global={}, {}
    results, results_global=loop({p:paras[p + '_iter'] for p in paras['paras_to_loop']}, paras['paras_to_loop'], paras, results, results_global, vers=vers)
    return results, results_global

def loop(a, level, parass, results, results_global, vers=version):
    """
    New in 2.6: Makes an arbitrary number of loops
    a: dictionnary, with keys as parameters to loop on and values as the values on which to loop.
    level: list of parameters on which to loop. The first one is the most outer loop, the last one is the most inner loop.
    """
    if level==[]:
        #rep, name= build_path_average(paras, vers=vers)
        # print parass['par']
        # print parass['AC_dict']
        
        rep, name = build_path_average(parass, vers=vers)
        with open(rep + name, 'r') as f:
            results = pickle.load(f)

        results_global={met:{k:sum([results[met][p][k]*parass['AC_dict'][p]/float(parass['ACtot']) \
                for p in results[met].keys() if parass['AC_dict'].has_key(p)]) for k in ['avg', 'std']} for met in ['satisfaction', 'regulated_FPs', 'regulated_F']} 

        return results,results_global
    else:
        assert level[0] in a.keys()
        for i in a[level[0]]:
            # print level[0], '=', i
            parass.update(level[0],i)
            # print parass['AC_dict']
            results[i], results_global[i]=loop(a, level[1:], parass, {}, {}, vers=vers)
            
    return results, results_global
    
def overlap_paths(path1, path2, typ='site'): # or 'link'
    assert typ=='site' or typ=='link'
    if typ=='site':
        N=0
        for n in path1:
            if n in path2:
                N+=1
    if typ=='link':
        N=0
        for i in range(len(path1)-1):
            if path1[i] in path2:
                j=path2.index(path1[i])
                if path1[i+1]==path2[j+1]:
                    N+=1
    return N
    
def overlap_network(G, typ='site', n_shortest_paths = 1):
    overlap=[]
    if n_shortest_paths>G.Nfp:
        raise
    else:
        for i1,j1 in G.pairs:
            for i2,j2 in G.pairs:
                if (i1,j1)!=(i2,j2): 
                    for path1 in G.short[(i1,j1)][:n_shortest_paths]:
                        for path2 in G.short[(i2,j2)][:n_shortest_paths]:
                            overlap.append(overlap_paths(path1, path2, typ=typ)/float(min(len(path1), len(path2))))
    return np.mean(overlap)
#def take_several_graphs()

if __name__=='__main__':
    paras = read_paras()

    results_list = []
    vers = '2.9' 
    if 0:
        a = paras['paras_to_loop'][0]
        paras['paras_to_loop'][0] = paras['paras_to_loop'][1]
        paras['paras_to_loop'][1] = a
    print 'Loading data...'  
        
    rep = build_path(paras, vers=vers)
    print rep

    # os.system('mkdir -p ' + rep)
    
    # results, results_global = get_results(paras, vers=vers)

    # if 0: #average on one level
    #     level_average = 0
    #     if len(paras['paras_to_loop'])==3:
    #         P = pd.Panel(results_global)
    #         df = P.mean(level_average)
    #         results_global = df.to_dict()
    #         paras['paras_to_loop'].remove(paras['paras_to_loop'][level_average])
    #     else:
    #         raise Exception("Not implemented")

    # #print results
    # # for k, v in results.items():
    # #     print k
    # #     print v['satisfaction']
    
    # if len(paras['paras_to_loop'])==2:
    #     # Dooble loop
    #     global_met_vs_par(results_global,paras, rep=rep, 
    #                                             loc=loc['ur'], 
    #                                             #figsize=(12, 7),
    #                                             #labelx=r'$\log(\beta/\alpha)$',
    #                                             labely=r'$\mathcal{S}$',#'Satisfaction'
    #                                             norm_initial_values=False,
    #                                             #xlim=(0., 1.),
    #                                             #ylim=(0., 1.),
    #                                             #anchor_box=(1.24, 1.),
    #                                             #reduce_plot_x=0.1
    #                                             )
    #     contour_global_met_vs_par(results_global,paras, rep=rep, 
    #                                             loc=loc['ur'], 
    #                                             #figsize=(12, 7),
    #                                             #labelx='',
    #                                             labely=r'Number of flights plans',
    #                                             labelz=r'$\mathcal{S}$',
    #                                             norm_initial_values=False,
    #                                             n_levels=1000,
    #                                             #xlim=(0., 1.),
    #                                             #ylim=(0., 1.),
    #                                             #anchor_box=(1.22, 0.55),
    #                                             #reduce_plot_x=0.1
    #                                             )

    #     # This is for loop on ACperwaves and loop on par.
    #     # global_met_vs_par(results_global,paras, rep=rep, 
    #     #                                         loc=loc['ur'], 
    #     #                                         norm_initial_values=False, 
    #     #                                         figsize=(10, 7),
    #     #                                         labelx='Number of flights',
    #     #                                         labely=r'$\mathcal{S}$'#'Satisfaction'
    #     #                                         )

    #     # This is for loop on ACperwaves and loop on par for referee (alpha=0 and beta=0).
    #     # global_met_vs_par(results_global,paras, rep=rep, 
    #     #                                         loc=0, 
    #     #                                         norm_initial_values=False, 
    #     #                                         figsize=(10, 7),
    #     #                                         put_first_last=True,
    #     #                                         labelx='Number of flights',
    #     #                                         labely=r'$\mathcal{S}$',#'Satisfaction'
    #     #                                         zorders={((1., 0., 0.), (1., 0., 1.)):6, ((0., 0., 1.), (1., 0., 1.)):6},
    #     #                                         anchor_box=(1.1, 1.1),
    #     #                                         fmts={((1., 0., 0.), (1., 0., 1.)):'-', ((0., 0., 1.), (1., 0., 1.)):'-'}
    #     #                                         )

    #     # this is for loop delta t loop par
    #     # global_met_vs_par(results_global,paras, rep=rep, 
    #     #                                         loc=loc['ul'], 
    #     #                                         figsize=(10, 7),
    #     #                                         labelx=r'$\log(\beta/\alpha)$',
    #     #                                         labely=r'$\mathcal{S}$',#'Satisfaction'
    #     #                                         norm_initial_values=False)
    
    #     # This is for loop_na and loop_delta t
    #     # global_met_vs_par(results_global,paras, rep=rep, 
    #     #                                         loc=0, 
    #     #                                         figsize=(12, 7),
    #     #                                         #labelx=r'$\log(\beta/\alpha)$',
    #     #                                         labely=r'$\mathcal{S}^{TOT}$',#'Satisfaction'
    #     #                                         norm_initial_values=False,
    #     #                                         xlim=(0., 1.),
    #     #                                         ylim=(0., 1.),
    #     #                                         anchor_box=(1.22, 0.55),
    #     #                                         reduce_plot_x=0.1)

    #     # normalized_met_vs_par(results,paras, rep=rep, 
    #     #                                         loc=loc['ur'], 
    #     #                                         #figsize=(12, 7),
    #     #                                         #labelx=r'$\log(\beta/\alpha)$',
    #     #                                         #labely=r'$\mathcal{S}^{TOT}$',#'Satisfaction'
    #     #                                         xlim=(0., 1.),
    #     #                                         ylim=(1., 1.8),
    #     #                                         #anchor_box=(1.22, 0.55),
    #     #                                         #reduce_plot_x=0.1)
    #     #                                         )

        
    #     #global_met_vs_par(results_global,paras, rep=rep, loc=loc['lr'], met='regulated_F')
    #     #global_met_vs_par(results_global,paras, rep=rep, met='regulated_FPs',loc=4)
    #     #normalized_met_vs_par(results,paras, rep=rep, loc=0)#loc['ll'])
    #     #global_met1_VS_met2(results_global, paras, rep=rep)
    #     #AC_met_vs_par(results, paras, AC=paras['par'][0], rep=rep, loc=loc['ll'])
    #     if 'nA' in paras['paras_to_loop']:
    #         # AC_met_vs_par(results, paras, AC=paras['par'][0],rep=rep, loc=loc['ll'])
    #         # AC_met_vs_par(results, paras, AC=paras['par'][1],rep=rep, loc=loc['lr'])
    #         # AC_met_vs_par(results, paras, AC=paras['par'][0],rep=rep, loc=loc['ll'], labely=r'$\mathcal{S}^S$', xlim=(0., 1.))
    #         # AC_met_vs_par(results, paras, AC=paras['par'][1],rep=rep, loc=0, labely=r'$\mathcal{S}^R$', xlim=(0., 1.), anchor_box=(1.12, 0.55))
    #         pass
    #     #just_legend(results, paras, rep=rep)
    #     #ratio_AC_met_vs_par(results, paras, rep=rep, loc=0)
    #     # difference_AC_met_vs_par(results, paras,rep=rep, 
    #     #                                         figsize=(12, 7),
    #     #                                         loc=0,
    #     #                                         labely=r'$\mathcal{S}^{(S)} - \mathcal{S}^{(R)}$',
    #     #                                         anchor_box=(1.24, 0.9),
    #     #                                         reduce_plot_x=0.1)
    #     print 'Graphs saved in ', rep
    
    # else:
    #     #Single loop   
    #     #print results_global
    #     #print paras['AC_dict']
    #     global_vs_main_loop(results_global, paras, met='satisfaction', rep=rep)#, err='small')
    #     global_vs_main_loop(results_global, paras, met='regulated_FPs', rep=rep)#, err='small')
    #     global_vs_main_loop(results_global, paras, met='regulated_F', rep=rep)#, err='small')
    #     if 'nA' in paras['paras_to_loop']:
    #         per_AC_vs_main_loop(results, paras, ACs=paras['par'], rep = rep)
    #         #per_AC_vs_main_loop(results, paras, AC=paras['par'][1])
    
    # plt.show()
    
    # os.system('cp ABMvars.py ' + rep + '/')
    # print
    # print
    # print 'Done. Plots saved in ', rep
    
