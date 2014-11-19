#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 13:53:44 2013

@author: luca

===========================================================================
Advanced plots displaying different metrics against each other.
===========================================================================
"""

#import os
import pickle
from math import sqrt
from ABMvars import paras
import matplotlib.pyplot as plt
from iter_simO import build_path_average
from simulationO import build_path as build_path_single
import numpy as np
from string import split
import os
from matplotlib import rc

#rc('text', usetex=True)
#rc('font', family='serif')

version='2.9.0'
main_version=split(version,'.')[0] + '.' + split(version,'.')[1]
loc={'ur':1, 'ul':2, 'll':3, 'lr':4, 'r':5, 'cl':6, 'cr':7, 'lc':8, 'uc':9, 'c':10}

colors=('Blue','BlueViolet','Brown','CadetBlue','Crimson','DarkMagenta','DarkRed','DeepPink','Gold','Green','OrangeRed')

def rename_variable(var):
    if var=='nA':
        return r"$f_S$"
    elif var=='Delta_t':
        return r'$\Delta t$'
    else:
        return var

def build_path(paras, vers=main_version):
    rep=build_path_single(paras, vers=vers, in_title=[n for n in ['Nfp', 'tau', 'par', 'ACtot', 'nA', 'departure_times','Nsp_nav', 'n_iter'] if not n in paras['paras_to_loop']])
    for n in paras['paras_to_loop']:
        rep+='_loop_on_' + n
    return rep

# ---------- Plots with double loops ------------------ #

def global_met_vs_par(results, paras, met='satisfaction', name='', rep='.', labelx='', labely='', loc=3, norm_pairs_of_airports=False, norm_initial_values=False):
    """
    To have a global metric 'met' vs the par on which you loop. Gives several plots, corresponding to the outer loop.
    """
    
    if norm_pairs_of_airports and (paras['paras_to_loop'][1]!='density' or paras['paras_to_loop'][1]!='ACtot' or paras['paras_to_loop'][1]!='ACsperwave'):
        print 'You should not use norm_pairs of airports if the abscissa is not a number of flights !'
        #raise
    
    if name=='':
        name=met + '_vs_' + paras['paras_to_loop'][1]#flights'
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
            
    plt.figure()
    plt.xlabel(labelx)
    plt.ylabel(labely)
    leg=[]
    i=0
    coin=sorted(results.keys())
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
        print x
        print
        print y 
        print
        print ey
        if norm_pairs_of_airports:
            x=np.array(x)/float(len_pairs[p1.name]*overlaps[p1.name])
        if norm_initial_values:
            y=np.array(y)/np.array(y[0])

        plt.errorbar(x,y,ey,fmt='o--',color=[i/float(len(results.keys())),0.,1.-i/float(len(results.keys()))])
        if paras['paras_to_loop'][0]=='par':
            leg.append(r'$\alpha$' + '=' + str(p1[0][0]) + '; ' + r'$\beta$' + '=' + str(p1[0][2]))    
        else:
            leg.append(rename_variable(paras['paras_to_loop'][0]) + '=' + str(p1))
        i+=1
    #plt.xlim([0.,100.])
    if loc!=0:
        plt.legend(leg, loc=loc)    
    plt.savefig(rep + '/' +name + ".png")   
    
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
    
def AC_met_vs_par(list_results_r, paras, met='satisfaction', name='', rep='.',AC=[],  labelx='', labely='', loc=3):
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
    plt.figure()
    plt.xlabel(labelx)
    plt.ylabel(labely)
    leg=[]
    i=0
    coin=sorted(list_results_r.items(), key=lambda a: a[0])
    for parconf,r1 in coin:
        x=sorted(r1.keys())
        for v in x:
            print 'POUEEEEET', r1[v][met]
            print r1[v][met][AC]
        y=[r1[v][met][AC]['avg'] for v in x]
        ey=[r1[v][met][AC]['std'] for v in x]
        plt.errorbar(x,y,ey,fmt='o--',color=[i/float(len(list_results_r.keys())),0.,1.-i/float(len(list_results_r.keys()))])
        #leg.append(u'\u03b1' + '=' + str(parconf[0][0]) + '; ' + u'\u03b2' + '=' + str(parconf[0][2]))    
        leg.append(rename_variable(paras['paras_to_loop'][0]) + '=' + str(parconf)) 
        i+=1
        
    if loc!=0:
        plt.legend(leg, loc=loc)    
    plt.savefig(rep + '/' + name + ".png")   
   
def difference_AC_met_vs_par(list_results_r, paras, met='satisfaction', name='', rep='.',AC=[],  labelx='', labely='', loc=3, norm=False):
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
        if norm:
            y=[r1[v][met][AC[0]]['avg']/r1[1.][met][AC[0]]['avg'] - r1[v][met][AC[1]]['avg']/r1[0.][met][AC[1]]['avg'] for v in x]
            ey=[(r1[v][met][AC[0]]['std']/r1[1.][met][AC[0]]['avg'] + r1[v][met][AC[1]]['std']/r1[0.][met][AC[1]]['avg']) for v in x]
        else:
            y=[r1[v][met][AC[0]]['avg'] - r1[v][met][AC[1]]['avg'] for v in x]
            ey=[(r1[v][met][AC[0]]['std'] + r1[v][met][AC[1]]['std']) for v in x]
        plt.errorbar(x,y,ey,fmt='o--',color=[i/float(len(list_results_r.keys())),0.,1.-i/float(len(list_results_r.keys()))])
        #plt.plot(x,y,'--o',color=[i/float(len(list_results_r.keys())),0.,1.-i/float(len(list_results_r.keys()))])
        #leg.append(u'\u03b1' + '=' + str(parconf[0][0]) + '; ' + u'\u03b2' + '=' + str(parconf[0][2]))    
        leg.append(rename_variable(paras['paras_to_loop'][0]) + '=' + str(parconf)) 
        i+=1
        
    plt.plot([x[0],x[-1]],[0.,0.],'k--')
        
    if loc!=0:
        plt.legend(leg, loc=loc)    
    plt.savefig(rep + '/' + name + ".png")   
    
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
    
def normalized_met_vs_par(results, paras, met='satisfaction', name='', rep='.', labelx='', labely='', loc=3):
    """
    Same as previous one, for normalized satisfaction.
    """
    try:
        assert paras['paras_to_loop'][-1]=='nA'
    except:
        print 'The inner loop should be over nA! Try to swap the order in paras_to_loop.'
        raise
    
    if name=='':
         name=met + '_vs_flights'
             
    if labelx=='':
        labelx=rename_variable('nA')
    if labely=='':
        labely='Normalized ' + met
        
    plt.figure()
    plt.xlabel(labelx)
    plt.ylabel(labely)
    leg=[]
    i=0
    
    ACs=results.values()[0].values()[0].values()[0].keys()
    coin=sorted(results.keys())
    for p1 in coin:
        r1=results[p1]
        x=sorted(r1.keys())
        y=[((r1[nA][met][ACs[1]]['avg']/r1[1.0][met][ACs[1]]['avg'])*nA + \
            (r1[nA][met][ACs[0]]['avg']/r1[0.][met][ACs[0]]['avg'])*(1.-nA)) for nA in x] #loop sur les nA
        
        ey=[(r1[nA][met][ACs[1]]['std']/r1[nA][met][ACs[1]]['avg']+r1[1.0][met][ACs[1]]['std']/r1[1.0][met][ACs[1]]['avg'])*(r1[nA][met][ACs[1]]['avg']/r1[1.0][met][ACs[1]]['avg'])*nA\
        +(r1[nA][met][ACs[0]]['std']/r1[nA][met][ACs[0]]['avg']+r1[0.][met][ACs[0]]['std']/r1[0.][met][ACs[0]]['avg'])*(r1[nA][met][ACs[0]]['avg']/r1[0.][met][ACs[0]]['avg'])*(1.-nA)\
        for nA in x]
        
        plt.plot(x,y,'o--',color=[i/float(len(results.keys())),0.,1.-i/float(len(results.keys()))])
        leg.append(rename_variable(paras['paras_to_loop'][0]) + '=' + str(p1)) 
        i+=1
        
    if loc!=0:
        plt.legend(leg,loc=loc)     
    plt.savefig(rep + '/' + name + ".png",dpi=100)   
    
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
    
# def per_AC_vs_main_loop(results, paras, met='satisfaction', name='', rep='.', AC=[], pop='A'):
#     if name=='':
#         name=met + '_vs_' + paras['paras_to_loop'][0]
#     plt.figure()
#     plt.xlabel(paras['paras_to_loop'][0])
#     plt.ylabel(met + ' for AC type ' + pop)
    
#     x=sorted(results.keys())
#     y=[results[xx][met][AC]['avg'] for xx in x]
#     ey=[results[xx][met][AC]['std'] for xx in x]
#     plt.errorbar(x,y,ey,fmt='o--',color='r')
    
#     plt.savefig(rep + '/' + name + ".png",dpi=100)
    
def AC_met_vs_main_loop(results, paras, met='satisfaction', name='', rep='.',AC=[],  labelx='', labely='', err = 'big'):
    """
    To have the metric 'met' per AC vs the par on which you loop.
    """
    
    print 'Doing plots', met, 'for company', AC
    
    if AC==(1.,0.,1000.):
        pop=r'AO$_R$'
    elif AC==(1.,0.,0.001):
        pop=r'AO$_S$'
    else:
        print "I did not recognize the company, I put pop='A'"
        pop=r'AO_$A$'

    if name=='':
         name=met + '_vs_' + paras['paras_to_loop'][0] + '_' + pop
    if labelx=='':
        labelx=rename_variable(paras['paras_to_loop'][0])
    if labely=='':
        labely=met + ' of ' + pop
    plt.figure()
    plt.xlabel(labelx)
    plt.ylabel(labely)
    #leg=[]
    #i=0
    #coin=sorted(list_results_r.items(), key=lambda a: a[0])
    #for parconf,r1 in coin:
    x=sorted(results.keys())
    try:
        x.remove(1.0)
    except ValueError:
        pass
    except:
        raise
    try:
        x.remove(0.0)
    except ValueError:
        pass
    except:
        raise    
    y=[results[v][met][AC]['avg'] for v in x]
    ey=[results[v][met][AC]['std'] for v in x]
    if err=='small':
        ey=np.array(ey)/sqrt(float(paras['n_iter']))
    plt.errorbar(x, y, ey, fmt='o--', color='r')#[i/float(len(list_results_r.keys())),0.,1.-i/float(len(list_results_r.keys()))])
    #leg.append(u'\u03b1' + '=' + str(parconf[0][0]) + '; ' + u'\u03b2' + '=' + str(parconf[0][2]))    
    #leg.append(rename_variable(paras['paras_to_loop'][0]) + '=' + str(parconf)) 
    #i+=1
        
    #if loc!=0:
    #    plt.legend(leg, loc=loc)    
    plt.savefig(rep + '/' + name + ".png") 
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
        
        rep, name= build_path_average(parass, vers=vers)
        with open(rep + name, 'r') as f:
            results=pickle.load(f)
        # print 'results'
        # print results
        # print
        #results_global={met:{k:sum([results[met][p][k]*paras['AC_dict'][p]/float(paras['ACtot']) \
        #        for p in results[met].keys()]) for k in ['avg', 'std']} for met in ['satisfaction', 'regulated_FPs', 'regulated_F']}
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
    
def overlap_network(G,typ='site',n_shortest_paths=1):
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
    results_list=[]
    vers='2.9' 
    if 0:
        a=paras['paras_to_loop'][0]
        paras['paras_to_loop'][0]=paras['paras_to_loop'][1]
        paras['paras_to_loop'][1]=a
    print 'Loading data...'  
    print paras['paras_to_loop']
    print 'paras.levels'
    print paras.levels
    results, results_global={}, {}
    results, results_global=loop({p:paras[p + '_iter'] for p in paras['paras_to_loop']}, paras['paras_to_loop'], paras, results, results_global, vers=vers)
    
    rep=build_path(paras,vers=vers)
    os.system('mkdir -p ' + rep)
    
    print'Done.'
    if len(paras['paras_to_loop'])==2:
        # Double loop
        global_met_vs_par(results_global, paras, rep=rep, loc=loc['ll'], norm_initial_values=False)
        #global_met_vs_par(results_global,paras, rep=rep, loc=loc['lr'], met='regulated_F')
        #global_met_vs_par(results_global,paras, rep=rep, met='regulated_FPs',loc=4)
        #normalized_met_vs_par(results,paras, rep=rep, loc=0)#loc['ll'])
        #global_met1_VS_met2(results_global, paras, rep=rep)
        if 'nA' in paras['paras_to_loop']:
            AC_met_vs_par(results, paras, AC=paras['par'][0], rep=rep, loc=0)
            AC_met_vs_par(results, paras, AC=paras['par'][1], rep=rep, loc=0)#loc['ll'])
        #just_legend(results, paras, rep=rep)
        #difference_AC_met_vs_par(results, paras, rep=rep, loc=0,norm=True)
    else:
        #Single loop   
        global_vs_main_loop(results_global, paras, met='satisfaction', rep=rep, err='small')
        global_vs_main_loop(results_global, paras, met='regulated_F', rep=rep, err='small')
        global_vs_main_loop(results_global, paras, met='regulated_FPs', rep=rep, err='small')
        if 'nA' in paras['paras_to_loop']:
            AC_met_vs_main_loop(results, paras, AC=paras['par'][0], rep=rep, err='small')
            AC_met_vs_main_loop(results, paras, AC=paras['par'][1], rep=rep, err='small')
            AC_met_vs_main_loop(results, paras, AC=paras['par'][0], rep=rep, met = 'regulated_F', err='small')
            AC_met_vs_main_loop(results, paras, AC=paras['par'][1], rep=rep, met = 'regulated_F', err='small')
            AC_met_vs_main_loop(results, paras, AC=paras['par'][0], rep=rep, met = 'regulated_FPs', err='small')
            AC_met_vs_main_loop(results, paras, AC=paras['par'][1], rep=rep, met = 'regulated_FPs', err='small')
        #per_AC_vs_main_loop(results, paras, AC=paras['par'][0])
    
    plt.show()
    
    
    print
    print
    print 'Done. Plots saved in ', rep
    
