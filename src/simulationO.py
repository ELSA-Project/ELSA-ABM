#!/usr/bin/env python

from simAirSpaceO import AirCompany, Network_Manager
import networkx as nx
import ABMvars
#from random import getstate, setstate, 
from random import shuffle, uniform,  sample, seed, choice, gauss, randrange
import pickle
from string import split
import matplotlib.pyplot as plt
import os
import numpy as np
import copy
from utilities import draw_network_map
from math import ceil
from general_tools import draw_network_and_patches, header, delay, clock_time, delay, date_human
from tools_airports import extract_flows_from_data
#from utilitiesO import compare_networks

version='2.9.4'
main_version=split(version,'.')[0] + '.' + split(version,'.')[1]

if 0:
    #see = 7122008
    see = randrange(1,10000000)
    print 'Caution! Seed:', see
    seed(see)

def build_path(paras, vers=main_version, in_title=['Nfp', 'tau', 'par', 'ACtot', 'nA', 'departure_times','Nsp_nav', 'old_style_allocation', 'noise'], only_name = False):
    """
    Used to build a path from a set of paras. 
    Changed 2.2: is only for single simulations.
    Changed 2.4: takes different departure times patterns. Takes names.
    Changed in 2.5: added N_shocks and improved the rest.
    """
    
    name='Sim_v' + vers + '_' + paras['G'].name
    if not only_name:
        name = '../results/' + name
    
    in_title=list(np.unique(in_title))
        
    if paras['departure_times']!='zeros':
        try:
            in_title.remove('ACtot')
        except ValueError:
            pass
        in_title.insert(1,'density')
        in_title.insert(2,'day')
        if paras['departure_times']=='square_waves':
            in_title.insert(1,'Delta_t')
            
    if paras['N_shocks']==0 and paras['mode_M1']=='standard':
        try:
            in_title.remove('N_shocks')
        except ValueError:
            pass      
    elif paras['N_shocks']!=0 and paras['mode_M1']=='standard':
        in_title.insert(-1,'N_shocks')
    elif paras['mode_M1']=='sweep':
        in_title.insert(-1,'STS')
    
    in_title=np.unique(in_title)
    
    for p in in_title:
        if p=='par':
            if len(paras[p])==1 or paras['nA']==1.:
                coin=str(float(paras[p][0][0])) + '_' + str(float(paras[p][0][1])) + '_' + str(float(paras[p][0][2]))
            elif len(paras[p])==2:
                coin=str(float(paras[p][0][0])) + '_' + str(float(paras[p][0][1])) + '_' + str(float(paras[p][0][2])) + '__' +str(float(paras[p][1][0])) + '_' + str(float(paras[p][1][1])) + '_' + str(float(paras[p][1][2])) 
            else:
                coin='_several'
        else:
            coin=str(paras[p])
        name+='_' + p + coin

    return name

def check_object(G):
    """
    Use to check if you have an old object. Used for legacy.
    """
    
    return hasattr(G, 'comments')
        

class Simulation:
    """
    Class Simulation. 
    =============
    Main object for the simulation. Initialize loads, prepares air companies, 
    calls network manager, keeps the M1 queue in memory, calls for possible 
    shocks.
    """
    def __init__(self, paras, G=None, verbose=False, make_dir=False):
        """
        Initialize the simulation, build the network if none is given in argument, set the verbosity
        Change in 2.9.3: self.make_times is computed at each simulations.
        Changed in 2.9.3: update shortest paths if there is a change of Nsp_nav.
        """
        self.paras = paras
        
        for k in ['AC', 'Nfp', 'na', 'tau', 'departure_times', 'ACtot', 'N_shocks','Np',\
            'ACsperwave','Delta_t', 'width_peak', 'old_style_allocation', 'flows', 'nA', \
            'day', 'noise', 'Nsp_nav', 'STS']:
            if k in paras.keys():
                setattr(self, k, paras[k])

        self.make_times()#, times_data=paras['times'])
        self.pars=paras['par']

        assert check_object(G)
        assert G.Nfp==paras['Nfp']
        
        self.G=G.copy()
        self.verb=verbose
        self.rep=build_path(paras)

        if self.Nsp_nav!= self.G.Nsp_nav:
            if verbose:
                print 'Updating shortest path due to a change of Nsp_nav...'
            self.G.choose_short(self.Nsp_nav)

        if make_dir:
            os.system('mkdir -p ' + self.rep)
        
    def make_simu(self, clean=False, storymode=False, flows= {}):
        """
        Do the simulation, clean afterwards the network (useful for iterations).
        Changed in 2.9.6: added the shuffle_departure.
        """
        if self.verb:
            print 'Doing simulation...'
        #self.make_times()#, times_data=paras['times'])
        Netman = Network_Manager(old_style = self.old_style_allocation)
        self.storymode = storymode

        #------------------------------------------------------# 

        Netman.initialize_load(self.G, length_day = int(self.day/60.))

        if self.flows == {}:
            self.build_ACs()
        else:
            self.build_ACs_from_flows()

        if clean:
            Netman.initialize_load(self.G, length_day = int(self.day/60.))

        self.queue = Netman.build_queue(self.ACs)

        self.shuffle_departure_times()

        Netman.allocate_queue(self.G, self.queue, storymode=storymode)

        self.mark_best_of_queue()

        self.M0_queue = copy.deepcopy(self.queue)
        #Netman.M0_to_M1(self.G, self.queue, self.N_shocks, self.tau, self.Nsp_nav, storymode=True)
        Netman.M0_to_M1_quick(self.G, self.queue, self.N_shocks, self.tau, self.Nsp_nav, storymode=True, sectors_to_shut = self.STS)
        
    def build_ACs(self):
        """
        Build all the Air Companies based on the number of Air Companies given in paras.
        If n_AC is an integer and if several sets of parameters are given for the utility 
        function, make the same number of AC sor each set. If n_AC is an array of integers 
        of the same size than the number of set of parameters, populates the different
        types with this array.
        Exs:
            self.n_AC=30, self.pars=[[1,0,0]] 
            gives 30 ACs with parameters [1,0,0].
            
            self.n_AC=30, self.pars=[[1,0,0], [0,0,1]] 
            gives 15 ACs with parameters [1,0,0] and 15 ACs with parameters [0,0,1].
            
            self.n_AC=[10,20], self.pars=[[1,0,0], [0,0,1]] 
            gives 10 ACs with parameters [1,0,0] and 20 ACs with parameters [0,0,1].
            
        Builds all flight plans for all ACs.

        Changed in 2.9: pairs given to ACs are navpoints, not sectors
        """
        
        if type(self.AC)==type(1):
            self.AC=[self.AC/len(self.pars) for i in range(len(self.pars))]
        
        try:
            assert len(self.AC)==len(self.pars)
        except:
            print 'n_AC should have the same length than the parameters, or be an integer'
            raise
    
        self.ACs={}
        k=0
        assert len(self.t0sp)==sum(self.AC)
        shuffle(self.t0sp)
        for i,par in enumerate(self.pars):
            for j in range(self.AC[i]):
                self.ACs[k]=AirCompany(k, self.Nfp, self.na, self.G.G_nav.pairs, par)
                self.ACs[k].fill_FPs(self.t0sp[k], self.tau, self.G)
                k+=1
        # if clean:  # Not sure if this is useful.
        #     self.Netman.initialize_load(self.G)

    def build_ACs_from_flows(self):
        """
        New in 2.9.2: the list of ACs is built from the flows (given by times). 
        (Only the number of flights can be matched, or also the times, which are taken as desired times.)
        """
        self.ACs={}
        k=0
        for ((source, destination), times) in self.flows.items():
            idx_s = self.G.G_nav.idx_navs[source]
            idx_d = self.G.G_nav.idx_navs[destination]
            if idx_s in self.G.G_nav.airports and idx_d in self.G.G_nav.airports and self.G.G_nav.short.has_key((idx_s, idx_d)):    
                n_flights_tot = len(times)
                n_flights_A = int(self.nA*n_flights_tot)
                n_flights_B = n_flights_tot - int(self.nA*n_flights_tot)
                AC= [n_flights_A, n_flights_B]
                l=0
                for i, par in enumerate(self.pars):
                    for j in range(AC[i]):
                        time = times[l]
                        self.ACs[k] = AirCompany(k, self.Nfp, self.na, self.G.G_nav.short.keys(), par)
                        time = int(delay(time, starting_date = [time[0], time[1], time[2], 0., 0., 0.])/60.)
                        self.ACs[k].fill_FPs([time], self.tau, self.G, pairs = [(idx_s, idx_d)])
                        k+=1
                        l+=1
            else:
                print "I do " + (not idx_s in self.G.G_nav.airports)*'not', "find", idx_s, ", I do " + (not idx_d in self.G.G_nav.airports)*'not', "find", idx_d,\
                 'and the couple is ' + (not self.G.G_nav.short.has_key((idx_s, idx_d)))*'not', 'in pairs.'
                print 'I skip this flight.'
        # if clean:   
        #     self.Netman.initialize_load(self.G)

    def compute_flags(self):
        """
        Computes flags, bottlenecks and overloadedFPs for each AC and in total.
        The queue is used to sort the flights, even within a given AC.
        """
        for ac in self.ACs.values():
            ac.flag_first,ac.bottlenecks, ac.overloadedFPs = [], [],[]
        self.bottlenecks, self.flag_first, self.overloadedFPs= [], [], []
        for f in self.queue:
            f.make_flags()
        
            self.ACs[f.ac_id].flag_first.append(f.flag_first)
            self.ACs[f.ac_id].bottlenecks.append(f.bottlenecks)
            self.ACs[f.ac_id].overloadedFPs.append(f.overloadedFPs)
            
            self.flag_first.append(f.flag_first)
            self.overloadedFPs.append(f.overloadedFPs)
            self.bottlenecks.append(f.bottlenecks)
        
        self.results={p:{ac.id:{'flags':ac.flag_first,'overloadedFPs':ac.overloadedFPs, 'bottlenecks':ac.bottlenecks}\
                    for ac in self.ACs.values() if ac.par==p} for p in self.pars}
        self.results['all']=self.flag_first, self.overloadedFPs, self.bottlenecks
        
        # if self.verb:
        #     print 'flags', self.flag_first
        #     #print 'overloadedFPs', self.overloadedFPs
        #     #print 'bottlenecks', self.bottlenecks
        #     print
        #     print
            
    def save(self, rep='', split=False, only_flags=False):
        """
        Save the network in a pickle file, based on the paras.
        Can be saved in a single file, or different files to speed up the post-treatment.
        """
        if rep=='':
            rep=build_path(self.paras)

        if only_flags:
            f=open(rep + '_flags.pic','w')
            pickle.dump((self.flag_first, self.bottlenecks, self.overloadedFPs),f)
            f.close()
        else:
            if not split:
                print 'Saving whole object in ', rep
                with open(rep + '/sim.pic','w') as f:
                    pickle.dump(self,f)
            else:
                print 'Saving split object in ', rep
                with open(rep + '_ACs.pic','w') as f:
                    pickle.dump(self.ACs,f)
                
                f=open(rep + '_G.pic','w')
                pickle.dump(self.G,f)
                f.close()
                
                f=open(rep + '_flags.pic','w')
                pickle.dump((self.flag_first, self.bottlenecks, self.overloadedFPs),f)
                f.close()
            
    def load(self, rep=''):
        """
        Load a split Simulation from disk.
        """            
        if rep=='':
            rep=build_path(self.paras)
        print 'Loading split simu from ', rep
        f=open(rep + '_ACs.pic','r')
        self.ACs=pickle.load(f)
        f.close()
        
        f=open(rep + '_G.pic','r')
        self.G=pickle.load(f)
        f.close()
        
        f=open(rep + '_flags.pic','r')
        (self.flag_first, self.bottlenecks, self.overloadedFPs)=pickle.load(f)
        f.close()
        
    def make_times(self, times_data=[]):
        """
        Prepare t0sp, the matrix of desired times.
        New in 2.9: added departures from data.
        New in 2.9.3: added noise.
        """
        if self.departure_times=='zeros':
            self.t0sp=[[0 for j in range(self.na)] for i in range(self.ACtot)]     
        elif self.departure_times=='uniform':
            self.t0sp=[[uniform(0, self.day) for j in range(self.na)] for i in range(self.ACtot)]
        elif self.departure_times=='square_waves':
            self.t0sp=[]
            if self.na==1:
                for i in range(self.Np):
                    for j in range(self.ACsperwave):
                        self.t0sp.append([uniform(i*(self.width_peak+self.Delta_t),i*(self.width_peak+self.Delta_t)+self.width_peak)])
            else:
                print 'Not implemented yet...'
                raise
        # elif self.departure_times=='from_data':
        #     self.t0sp=[]
        #     if self.na==1:
        #         for i in range(self.ACtot):
        #             self.t0sp.append([choice(times_data)])
        elif self.departure_times=='exterior':
            self.t0sp=[]

    def shuffle_departure_times(self):
        if self.noise!=0:
            for f in self.queue:
                f.shift_desired_time(gauss(0., self.noise))

    def mark_best_of_queue(self):
        for f in self.queue:
            f.best_fp_cost=f.FPs[0].cost
            
                          
def post_process_queue(queue):
    """
    Used to post-process results. Every processes between the simulation and 
    the plots should be here.
    Changed in 2.4: add satisfaction, regulated flight & regulated flight plans. On level of iteration added (on par).
    Changed in 2.5: independent function.
    Changed in 2.7: best cost is not the first FP's one.
    Changed in 2.9.3: added regulated_1FP
    """

    for f in queue:   
        # Make flags
        f.make_flags() # Maybe useless
        
        bestcost = f.best_fp_cost
        acceptedFPscost = [FP.cost for FP in f.FPs if FP.accepted]
                
        if len(acceptedFPscost) != 0:
            f.satisfaction = bestcost/min(acceptedFPscost)
        else:
            f.satisfaction = 0                    
            
        # Regulated flight plans
        f.regulated_FPs=len([FP for FP in f.FPs if not FP.accepted])

        # At least one flight plan regulated
        f.regulated_1FP = float(len([FP for FP in f.FPs if not FP.accepted])!=0)

        # Regulated flights
        if len([FP for FP in f.FPs if FP.accepted])==0:
            f.regulated_F = 1.
        else:
            f.regulated_F = 0.

    return queue
    
def extract_aggregate_values_on_queue(queue, types_air_companies, mets=['satisfaction', 'regulated_F', 'regulated_FPs']):
    results={}
    for m in mets:
        results[m]={}
        for tac in types_air_companies:
            pouet=[getattr(f,m) for f in queue if tuple(f.par)==tuple(tac)]
            if pouet!=[]:
                results[m][tuple(tac)]=np.mean(pouet)
            else:
                results[m][tuple(tac)]=0.
        
    return results                
            
def extract_aggregate_values_on_network(G):
    coin1=[]
    coin2=[]
    for n in G.nodes():
        if len(G.node[n]['load'])>2:
            avg=np.average(G.node[n]['load'])
            coin1.append(avg)
            coin2.append(avg/float(G.node[n]['capacity']))
        else:
            coin1.append(0.)
            coin2.append(0.)

    return {'loads': np.mean(coin1), 'loads_norm':np.mean(coin2)}
                   
def plot_times_departure(queue, rep='.'):
    t_pref=[f.FPs[0].t for f in queue]
    t_real=[f.fp_selected.t for f in queue if f.fp_selected!=None]
    
    plt.figure(1)
    bins=range(int(ceil(max(t_real + t_pref))) + 10)
    plt.hist(t_pref,label='pref',facecolor='green', alpha=0.75, bins=bins)
    plt.hist(t_real,label='real',facecolor='blue', alpha=0.25, bins=bins)
    plt.legend()
    plt.savefig(rep + '/departure_times.png')
    plt.show()
        
def compute_M1_trajectories(queue):
    """
    Returns some trajectories (navpoint names) based on the given queue.
    """
    #trajectories=[]
    trajectories_nav=[]
    for f in queue:
        try:
            #trajectories.append(f.FPs[[fpp.accepted for fpp in f.FPs].index(True)].p) 
            trajectories_nav.append(f.FPs[[fpp.accepted for fpp in f.FPs].index(True)].p_nav) 
        except ValueError:
            pass

    return trajectories_nav

def generate_traffic(paras, G, dump = None, simple_setup=True):
    """
    High level function to create traffic on a given network with given parameters. 
    It is not really intented to use as a simulation by itself, but only to generate 
    some synthetic traffic, mainly for the tactical ABM.
    Returns a set of M1 trajectories.
    If simple_setup is True, the function uses some default parameters suitable for 
    quick generation of traffic.
    New in 2.9.4.
    """
    #GG = ABMvars.G
    #paras = ABMvars.paras.copy()

    print header(paras,'Generation of traffic', version, paras_to_display=['ACtot'])

    with clock_time():
        sim=Simulation(paras, G=G, make_dir=True, verbose=True)
        sim.make_simu(storymode=False)
        sim.compute_flags()
        queue=post_process_queue(sim.queue)
        M0_queue=post_process_queue(sim.M0_queue)

    #print 'ACs:'
    #print sim.ACs
    print
    print
    print 'Global metrics for M1:'
    print extract_aggregate_values_on_queue(queue, paras['par'])
    print 'Global metrics for M0:'
    print extract_aggregate_values_on_queue(M0_queue, paras['par'])
    print
    print 'Number of rejected flights:', len([f for f in sim.queue if not f.accepted])
    print 'Number of rejected flight plans:', len([fp for f in sim.queue for fp in f.FPs if not fp.accepted]), '/', len(sim.queue)*sim.Nfp
    #print "Satisfaction: "
    #print [f.satisfaction for f in sim.queue]
    print

    trajectories = compute_M1_trajectories(queue)

    return trajectories

def write_trajectories_file(G, trajectories, fil='../trajectories/trajectories.dat', starting_date = [2010, 6, 5, 10, 0, 0]):
    with open(fil, 'w') as f:
        for i,trajectory in enumerate(trajectories):
            x, y, ts = [], [], []
            t = 0.
            for j,n in enumerate(trajectory):
                x.append(G.node[n]['coord'][0])
                y.append(G.node[n]['coord'][1])
                t = 0 if j==0 else t + G[n][trajectory[j-1]]['weight']
                ts.append(date_abm_tactic(date_st(t, starting_date = starting_date)))
            print >>f, i, "\t", len(t), "\t", [(x[j], ",", y[j], ",", 0., ",", ts[j], "\t") for j in range(len(trajectory))]

    print "Trajectories saved in", fil

def date_abm_tactic(date):
    """
    Transform a list [year, month, day, hours, minutes, seconds] in 
    YYYY-MM-DD H:mm:s:0
    """
    year, month, day, hours, minutes, seconds = tuple(date)
    month = str(month) if month<10 else "0" + str(month)
    day = str(day) if day<10 else "0" + str(day)

    date_abm = date_human([str(year), month, day, str(hours), str(minutes), str(seconds)]) + ':0'
    return date_abm

if __name__=='__main__': 
    """
    Manual single simulation used for "story mode" and debugging.
    """
    GG = ABMvars.G
    paras = ABMvars.paras.copy()

    print header (paras,'SimulationO', version, paras_to_display=['ACtot'])

    with clock_time():
        sim=Simulation(paras, G=GG, make_dir=True, verbose=True)
    
        sim.make_simu(storymode=False)
    # with open(build_path(sim.paras) + '_sim.pic', 'w') as f:
    #     pickle.dump(sim, f)
        sim.compute_flags()
        queue=post_process_queue(sim.queue)
        M0_queue=post_process_queue(sim.M0_queue)

    #print 'ACs:'
    #print sim.ACs
    print
    print
    print 'Global metrics for M1:'
    print extract_aggregate_values_on_queue(queue, paras['par'])
    print 'Global metrics for M0:'
    print extract_aggregate_values_on_queue(M0_queue, paras['par'])
    print
    print 'Number of rejected flights:', len([f for f in sim.queue if not f.accepted])
    print 'Number of rejected flight plans:', len([fp for f in sim.queue for fp in f.FPs if not fp.accepted]), '/', len(sim.queue)*sim.Nfp
    #print "Satisfaction: "
    #print [f.satisfaction for f in sim.queue]
    print
    
    #coin=[797, 430, 911, 792, 343, 1161, 1162, 840, 522, 1031, 1013, 596, 452, 376, 441, 736, 561, 244, 459, 69, 209, 98, 635, 926, 315, 1124, 1109, 641, 250]
    # for f in sim.queue[:10]:
    #     for fp in f.FPs:
    #         #if np.array(fp.p_nav)=coin and np.array(fp.p_nav)!=reversed(coin):
    #             #print np.array(fp.p_nav)==coin 
    #             print fp.p_nav
    #             print fp.p

    #             print fp.accepted
    #             #print filter(lambda x: not x in fp.p_nav, coin)
                
    #     print
    #     print
    
    for n in sim.G.nodes():
        print n, sim.G.node[n]['capacity'], sim.G.node[n]['load']
        if max(sim.G.node[n]['load']) == sim.G.node[n]['capacity']:
            print "Capacity's reached for sector:", n
        if max(sim.G.node[n]['load']) > sim.G.node[n]['capacity']:
            print "Capacity overreached for sector:", n, '!'
    #draw_network_map(sim.G.G_nav, title=sim.G.G_nav.name, load=False, generated=True,\
    #        airports=True, stack=True, nav=True, queue=sim.queue)
    
    trajectories=[]
    trajectories_nav=[]
    for f in sim.queue:
        try:
            trajectories.append(f.FPs[[fpp.accepted for fpp in f.FPs].index(True)].p) 
            trajectories_nav.append(f.FPs[[fpp.accepted for fpp in f.FPs].index(True)].p_nav) 
        except ValueError:
            pass

    #  Real trajectories
    trajectories_real = []
    print len(sim.G.flights_selected)
    for f in sim.G.flights_selected:
        navpoints = set([sim.G.G_nav.idx_navs[p[0]] for p in f['route_m1']])
        if navpoints.issubset(set(sim.G.G_nav.nodes())) and (sim.G.G_nav.idx_navs[f['route_m1'][0][0]], sim.G.G_nav.idx_navs[f['route_m1'][-1][0]]) in sim.G.G_nav.short.keys():
            trajectories_real.append([sim.G.G_nav.idx_navs[p[0]] for p in f['route_m1']])
    # real_flights = extract_flows_from_data(sim.G.paras_real, [sim.G.G_nav.node[n]['name'] for n in sim.G.G_nav.nodes()])[2]

    print len(trajectories_real)

    # trajectories_real = []
    # for f in real_flights:
    #     navpoints = set([sim.G.G_nav.idx_navs[p[0]] for p in f['route_m1']])
    #     #edges = set([(sim.G.G_nav.idx_navs[f['route_m1'][i][0]], sim.G.G_nav.idx_navs[f['route_m1'][i+1][0]]) for i in range(len(f['route_m1'])-1)])
        
    # # print len(trajectories_real)
    
    draw_network_and_patches(sim.G, sim.G.G_nav, sim.G.polygons, name='trajectories_nav', flip_axes=True, trajectories=trajectories_nav, trajectories_type='navpoints', rep = sim.rep)

    #draw_network_and_patches(sim.G, sim.G.G_nav, sim.G.polygons, name='trajectories_real', flip_axes=True, trajectories=trajectories_real, trajectories_type='navpoints', save = True, rep = build_path(sim.paras), dpi = 500)



    if 0:
        sim.save(rep = build_path(sim.paras))

    if 0:
        #draw_network_and_patches(sim.G,sim.G.G_nav,sim.G.polygons, name='network_small_beta', flip_axes=True, trajectories=trajectories_nav, trajectories_type='navpoints')
        p0=20
        p1=65
        #all_possible_trajectories=sorted([path for paths in sim.G.short_nav[(p0,p1)].values() for path in paths], key= lambda p: sim.G.G_nav.weight_path(p))
        possible_trajectories=[p for p in sim.G.G_nav.short[(p0,p1)]]
        if 0:
            print 'All possible trajectories:'
            for path in all_possible_trajectories:
                print path
            print
            print 'Possible trajectories:'
            for path in possible_trajectories:
                print path
        
        #draw_network_and_patches(sim.G,sim.G.G_nav,sim.G.polygons, name='all_possible_trajectories', flip_axes=True, trajectories=all_possible_trajectories, trajectories_type='navpoints')
        draw_network_and_patches(sim.G,sim.G.G_nav,sim.G.polygons, name='possible_trajectories', flip_axes=False, trajectories=possible_trajectories, trajectories_type='navpoints', rep = sim.rep)
        #draw_network_and_patches(sim.G, None, sim.G.polygons, name='numbers', , flip_axes=True, numbers=True)

        #draw_network_map(sim.G, title=sim.G.name, load=False, generated=True,\
        #         airports=True, trajectories=trajectories, add_to_title='_high_beta')
    # if 1
    #     draw_network_map(sim.G, title=sim.G.name, load=False, generated=True,\
    #             airports=True, trajectories=trajectories, add_to_title='_high_beta')#stack=True
    #     draw_network_map(sim.G.G_nav, title=sim.G.name, load=False, generated=True,\
    #             airports=True, trajectories=trajectories_nav, add_to_title='_nav_high_beta')
    # if 0:
    #     print sim.queue[0].FPs[0].p_nav
    #     for p in sim.queue[0].FPs[0].p_nav:
    #         print sim.G.G_nav.node[p]['sec']
    #     print
    #     print sim.queue[0].FPs[1].p_nav
    #     for p in sim.queue[0].FPs[2].p_nav:
    #         print sim.G.G_nav.node[p]['sec']
    #     sim.queue[0].FPs[1].p_nav
    #     draw_network_map(sim.G.G_nav, title=sim.G.name, load=False, generated=True,\
    #             airports=True, trajectories=[sim.queue[0].FPs[0].p_nav], add_to_title='_nav_high_beta', polygons=sim.G.polygons.values())
    #     draw_network_map(sim.G.G_nav, title=sim.G.name, load=False, generated=True,\
    #             airports=True, trajectories=[sim.queue[0].FPs[2].p_nav], add_to_title='_nav_high_beta', polygons=sim.G.polygons.values())
    #     plt.show()
        
    #sim.save()
    
   # print
   # print
   # for f in queue:
   #     print f
   #     fp=f.FPs[0]
   #     i=0
   #     while not fp.accepted and i<len(f.FPs):
   #         fp=f.FPs[i]
   #         i+=1
       
   #     if i<len(f.FPs):
   #         print fp.p
   #plot_times_departure(sim.queue, rep=sim.rep)
   #  draw_network_map(sim.G, title=sim.G.name, queue=sim.queue, generated=True, rep=sim.rep)
    
   # print sim.M0_queue
   
   # print sim.queue

    print 'Done.'
    

        
