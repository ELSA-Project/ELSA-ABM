#! /usr/bin/env python
# -*- coding: utf-8 -*-

import networkx as nx
import sys
from paths import path_ksp
sys.path.insert(1,path_ksp)
from graph import DiGraph
from algorithms import ksp_yen, ksp_yen_old
from random import sample, uniform, gauss, shuffle, choice
import numpy as np
from numpy.random import lognormal
import matplotlib.delaunay as triang
from triangular_lattice import build_triangular
from math import sqrt, log
import pickle
import os
import matplotlib.pyplot as plt
from utilities import draw_network_map
from general_tools import counter, silence
from string import split
import copy

version='2.9.8'

class Network_Manager:
    """
    Class Network_Manager. 
    =============
    The network manager receives flihgt plans from air companies anfd tries to
    fill the best ones on the network, by increasing order of cost. If the 
    flight plan does not create overeach the capacity of any sector, it is 
    allocated to the network and the sector loads are updated. 
    The network manager can also inform the air companies if a shock occurs on
    the network, i.e if some sectors are shut. It asks for a new bunch of 
    flights plans from the airlines impacted.
    New in 2.9.2: gather methods coming from class Net (and Simulation) to make a proper agent.
    """
    def __init__(self, old_style=False):
        """
        'Old style' means that we detect the peaks in the loads (we have the full 
        profile of the load with time, and we detect if the maximum is smaller than the capacity)
        The 'new style computes' just the number of flights which cross the sector during a hour, 
        and then detect if this number is smaller than the capacity.
        """
        self.old_style = old_style 
        if not old_style:
            self.overload_sector = self.overload_sector_hours
            self.allocate = self.allocate_hours
            self.deallocate = self.deallocate_hours
            self.overload_airport = self.overload_airport_hours
        else:
            self.overload_sector = self.overload_sector_peaks
            self.allocate = self.allocate_peaks
            self.deallocate = self.deallocate_peaks
            self.overload_airport = self.overload_airport_peaks

    def initialize_load(self, G, length_day=24):
        """
        Initialize loads, with length given by t_max.
        Changed in 2.2: keeps in memory only the intervals.
        Changed in 2.5: t_max is set to 10**6.
        Changed in 2.7: load of airports added.
        Changed in 2.9: no more load of airports. Load is an array giving the load for each hour.
        Changed in 2.9.3: airports again :).
        """
        if not self.old_style:
            for n in G.nodes():
                G.node[n]['load']=[0 for i in range(length_day)]
            for a in G.airports:
                G.node[a]['load_airport']=[0 for i in range(length_day)]
        else:
            for n in G.nodes():
                G.node[n]['load_old']=[[0,0],[10**6,0]] 
            for a in G.airports:
                G.node[a]['load_old_airport']=[[0,0],[10**6,0]] 

    def build_queue(self, ACs):
        """
        Add all the flights of all ACs to a queue, in random order.
        @return queue
        """
        queue=[]
        for ac in ACs.values():
            for f in ac.flights:
                queue.append(f)
        shuffle(queue)

        return queue

    def allocate_queue(self, G, queue, storymode=False):
        """
        For each flight of the queue, tries to allocate it to the airspace.
        """

        for i,f in enumerate(queue):
            if storymode:
                print "Flight with position", i, "from", f.source, "to", f.destination, "of company", f.ac_id
                print "with parameters", f.par
                print "tries to be allocated."
            f.pos_queue=i
            self.allocate_flight(G, f, storymode=storymode)
            if storymode:
                print "flight accepted:", f.fp_selected!=None
                if f.fp_selected==None:
                    print 'because '
                print
                print 
                #print 'New load of source airport (', G.G_nav.node[f.source]['sec'], '):', G.node[G.G_nav.node[f.source]['sec']]['load_old']

    def allocate_flight(self, G, flight, storymode=False):
        """
        Tries to allocate the flights by sequentially checking if each flight plan does not overload any sector,
        beginning with the best ones. The rejection of the flights is kept in memory, as
        well as the first sector overloaded (bottleneck), and the flight plan selected,.
        Changed in 2.2: using intervals.
        Changed in 2.7: sectors of airports are checked independently.
        Changed in 2.9: airports are not checked anymore.
        Changed in 2.9.3: airports are checked again :).
        """

        i=0
        found=False
        #print 'flight id', flight.ac_id
        while i<len(flight.FPs) and not found:
            # print 'fp id', i
            fp=flight.FPs[i]
            #print 'fp.p', fp.p
            self.compute_flight_times(G, fp)
            path, times=fp.p, fp.times

            if storymode:
                print "     FP no", i, "tries to be allocated with trajectory (sectors):"
                print fp.p
                print "and crossing times:"
                print fp.times

            first=1 ###### ATTENTION !!!!!!!!!!!
            last=len(path)-1 ########## ATTENTION !!!!!!!!!!!
            
            j=first
            while j<last and not self.overload_sector(G, path[j],(times[j],times[j+1])):#and self.node[path[j]]['load'][j+time] + 1 <= self.node[path[j]]['capacity']:
                j+=1 

            fp.accepted = not ((j<last) or self.overload_airport(G, path[0],(times[0],times[1])) or self.overload_airport(G, path[-1],(times[-2],times[-1])))
                  
            path_overload = j<last
            source_overload = self.overload_airport(G, path[0],(times[0],times[1]))
            desetination_overload = self.overload_airport(G, path[-1],(times[-2],times[-1]))

            if storymode:
                print "     FP has been accepted:", fp.accepted
                if not fp.accepted:
                    if path_overload: 
                        print "     because sector", path[j], "was full."
                    if source_overload:
                        print "     because source airport was full."
                        print G.node[path[0]]['load_airport']
                        print G.node[path[0]]['capacity_airport']
                    if desetination_overload:
                        print "     because destination airport was full."

            if fp.accepted:
                self.allocate(G, fp, storymode=storymode)
                flight.fp_selected=fp
                flight.accepted = True
                found=True
            else:
                if j<last:
                    fp.bottleneck=path[j]
            i+=1 

        if not found:
            flight.fp_selected=None
            flight.accepted = False
        
    def compute_flight_times(self, G, fp):
        """
        Compute the entry times and exit times of each sector the trajectory of the given flight plan.
        Changed in 2.8: based on navpoints.

        @input G: network.
        @input fp: flight plan.
        """
        entry_times=[0.]*(len(fp.p)+1)
        entry_times[0]=fp.t
        road=fp.t
        sec=G.G_nav.node[fp.p_nav[0]]['sec']
        j=0
        for i in range(1,len(fp.p_nav)):
            w=G.G_nav[fp.p_nav[i-1]][fp.p_nav[i]]['weight']
            if G.G_nav.node[fp.p_nav[i]]['sec']!=sec:
                j+=1
                entry_times[j]=road + w/2.
                sec=G.G_nav.node[fp.p_nav[i]]['sec']
            road+=w
        entry_times[len(fp.p)]=road
        fp.times=entry_times
        
    def overload_sector_hours(self, G, n, (t1, t2)):
        """
        Check if the sector n would be overloaded if an additional flight were to 
        cross it between times t1 and t2.
        Changed in 2.9: it does not check anymore if the maximum number of flights have 
        overreached the capacity, but if the total number of flights during an hour 
        (counting those already there at the beginning and those still there at the end)
        is greater than the capacity. 

        There is now an absolute reference in time and the weights of the network are 
        in minutes.

        @input G: network on which to allocate the flight.
        @input n: sector to check
        @input (t1, t2): time of entry and exit of the flight in the sector n.
        @return overload: True if the sector would be overloaded with the allocation of this flight plan.
        """

        overload = False
        h = 0 
        # h=max(0, floor(t1/60.)) # a tester pour virer la premiere condition.
        while float(h) <= t2/60. and not overload:
            try:
                if h+1 > t1/60. and G.node[n]['load'][h]+1>G.node[n]['capacity']:
                    overload = True
            except:
                print t1/60., t2/60., h
                print G.node[n]['load']
                raise
            h += 1
        return overload

    def overload_sector_peaks(self, G, n, (t1, t2)):
        """
        Old version (2.8.2) of previous method. Based on maximum number of 
        planes in a given sector at any time.

        Note: was previously called overload_capacity
        """
        ints=np.array([p[0] for p in G.node[n]['load_old']])
        
        caps=np.array([p[1] for p in G.node[n]['load_old']])
        i1=max(0,list(ints>=t1).index(True)-1)
        i2=list(ints>=t2).index(True)
        
        pouet=np.array([caps[i]+1 for i in range(i1,i2)])

        return len(pouet[pouet>G.node[n]['capacity']]) > 0

    def overload_airport_hours(self, G, n, (t1, t2)):
        """
        Same than overload_sector_hours, for airports.
        """
        overload = False
        h = 0 
        # h=max(0, floor(t1/60.)) # a tester pour virer la premiere condition.
        while float(h) <= t2/60. and not overload:
            if h+1 > t1/60. and G.node[n]['load_airport'][h]+1>G.node[n]['capacity_airport']:
                overload = True
            h += 1
        return overload

    def overload_airport_peaks(self, G, n, (t1, t2)):
        """
        Same then overload_sector_peaks, for airports.
        """
        ints=np.array([p[0] for p in G.node[n]['load_old_airport']])
        
        caps=np.array([p[1] for p in G.node[n]['load_old_airport']])
        i1=max(0,list(ints>=t1).index(True)-1)
        i2=list(ints>=t2).index(True)
        
        pouet=np.array([caps[i]+1 for i in range(i1,i2)])

        return len(pouet[pouet>G.node[n]['capacity_airport']]) > 0
        
    def allocate_hours(self, G, fp, storymode=False):
        """
        Fill the network with the given flight plan. For each sector of the flight plan, 
        add one to the load for each tranch of time (one hour tranches) in which the flight 
        is present in the sector.
        Changed in 2.9: completely changed (count number of flights per hour).
        Changed in 2.9.5: does change the load of the first and last sector.

        @input G: network.
        @input fp: flight plan to allocate.
        """
        if storymode:
            print "NM allocates the flight."
        path,times=fp.p,fp.times
        for i in range(1,len(path)-1):
            n=path[i]
            t1,t2=times[i]/60.,times[i+1]/60.
            h=0
            while h<t2:
                if h+1>t1:
                    if storymode:
                        print "Load of sector", n, "goes from",  G.node[n]['load'][h], "to", G.node[n]['load'][h]+1, "for interval", h, "--", h+1
                    G.node[n]['load'][h]+=1
                h+=1

    def allocate_peaks(self, G, fp, storymode=False):
        """
        Old version of previous method.
        """
        path,times=fp.p,fp.times
        for i,n in enumerate(path):
            t1,t2=times[i],times[i+1]
            ints=np.array([p[0] for p in G.node[n]['load_old']])
            caps=np.array([p[1] for p in G.node[n]['load_old']])
            i1=list(ints>=t1).index(True)
            i2=list(ints>=t2).index(True)
            if ints[i2]!=t2:
                G.node[n]['load_old'].insert(i2,[t2,caps[i2-1]])
            if ints[i1]!=t1:
                G.node[n]['load_old'].insert(i1,[t1,caps[i1-1]])
                i2+=1
            for k in range(i1,i2):
                G.node[n]['load_old'][k][1]+=1

    def deallocate_hours(self, G, fp):
        """
        New in 2.5: used to deallocate a flight plan not legit anymore (because one sector has been shutdown).
        Changed in 2.9: completely changed, based on hour tranches.
        @input G: network
        @input fp: flight plan to deallocate.
        """
        
        path,times=fp.p,fp.times
        for i in range(1, len(path)-1):
            n = path[i]
            t1,t2=times[i]/60.,times[i+1]/60.
            h=0
            while h<t2:
                if h+1>t1:
                    G.node[n]['load'][h]-=1
                h+=1

    def deallocate_peaks(self,fp):
        """
        Old version of previous method.
        """

        path,times=fp.p,fp.times
        for i,n in enumerate(path):
            t1,t2=times[i],times[i+1]
            ints=np.array([p[0] for p in G.node[n]['load_old']])
            i1=list(ints==t1).index(True)
            i2=list(ints==t2).index(True)
            for k in range(i1,i2):
                G.node[n]['load_old'][k][1]-=1
            
            if G.node[n]['load_old'][i2-1][1]==G.node[n]['load_old'][i2][1]:
                G.node[n]['load_old'].remove([t2,G.node[n]['load_old'][i2][1]])
            if G.node[n]['load_old'][i1-1][1]==G.node[n]['load_old'][i1][1]:
                G.node[n]['load_old'].remove([t1,G.node[n]['load_old'][i1][1]])

    def M0_to_M1(self, G, queue, N_shocks, tau, Nsp_nav, storymode=False):
        """
        Routine aiming at modelling the shut down of sectors due to bad weather or strikes. Some 
        sectors are shut down at random. Flight plans crossing these sectors are deallocated. 
        Shortest paths are recomputed. Finally, deallocated flights are reallocated on the 
        new network, with the same initial order. This procedure is repeated after each sector 
        is shut.
        Changed in 2.9: updated for navpoints and can now shut down sectors containing airports.
        Transferred from simulationO.

        @input G: network
        @input queue: initial queue before the shocks.
        @input N_shocks: number of sectors to shutdown.
        @input tau: parameter of shit in time (should not really be here...).
        @Nsp_nav: number of navpoints shortes paths per sector path (should not here either...)
        """
        sectors_to_shut = sample(G.nodes(), N_shocks)
        #sectors_to_shut = [33]

        for n in sectors_to_shut:
            flights_to_reallocate = []
            flights_suppressed = []          
            sec_pairs_to_compute = []
            nav_pairs_to_compute = []
            for f in queue:
                if f.accepted:
                    path_sec = f.fp_selected.p
                    if n in path_sec:
                        #print path_sec
                        if path_sec[0]==n or path_sec[-1]==n:
                            flights_suppressed.append(f) # The flight is suppressed if the source or the destination is within the shut sector
                        else:
                            flights_to_reallocate.append(f)
                            sec_pairs_to_compute.append((path_sec[0], path_sec[-1]))
                            nav_pairs_to_compute.append((f.fp_selected.p_nav[0], f.fp_selected.p_nav[-1]))

            sec_pairs_to_compute = list(set(sec_pairs_to_compute))
            nav_pairs_to_compute = list(set(nav_pairs_to_compute))
                    
            if storymode:
                first_suppressions = len(flights_suppressed)
                print
                print 'Shutting sector', n
                print 'Number of flights to be reallocated:', len(flights_to_reallocate)
                print 'Number of flights suppressed:', len(flights_suppressed)

            for f in flights_to_reallocate + flights_suppressed:
                self.deallocate(G, f.fp_selected)
                #queue.remove(f)
            
            G.shut_sector(n)
            G.build_H()
            G.G_nav.build_H()

            if storymode:
                print 'Recomputing shortest paths...'
            G.compute_all_shortest_paths(Nsp_nav, perform_checks = True, sec_pairs_to_compute = sec_pairs_to_compute, nav_pairs_to_compute = nav_pairs_to_compute)                
        

            #shuffle(flights_to_reallocate)
            
            for f in flights_to_reallocate:
                if not (f.source, f.destination) in G.G_nav.short.keys():
                    flights_suppressed.append(f)
                else:
                    f.compute_flightplans(tau, G)
                    self.allocate_flight(G, f)

            if storymode:
                print 'There were', len(flights_suppressed) - first_suppressions, 'additional flights which can not be allocated.'
            for f in flights_suppressed:
                for fp in f.FPs:
                    fp.accepted = False
                f.accepted = False

    def M0_to_M1_quick(self, G, queue, N_shocks, tau, Nsp_nav, storymode=False, sectors_to_shut = None):
        """
        Same method than previous one, but closes all sectors at the same time, then recomputes the hosrtest paths.
        New in 2.9.7.
        """
        if sectors_to_shut==None:
            #sectors_to_shut = shock_sectors(G, N_shocks)#sample(G.nodes(), N_shocks)
            sectors_to_shut = sample(G.nodes(), N_shocks)
        else:
            sectors_to_shut = [sectors_to_shut]
        #sectors_to_shut = [33]

        if sectors_to_shut!=[]:
            #for n in sectors_to_shut:
            flights_to_reallocate = []
            flights_suppressed = []          
            sec_pairs_to_compute = []
            nav_pairs_to_compute = []
            for f in queue:
                if f.accepted:
                    path_sec = f.fp_selected.p
                    if set(sectors_to_shut).intersection(set(path_sec))!=set([]):
                        #print path_sec
                        if path_sec[0] in sectors_to_shut or path_sec[-1] in sectors_to_shut:
                            flights_suppressed.append(f) # The flight is suppressed if the source or the destination is within the shut sector
                        else:
                            flights_to_reallocate.append(f)
                            sec_pairs_to_compute.append((path_sec[0], path_sec[-1]))
                            nav_pairs_to_compute.append((f.fp_selected.p_nav[0], f.fp_selected.p_nav[-1]))

            sec_pairs_to_compute = list(set(sec_pairs_to_compute))
            nav_pairs_to_compute = list(set(nav_pairs_to_compute))
                    
            if storymode:
                first_suppressions = len(flights_suppressed)
                print
                print 'Shutting sectors', sectors_to_shut
                print 'Number of flights to be reallocated:', len(flights_to_reallocate)
                print 'Number of flights suppressed:', len(flights_suppressed)

            for f in flights_to_reallocate + flights_suppressed:
                self.deallocate(G, f.fp_selected)
                #queue.remove(f)
            
            for n in sectors_to_shut:
                G.shut_sector(n)

            G.build_H()
            G.G_nav.build_H()

            if storymode:
                print 'Recomputing shortest paths...'
            G.compute_all_shortest_paths(Nsp_nav, perform_checks = True, sec_pairs_to_compute = sec_pairs_to_compute, nav_pairs_to_compute = nav_pairs_to_compute, verb = storymode)                
        

            #shuffle(flights_to_reallocate)
            
            for f in flights_to_reallocate:
                if not (f.source, f.destination) in G.G_nav.short.keys():
                    flights_suppressed.append(f)
                else:
                    f.compute_flightplans(tau, G)
                    self.allocate_flight(G, f)

            if storymode:
                print 'There were', len(flights_suppressed) - first_suppressions, 'additional flights which can not be allocated.'
                print
            for f in flights_suppressed:
                for fp in f.FPs:
                    fp.accepted = False
                f.accepted = False

class FlightPlan:
    """
    Class FlightPlan. 
    =============
    Keeps in memory its path, time of departure, cost and id of AC.
    Changed in 2.8: added p_nav.
    Changed in 2.9.6: added shift_time method.
    """
    def __init__(self,path,time,cost,ac_id, path_nav):
        self.p=path
        self.p_nav=path_nav
        self.t=time
        self.cost=cost
        self.ac_id=ac_id
        self.accepted=True
        self.bottleneck=-1

    def shift_time(self, shift):
        self.t += shift
        
class Flight:
    """
    Class Flight. 
    =============
    Keeps in memory its id, source, destination, prefered time of departure and id of AC.
    Thanks to AirCompany, keeps also in memory its flight plans (self.FPs).
    Changed in 2.9.6: Compute FPs added (coming from AirCompany object).
    New in 2.9.6: method shift_desired_time.
    """
    def __init__(self, Id, source, destination, pref_time, ac_id, par, Nfp): # Id is relative to the AC
        """
        Changed in 2.9.6: added Nfp.
        """
        self.id=Id
        self.source=source
        self.destination=destination
        self.pref_time=pref_time
        #self.FPs=FPs
        self.ac_id=ac_id 
        self.par=par
        self.Nfp = Nfp

    def compute_flightplans(self, tau, G): 
        """
        Compute the flight plans for a given flight, based on Nfp, Nsp_nav and the best paths, and the utility function.
        Changed in 2.2: tau introduced.
        Changed in 2.8: paths made of navpoints and then converted.
        Changed in 2.9: ai and aj are navpoints, not sectors.
        Changed in 2.9.6: use the convert_path method of the Net object.
        New in 2.9.6: comes from AirCompany object.
        Changed in 2.9.7: ai and aj are source and destination.
        """

        ai, aj = self.source, self.destination
        t0sp = self.pref_time
        try:
            for k, v in G.G_nav.short.items():
                assert len(v)==G.Nfp
        except:
            print k, len(v), G.Nfp
            raise

        SP=[(p, G.convert_path(p), G.G_nav.weight_path(p)) for p in G.G_nav.short[(ai,aj)]]

        #print 'len(SP)', len(SP)
        
        uworst=utility(self.par, SP[0][-1], t0sp, SP[-1][-1], t0sp)
                
        #u=[[(p,t0sp + i*tau,utility(self.par,G.weight_path(shortestPaths[0]),t0sp,G.weight_path(p),t0sp + i*tau)) for p in shortestPaths] for i in range(self.Nfp)\
        #    if utility(self.par,G.weight_path(shortestPaths[0]),t0sp,G.weight_path(shortestPaths[0]),t0sp + i*tau)<=uworst]
        u=[[(cp, t0sp + i*tau, utility(self.par, SP[0][-1], t0sp, c, t0sp + i*tau),p) for p,cp,c in SP] for i in range(self.Nfp)\
            if utility(self.par,SP[0][-1], t0sp, SP[0][-1],t0sp + i*tau)<=uworst]
            
        fp=[FlightPlan(a[0],a[1],a[2],self.id,a[3]) for a in sorted([item for sublist in u for item in sublist], key=lambda a: a[2])[:self.Nfp]]

        if len(fp)!=self.Nfp:
            raise Exception('Boum', len(fp))
    
        if not G.weighted:
            # ------------- shuffling of the flight plans with equal utility function ------------ #
            uniq_util=np.unique([item.cost for item in fp])
            sfp=[]
            for i in uniq_util:
                v=[item for item in fp if item.cost==i]
                shuffle(v)
                sfp=sfp+v
            fp=sfp
        
        self.FPs = fp
        
    def make_flags(self):
        """
        Used to remember the flight plans which were overloading the network, 
        as well as the first sector to be overloaded on the trajectories.
        """
        try:
            self.flag_first=[fp.accepted for fp in self.FPs].index(True)
        except:
            self.flag_first=len(self.FPs)
            
        self.overloadedFPs=[self.FPs[n].p for n in range(0,self.flag_first)]
        self.bottlenecks=[fp.bottleneck for fp in self.FPs if fp.bottleneck!=-1]

    def shift_desired_time(self, shift):
        """
        Shift the desired time of all flight plans of the flight.
        """
        #print 'shift', shift
        shift = int(shift)
        #print 'Shift: previous time:', self.pref_time, 'new time:', self.pref_time + shift
        self.pref_time += shift
        for fp in self.FPs:
            fp.shift_time(shift)
        
    def __repr__(self):
        return 'Flight number ' + str(self.id) + ' from AC number ' + str(self.ac_id) +\
            ' from ' + str(self.source) + ' to ' + str(self.destination)
        
class AirCompany:
    """
    Class AirCompany
    ================
    Keeps in memory the underlying network and several parameters, in particular the 
    coefficients for the utility function and the pairs of airports used.
    """
    def __init__(self, Id, Nfp, na, pairs, par):
        self.Nfp=Nfp
        self.par=par
        self.pairs=pairs
        #self.G=G
        self.na=na
        self.id=Id
        
    def fill_FPs(self, t0spV, tau, G, pairs=[]):
        """
        Fills na flights with Nfp flight plans each, between airports given by pairs.
        New in 2.9.5: can specify a pair of airports.
        Changed in 2.9.6: the flight computes the flight plans itself.
        """
        if pairs==[]:
            assigned_airports=sample(self.pairs, self.na) 
        else:
            assigned_airports = pairs
        self.flights=[]
        i=0
        for (ai,aj) in assigned_airports:
            self.flights.append(Flight(i, ai, aj, t0spV[i], self.id, self.par, self.Nfp))
            #self.flights[-1].FPs=self.add_flightplans(ai,aj,t0spV[i],tau, G)
            self.flights[-1].compute_flightplans(tau, G)
            i+=1
    
    # def convert_path(self, G, path):
    #     """
    #     New in 2.8: used to convert a path of navigation points into a path of sectors.
    #     """
    #     path_sec=[]
    #     j,sec=0,G.G_nav.node[path[0]]['sec']
    #     while j<len(path):
    #         path_sec.append(sec)
    #         while j<len(path) and G.G_nav.node[path[j]]['sec']==sec:
    #             j+=1
    #         if j<len(path):
    #             sec=G.G_nav.node[path[j]]['sec']
    #     return path_sec
        
    def add_dummy_flightplans(self,ai,aj,t0sp): 
        """
        New in 2.5: Add dummy flight plans to a given flight. Used if there is no route between source and destination airports.
        """
        
        fp=[FlightPlan([],t0sp,10**6,self.id) for i in range(self.Nfp)]
        
        return fp
        
    def __repr__(self):
        return 'AC with para ' + str(self.par)
        
class Net(nx.Graph):
    """
    Class Net
    =========
    Derived from nx.Graph. Several methods added to build it, generate, etc...
    """
    def __init__(self):
        super(Net, self).__init__()
        
    def import_from(self, G, numberize=False, verb = False):
        """
        Used to import the data of an already existing graph (networkx) in a Net obect.
        Weights and directions are conserved.
        Changed in 2.9: included case where G is empty
        """
        if verb:
            print 'Importing network...'
        if len(G.nodes())!=0:
            if not numberize:
                self.add_nodes_from(G.nodes(data=True))
                self.add_weighted_edges_from([(e[0],e[1],G[e[0]][e[1]]['weight']) for e in G.edges()])
            else:
                self.idx_sectors={s:i for i,s in enumerate(G.nodes())}
                for n in G.nodes():
                    self.add_node(self.idx_sectors[n], name=n, **G.node[n])
                for e in G.edges():
                    self.add_edge(self.idx_sectors[e[0]], self.idx_sectors[e[1]])

            e1=G.edges()[0]
            e2=G.edges()[1]
            self.weighted=not (self[e1[0]][e1[1]]['weight']==self[e2[0]][e2[1]]['weight']==1.)
            
            if verb:
                if self.weighted:
                    print 'Network was found weighted'
                else:
                    print 'Network was found NOT weighted'
                    print 'Example:', self[e1[0]][e1[1]]['weight']     
        else:
            print 'Network was found empty!'   
            
    def build_nodes(self, N, prelist=[]):
        """
        Add N nodes to the network, with coordinates taken uniformly in a square. 
        Alternatively, prelist gives the list of coordinates.
        New in 2.8.2
        """
        for i in range(N):
            self.add_node(i,coord=[uniform(-1.,1.),uniform(-1.,1.)])  
        for j,cc in enumerate(prelist):
            self.add_node(N+j,coord=cc)
            
    def build_net(self, N, Gtype='D',mean_degree=6):
        """
        Build edges, based on Delaunay triangulation or Erdos-Renyi graph. 
        No weight is computed at this point.
        """
        x,y =  np.array([self.node[n]['coord'][0] for n in self.nodes()]),np.array([self.node[n]['coord'][1] for n in self.nodes()])   
        cens,edg,tri,neig = triang.delaunay(x,y)
        if Gtype=='D':
            for p in tri:
                self.add_edge(p[0],p[1])#, weight=max(gauss(1., sigma),0.00001)) # generates weigthed links, centered on 1
                self.add_edge(p[1],p[2])#, weight=max(gauss(1., sigma),0.00001))
                self.add_edge(p[2],p[0])#, weight=max(gauss(1., sigma),0.00001))
        elif Gtype=='E':  
            prob=mean_degree/float(N-1) # <k> = (N-1)p - 6 is the mean degree in Delaunay triangulation.
            for n in self.nodes():
                for m in self.nodes():
                    if n!=m:
                        if np.random.rand()<=prob:
                            self.add_edge(n,m)
            
    def build(self, N, nairports, min_dis, generation_of_airports=True, Gtype='D', sigma=1., mean_degree=6, prelist=[]):
        """
        Build a graph of the given type, nodes, edges and possibly airports.
        """
        print 'Building random network of type', Gtype
        
        self.weighted=sigma==0.
        
        if Gtype=='D' or Gtype=='E':
            self.build_nodes(N, prelist=prelist)
            self.build_net(N, Gtype=Gtype, mean_degree=mean_degree)  
        elif Gtype=='T':
            xAxesNodes=np.sqrt(N/float(1.4))
            self=build_triangular(xAxesNodes)  

        #if generation_of_airports:
        #    self.generate_airports(nairports,min_dis) 
        
    def generate_weights(self,typ='gauss',par=[1.,0.01]):#,values=[]):
        """
        Generates weights with a gaussian distribution or given by the euclidean distance
        between nodes, with the factor tuned so that the avreage matches the one given as argument.

        @input typ: should be gauss or coords.
        @input par: is typ is 'gauss', gives the mean and deviation. Otherwise, should be a flot giving 
        the average weight.
        """
        assert typ in ['gauss', 'manual', 'coords']
        self.typ_weights, self.par_weights=typ, par
        if typ=='gauss':
            mu=par[0]
            sigma=par[1]
            for e in self.edges():
                self[e[0]][e[1]]['weight']=max(gauss(mu, sigma),0.00001)
        elif typ=='coords':
            for e in self.edges():
                #self[e[0]][e[1]]['weight']=sqrt((self.node[e[0]]['coord'][0] - self.node[e[1]]['coord'][0])**2 +(self.node[e[0]]['coord'][1] - self.node[e[1]]['coord'][1])**2)
                self[e[0]][e[1]]['weight']=np.linalg.norm(np.array(self.node[e[0]]['coord']) - np.array(self.node[e[1]]['coord']))
            avg_weight=np.mean([self[e[0]][e[1]]['weight'] for e in self.edges()])
            for e in self.edges():
                self[e[0]][e[1]]['weight']=par*self[e[0]][e[1]]['weight']/avg_weight
        self.typ_weights=typ
        self.weighted=True

    def fix_weights(self, weights, typ='data'):
        """
        Fix the weights from a dictionnary given as argument.
        @input weights: dictionnary of dictionnaries whose values are the weights.
        """
        for e,w in weights.items():
            self[e[0]][e[1]]['weight']=w
        self.typ_weights=typ
        self.weighted=True
            
    def generate_capacities(self, C=5, typ='constant', par=[1]):#, file_capacities=None): #func=lambda a:a):
        """
        Generates capacities with different distributions.
        If typ is 'constant', all nodes have the same capacity, given by C.
        If typ is 'gauss', the capacities are taken from a normal distribution with mean C
        and standard deviation par[0]
        If typ is 'uniform', the capacities are taken from a uniform distribution, with bounds C-par[0]/2.,C+par[0]/2.
        If typ is 'manual', the capacities are taken from the file given as argument.
        If typ is 'areas', the capacities are proportional to the area of the sector, with a mean close to C
        If typ is 'lognormal', the capacities are taken from a lognormal distribution.
        New in 2.7: added lognormal and areas.
        Changed in 2.9.8: removed "manual"
        """
        assert typ in ['constant', 'gauss', 'uniform', 'time', 'manual', 'lognormal', 'areas']
        self.C, self.typ_capacities, self.par_capacities = C, typ, par
        if typ=='constant':
            for n in self.nodes():
                self.node[n]['capacity']=C
        elif typ=='gauss':
            for n in self.nodes():
                self.node[n]['capacity']=max(1,int(gauss(C,par[0])))
        elif typ=='uniform':
            for n in self.nodes():
                self.node[n]['capacity']=max(1,int(uniform(C-par[0]/2.,C+par[0]/2.)))
        elif typ=='lognormal':
            for n in self.nodes():
                self.node[n]['capacity']=max(1,int(lognormal(log(C),par[0])))
        #elif typ=='manual':
        #    f=open(file_capacities,'r')
        #    properties=pickle.load(f)
        #    f.close()
        #    for n in self.nodes():
        #        self.node[n]['capacity']=properties[n]['capacity']
        elif typ=='areas':
            if par[0]=='sqrt':
                area_avg=np.mean([sqrt(self.node[n]['area']) for n in self.nodes()])
                alpha=C/area_avg
                for n in self.nodes():
                    self.node[n]['capacity']=max(1,int(alpha*sqrt(self.node[n]['area'])))
        
    def fix_capacities(self, capacities, typ='exterior'):
        """
        Fix the capacities given with the dictionnary of capacities.
        Checks if every sector has a capacity.
        New in 2.9.
        """

        self.typ_capacities = typ
        for n,v in capacities.items():
            self.node[n]['capacity'] = v

        try:
            for n in self.nodes():
                assert self.node[n].has_key('capacity')
        except AssertionError:
            raise Exception("One sector at least does not have a capacity!")

    def generate_airports(self, nairports, min_dis, C_airport=10):
        """
        Generates nairports airports. Builds the accessible pairs of airports for this network
        with a  minimum distance min_dis. Sets the capacity of the airport with the argument C_airport.
        """
        self.airports=sample(self.nodes(),nairports)
        #self.pairs=[(ai,aj) for ai in self.airports for aj in self.airports if len(nx.shortest_path(self, ai, aj))-2>=min_dis]
        self.short={(ai,aj):[] for ai in self.airports for aj in self.airports if len(nx.shortest_path(self, ai, aj))-2>=min_dis}
        
        for a in self.airports:
            self.node[a]['capacity']=100000                 # TODO: check this.
            self.node[a]['capacity_airport'] = C_airport

    def fix_airports(self, airports, min_dis, pairs=[], C_airport=10, singletons=False):
        """
        Fix airports given by user. The pairs can be given also by the user, 
        or generated automatically, with minimum distance min_dis.
        @input min_dis: minimum distance -- in nodes, excluding the airports -- betwen a pair of airports.
        @input pairs: pairs can be given by user, or computed automatically if set to [].
        @input C_airport: capatity of airports.
        @input singletons: if set to True, pairs in which the source is identical to the destination are authorized (but min_dis has to be smaller or equal to 2.)
        """
        self.airports=airports
        if pairs==[]:
            self.short = {(ai,aj):[] for ai in self.airports for aj in self.airports if len(nx.shortest_path(self, ai, aj))-2>=min_dis and ((not singletons and ai!=aj) or singletons)}
            #self.pairs=[(ai,aj) for ai in self.airports for aj in self.airports if len(nx.shortest_path(self, ai, aj))-2>=min_dis]
        else:
            self.short = {(ai,aj):[] for (ai,aj) in pairs if ((not singletons and ai!=aj) or singletons)}# if len(nx.shortest_path(self, ai, aj))-2>=min_dis and ((not singletons and ai!=aj) or singletons)}
            #self.pairs=pairs

        for a in self.airports:
            #self.node[a]['capacity']=100000
            self.node[a]['capacity_airport'] = C_airport
 
    def infer_airports_from_navpoints(self, C_airport, singletons=False):
        """
        Detects all sectors having at least one navpoint begin a source or a destination. 
        Mark them as airports, with a given capacity.
        Shouldd only be used by sector networks.
        @input C_airport: capacity of the airports.
        @input singletons: passed to @fix_airports.
        """
        pairs = []
        for (p1,p2) in self.G_nav.short.keys():
            s1 = self.G_nav.node[p1]['sec']
            s2 = self.G_nav.node[p2]['sec']
            pairs.append((s1,s2))
        pairs = list(np.unique(pairs))
        self.fix_airports(np.unique([self.G_nav.node[n]['sec'] for n in self.G_nav.airports]), -10, C_airport = C_airport, pairs=pairs, singletons=singletons)

    def infer_airports_from_short_list(self):
        """
        Compute the list of airports from all possible pairs of source/destination.
        """
        self.airports = list(set([a for b in self.short.keys() for a in b]))

    def build_H(self): 
        """
        Build the DiGraph obect used in the ksp_yen algorithm.
        """
        self.H=DiGraph()
        self.H._data={}
        for n in self.nodes():
            self.H.add_node(n)
        for e in self.edges():
            self.H.add_edge(e[0],e[1], cost=self[e[0]][e[1]]['weight'])
            self.H.add_edge(e[1],e[0], cost=self[e[1]][e[0]]['weight'])
            
    def weight_path(self,p): 
        """
        Return the weight of the given path. Can be used both for navpoints and sectors.
        """
        return sum([self[p[i]][p[i+1]]['weight'] for i in range(len(p)-1)])
        
    def compute_shortest_paths(self, Nfp, repetitions=True, use_sector_path=False, old=False, pairs = [], verb = 1):
        """
        Pre-Build Nfp weighted shortest paths between each pair of airports. 
        If repetitions is set to True, a path can have a given node twice or more. Otherwise, the function makes
        several iterations, considering longer and longer paths which don't have a repeated sector.
        If use_sector_path is set to True, then the navpoints paths are generated so that the sector paths do not
        have repeated sectors.
        If the function do not find enough paths, the corresponding source/destination pair is deleted.
        Changed in 2.9: added singletons option. Added repetitions options to avoid repeated sectors in paths.
        Changed in 2.9.4: added procedure to have always 10 distinct paths (in sectors).
        Changed in 2.9.7: modified the location of the not enough_path loop to speed up the process. Added
        pairs_to_compute, so that it does not necesseraly recompute every shortest paths.
        """
        if use_sector_path:
            try:
                assert self.type == 'sec'
            except:
                Exception('use_sector_path should not be used with navpoints networks.')

        if pairs==[]:
            pairs = self.short.keys()[:]

        #Nfp_init=Nfp
        #paths_additional=0
        
        deleted_pairs = []
        
        if repetitions:
            for (a,b) in pairs:
                enough_paths=False
                Nfp_init=Nfp
                while not enough_paths:
                    enough_paths=True
                    #self.short={(a,b):self.kshortestPath(a, b, Nfp, old=old) for (a,b) in self.short.keys()}
                    paths = self.kshortestPath(a, b, Nfp, old=old)
                    if len(paths) < Nfp_init:
                        enough_paths = False
                self.short[(a,b)] = paths[:]
                Nfp = Nfp_init
        else:
            if not use_sector_path:
                #pairs = self.short.keys()[:]
                for it, (a,b) in enumerate(pairs):
                    #if verb:
                    #    counter(it, len(pairs), message='Computing shortest paths...')
                    if a!=b:
                        enough_paths=False
                        Nfp_init=Nfp
                        while not enough_paths:
                            enough_paths=True
                            paths = self.kshortestPath(a, b, Nfp, old=old) #Initial set of paths
                            previous_duplicates=1
                            duplicates=[]
                            n_tries = 0
                            while len(duplicates)!=previous_duplicates and n_tries<50:
                                previous_duplicates=len(duplicates)
                                duplicates=[]
                                for sp in paths:
                                    if len(np.unique(sp))<len(sp): # Detect if some sectors are duplicated within sp
                                        duplicates.append(sp)

                                if len(duplicates)!=previous_duplicates: # If the number of duplicates has changed, compute some more paths.
                                    paths = self.kshortestPath(a, b, Nfp+len(duplicates), old=old)
                                n_tries+=1

                            #paths_init = paths[:]
                            for path in duplicates:
                                paths.remove(path)

                            #self.short[(a,b)]=paths[:]

                            try:
                                assert len(paths)==Nfp and len(duplicates)==previous_duplicates
                                enough_paths=True
                                paths = [list(vvv) for vvv in set([tuple(vv) for vv in paths])][:Nfp_init]
                                if len(paths) < Nfp_init:
                                    enough_paths = False
                                    print 'Not enough paths, doing another round (' + str(Nfp +1 - Nfp_init), 'additional paths).'
                                #paths_additional += 1
                                Nfp += 1
                            except AssertionError:
                                #print 'a:', a, 'b:', b, 'len(self.short[(a,b)]):', len(self.short[(a,b)])
                                print "kspyen can't find enough paths (only " + str(len(paths)) + ')', "for the pair", a, b, "; I delete this pair."
                                print 'Number of duplicates:', len(duplicates)
                                #print 'Number of paths with duplicates:', len(paths_init)
                                deleted_pairs.append((a,b))
                                del self.short[(a,b)]
                                #self.pairs = list(self.pairs)
                            except:
                                raise
                        Nfp = Nfp_init
                        if self.short.has_key((a,b)):
                            self.short[(a,b)] = paths[:]

            else:
                for it, (a,b) in enumerate(pairs):
                    #if verb:
                    #    counter(it, len(pairs), message='Computing shortest paths...')
                    if a!=b:
                        enough_paths=False
                        Nfp_init=Nfp
                        while not enough_paths:
                            enough_paths=True
                        #duplicates=[]
                            paths_nav = self.kshortestPath(a, b, Nfp, old=old)
                            paths = [self.convert_path(p) for p in paths_nav]
                            previous_duplicates=1
                            duplicates=[]
                            duplicates_nav=[]
                            while len(duplicates)!=previous_duplicates:
                                previous_duplicates=len(duplicates)
                                duplicates=[]
                                duplicates_nav=[]
                                for j,sp in enumerate(paths):
                                    if len(np.unique(sp))<len(sp):
                                        duplicates.append(sp)
                                        duplicates_nav.append(paths_nav[j])
                                
                                if len(duplicates)!=previous_duplicates:
                                    paths_nav=self.kshortestPath(a, b, Nfp+len(duplicates))
                                    paths = [self.convert_path(p) for p in paths_nav]

                            for path in duplicates_nav:
                                paths_nav.remove(path)
                                #print len(self.short[(a,b)])
                            #self.short[(a,b)]=paths_nav[:]

                            try:
                                assert len(paths_nav)==Nfp
                                enough_paths=True
                                paths_nav = [list(vvv) for vvv in set([tuple(vv) for vv in paths_nav])][:Nfp_init]
                                if len(paths_nav) < Nfp_init:
                                    enough_paths = False
                                    print 'Not enough paths, doing another round (' + str(Nfp - Nfp_init), 'additional paths).'
                                #paths_additional += 1
                                Nfp += 1
                            except AssertionError:
                                #print 'a:', a, 'b:', b, 'len(self.short[(a,b)]):', len(self.short[(a,b)])
                                print "kspyen can't find enough paths (only " + str(len(paths)) + ')', "for the pair", a, b
                                print 'Number of duplicates:', len(duplicates)
                                deleted_pairs.append((a,b))
                                del self.short[(a,b)]
                                #self.pairs = list(self.pairs)
                                raise
                            except:
                                raise  
                        Nfp = Nfp_init
                        if self.short.has_key((a,b)):
                            #assert len(paths_nav) == Nfp_init
                            self.short[(a,b)] = paths_nav[:]                          

        # enough_paths=True
        # for (a,b) in self.short.keys():
        #     if a!=b:
        #         #print 'Number of paths:', len(self.short[(a,b)])
        #         #print 'Number of distinct paths:', len(set([tuple(vv) for vv in self.short[(a,b)]]))
        #         self.short[(a,b)] = [list(vvv) for vvv in set([tuple(vv) for vv in self.short[(a,b)]])][:Nfp_init]
        #         if len(self.short[(a,b)]) < Nfp_init:
        #             enough_paths = False
        #paths_additional += 1
        # Nfp += 1
        # if not enough_paths:
        #     print 'Not enough paths, doing another round (', #paths_additional, 'additional paths).'

        #self.pairs = self.short.keys()
        return list(set(deleted_pairs))

    def compute_sp_restricted(self, Nfp, silent=True, pairs = []):
        """
        New in 2.9.0: Computes the k shortest paths of navpoints restricted to shortest path of sectors.
        Changed in 2.9.1: save all Nfp shortest paths of navpoints for each paths of sectors (Nfp also).
        Changed in 2.9.7: added pairs kwarg in order to compute less paths.
        Note: this is black magic. Don't make any change unless you know what you are doing. And you probably don't.
        """
        class NoShortH(Exception):
            pass
        #self.G_nav.short={}
        if not hasattr(self, 'short_nav'):
            self.short_nav={}

        if pairs ==[]:
            pairs = self.G_nav.short.keys()[:]
        else:
            pairs = [p for p in pairs if p in self.G_nav.short.keys()]

        for idx,(p0,p1) in enumerate(pairs):
            #counter(idx, len(pairs), message='Computing shortest paths (navpoints)...')
            s0,s1=self.G_nav.node[p0]['sec'], self.G_nav.node[p1]['sec']
            self.G_nav.short[(p0,p1)] = []
            self.short_nav[(p0,p1)] = {}
            assert len(self.short[(s0,s1)]) == self.Nfp
            try:
                for idx_sp, sp in enumerate(self.short[(s0,s1)]): # Compute the network of navpoint restricted of each shortest paths.
                    H_nav=NavpointNet()
                    with silence(silent):
                        #print 'Shortest path in sectors:', sp
                        HH=nx.Graph()
                        # Add every nodes in the sectors of the shortest paths.
                        for n in self.G_nav.nodes(): 
                            if self.G_nav.node[n]['sec'] in sp:
                                HH.add_node(n, **self.G_nav.node[n])

                        for e in self.G_nav.edges():
                            s0=self.G_nav.node[e[0]]['sec']
                            s1=self.G_nav.node[e[1]]['sec']
                            if s0!=s1 and s0 in sp and s1 in sp:
                                idxs_s0=np.where(np.array(sp)==s0)[0]
                                idxs_s1=np.where(np.array(sp)==s1)[0]
                                for idx_s0 in idxs_s0: # In case there is a repetition of s0 in sp.
                                    for idx_s1 in idxs_s1:
                                        if ((idx_s0<len(sp)-1 and sp[idx_s0+1]==s1) or (idx_s1<len(sp)-1 and sp[idx_s1+1]==s0)):
                                            HH.add_edge(*e, weight=self.G_nav[e[0]][e[1]]['weight'])
                            elif s0==s1 and s0 in sp: # if both nodes are in the same sector and this sector is in the shortest path, add the edge.
                                HH.add_edge(*e, weight=self.G_nav[e[0]][e[1]]['weight'])
                        H_nav.import_from(HH)
                        if len(H_nav.nodes())!=0:
                            try:
                                for i in range(len(sp)-1):
                                    found=False
                                    ss1, ss2=sp[i], sp[i+1]
                                    for n1, n2 in self.G_nav.edges():
                                        if (self.G_nav.node[n1]['sec']==ss1 and self.G_nav.node[n2]['sec']==ss2) or (self.G_nav.node[n1]['sec']==ss2 and self.G_nav.node[n2]['sec']==ss1):
                                            found=True
                                            if ss1==0 and ss2==31:
                                                print n1, n2, self.G_nav.node[n1]['sec'], self.G_nav.node[n2]['sec'], H_nav.has_node(n1), H_nav.has_node(n1), H_nav.has_edge(n1,n2)
                                    # if found==False:
                                    #     print 'Problem: sectors', ss1, 'and', ss2, 'are not adjacent in terms of navpoints.'
                                    # else:
                                    #     print 'sectors', ss1 , 'and', ss2, 'are adjacent.'
                                for i in range(len(sp)-1):
                                    found=False
                                    ss1, ss2=sp[i], sp[i+1]
                                    for n1, n2 in H_nav.edges():
                                        if (self.G_nav.node[n1]['sec']==ss1 and self.G_nav.node[n2]['sec']==ss2) or (self.G_nav.node[n1]['sec']==ss2 and self.G_nav.node[n2]['sec']==ss1):
                                            found=True
                                    # if found==False:
                                    #     print 'Problem: sectors', ss1, 'and', ss2, 'are not adjacent in terms of navpoints (H).'
                                    # else:
                                    #     print 'sectors', ss1 , 'and', ss2, 'are adjacent (H).'

                                H_nav.fix_airports([p0,p1], 0.)
                                H_nav.build_H()
                                try:
                                    H_nav.compute_shortest_paths(Nfp, repetitions=False, use_sector_path=True, old=False, verb = 0)
                                except AssertionError:
                                    raise NoShortH('')
                                except:
                                    raise
                                # try:
                                #     for v in H_nav.short.values():
                                #         assert len(v) == Nfp
                                # except AssertionError:
                                #     print len(v), Nfp
                                #     raise
                                # except:
                                #     raise

                                for k, p in enumerate(H_nav.short[(p0,p1)]):
                                    if self.convert_path(p)!=sp:
                                        print 'Alert: discrepancy between theoretical path of sectors and final one!'
                                        print 'Path number', k
                                        print 'Theoretical one:', sp, self.weight_path(sp)
                                        print 'Actual one:',  self.convert_path(p), self.weight_path(self.convert_path(p))        
                                        raise Exception("Problem")                        

                                shorts=[p for p in  H_nav.short[(p0,p1)] if self.convert_path(p)==sp]
                                assert len(shorts)==self.Nfp
                                self.G_nav.short[(p0,p1)] = self.G_nav.short.get((p0,p1),[]) + shorts
                                self.short_nav[(p0,p1)][tuple(sp)] = shorts
                                assert len(self.short_nav[(p0,p1)][tuple(sp)])==self.Nfp
                            except nx.NetworkXNoPath:
                                print 'No restricted shortest path between' ,p0, 'and', p1, 'but I carry on'
                                cc=nx.connected_components(H_nav)
                                print 'Composition of connected components (sectors):'
                                for c in cc:
                                    print np.unique([self.G_nav.node[n]['sec'] for n in c])
                                #print 'Everybody is attached:', check_everybody_is_attached(H_nav)
                                #print H_nav.nodes()
                                #print H_nav.edges()
                                raise
                            except:
                                #print 'Unexpected error:', sys.exc_info()[0]
                                raise 
                        else:
                            print 'The subgraph was empty, I carry on.'
            except AssertionError:
                print 'There is a problem with the number of shortest paths'
                raise
            except NoShortH:
                #self.G_nav.short[(p0,p1)]
                print "I can't find enough paths for this pair, so I delete it."
                del self.G_nav.short[(p0,p1)]
                del self.short_nav[(p0,p1)] 
            except:
                raise
            # try:
            #     assert len(self.short_nav[(p0,p1)]) == self.Nfp
            # except:
            #     print len(self.short_nav[(p0,p1)])
            #     raise
            # if self.G_nav.short.has_key((p0,p1)):
            #     self.G_nav.short[(p0,p1)] = sorted(list(set([tuple(o) for o in G.G_nav.short[(p0,p1)]])), key= lambda p: G.G_nav.weight_path(p))[:Nfp]
 
        #pairs = self.G_nav.short.keys()[:]
        for p in pairs:
            if self.G_nav.short.has_key(p) and self.G_nav.short[p]==[]:
                del self.G_nav.short[p]
                print 'I deleted pair', p ,'because no path was computed.'

    def compute_all_shortest_paths(self, Nsp_nav, perform_checks = False, sec_pairs_to_compute = [], nav_pairs_to_compute = [], verb = True):
        """
        Gather several methods to ensure a consistency between navpoints and sectors.
        """
        if verb:
            print 'Computing shortest paths for sectors...'
        pairs_deleted = self.compute_shortest_paths(self.Nfp, repetitions=False, old=False, pairs = sec_pairs_to_compute, verb = int(verb))   
        self.infer_airports_from_short_list()
        #self.G_nav.infer_airports_from_sectors(self.airports, paras_G['min_dis'])

        pairs = self.G_nav.short.keys()[:]
        for (p1,p2) in pairs:
            s1=self.G_nav.node[p1]['sec']
            s2=self.G_nav.node[p2]['sec']
            if (s1, s2) in pairs_deleted:
                del self.G_nav.short[(p1,p2)]
                #self.G_nav.pairs.remove((p1,p2))
                if verb:
                    print 'I removed the pair of navpoints', (p1,p2), ' because the corresponding pair', (s1, s2), ' has been removed.'

        if verb:
            print 'Computing shortest paths for navpoints...'
        self.G_nav.infer_airports_from_short_list()

        #print 'len(self.G_nav.short.keys()):', len(self.G_nav.short.keys())   
        self.compute_sp_restricted(self.Nfp, silent = not verb, pairs = nav_pairs_to_compute)
        self.G_nav.infer_airports_from_short_list()

        self.choose_short(Nsp_nav, pairs = nav_pairs_to_compute)

        if perform_checks:
            if verb:
                print
                print 'Performing checks...'
            with silence(not verb):
                self.check_repair_sector_airports()
                #print 'len(self.G_nav.short.keys()):', len(G.G_nav.short.keys())
                self.check_airports_and_pairs()
    
    def kshortestPath(self, i, j, k, old=False): 
        """
        Return the k weighted shortest paths on the network. Uses the DiGraph.
        """
        if not old:
            spath = [a['path'] for a in  ksp_yen(self.H, i, j, k)]
        else:
            spath = [a['path'] for a in  ksp_yen_old(self.H, i, j, k)]
        spath=sorted(spath, key=lambda a:self.weight_path(a))
        
        spath_new, ii = [], 0
        while ii<len(spath):
            w_old=self.weight_path(spath[ii])
            a=[spath[ii][:]]
            ii+=1
            while ii<len(spath) and abs(self.weight_path(spath[ii]) - w_old)<10**(-8.):
                a.append(spath[ii][:])
                ii+=1
            shuffle(a)
            spath_new+=a[:]

        return spath_new
        
    def choose_short(self, Nsp_nav, pairs = []):
        """
        New in 2.9: used to Nfp shortest path such as there are Nsp_nav nav-shortest path for each sector-shortest path.
        """
        if pairs == []:
            pairs = self.G_nav.short.keys()
        else:
            pairs = [p for p in pairs if p in self.G_nav.short.keys()]
        self.Nsp_nav = Nsp_nav
        assert self.type=='sec'
        for (p0,p1) in pairs:
            assert len(self.short_nav[(p0,p1)])==10
            self.G_nav.short[(p0,p1)] = sorted([path for paths in self.short_nav[(p0,p1)].values() for path in paths[:Nsp_nav]],\
                key= lambda p: self.G_nav.weight_path(p))[:self.Nfp]
    
    def convert_path(self,path):
        """
        New in 2.8: used to convert a path of navigation points into a path of sectors.
        """
        path_sec=[]
        j,sec=0,self.G_nav.node[path[0]]['sec']
        while j<len(path):
            path_sec.append(sec)
            while j<len(path) and self.G_nav.node[path[j]]['sec']==sec:
                j+=1
            if j<len(path):
                sec=self.G_nav.node[path[j]]['sec']
        return path_sec
            
    def basic_statistics(self, rep='.'):
        os.system('mkdir -p ' + rep)
        f=open(rep + '/basic_stats_net.txt','w')
        print >>f, 'Mean/std degree:', np.mean([self.degree(n) for n in self.nodes()]), np.std([self.degree(n) for n in self.nodes()])
        print >>f, 'Mean/std weight:', np.mean([self[e[0]][e[1]]['weight'] for e in self.edges()]), np.std([self[e[0]][e[1]]['weight'] for e in self.edges()])
        print >>f, 'Mean/std capacity:', np.mean([self.node[n]['capacity'] for n in self.nodes() if not n in self.airports]),\
            np.std([self.node[n]['capacity'] for n in self.nodes() if not n in self.airports])
        #print >>f, 'Mean/std load:', np.mean([np.mean(self.node[n]['load']) for n in self.nodes()]), np.std([np.mean(self.node[n]['load']) for n in self.nodes()])
        f.close()
        
    def shut_sector(self, n):
        """
        Changed in 2.9.6: Can shut an airport, so we delete coresponding pairs in the short list.
        We don't put a high value on the links anymore. We delete the node from the network.
        """
        
        for a1, a2 in self.short.keys():
            if a1 == n or a2 == n:
                del self.short[(a1,a2)]

        self.remove_node(n)
                
        deleted_navpoints = []
        for nav in self.G_nav.nodes():
            if self.G_nav.node[nav]['sec'] == n:
                self.G_nav.remove_node(nav)
                deleted_navpoints.append(nav)

        deleted_pairs = []
        for a1,a2 in self.G_nav.short.keys():
            if a1 in deleted_navpoints or a2 in deleted_navpoints:
                deleted_pairs.append((a1,a2))
                del self.G_nav.short[(a1,a2)]

        return deleted_pairs
   
    def show(self, stack=False, colors='b'):
        draw_network_map(self, title=self.name, load=False, generated=True, airports=hasattr(self,'airports'))

    def give_airports_to_network(self, airports, Nsp_nav, min_dis = 2, C_airport=None, singletons=False, repetitions=False, old=False, name=None):
        """
        New in 2.9.4: Gather several methods of net. Used to give a new pair of airports to the network.
        """
        if C_airport == None:
            for a in self.airports:
                C_airport = self.node[a]['capacity_airport']
        self.fix_airports(airports, min_dis, C_airport = C_airport)
        self.compute_shortest_paths(self.Nfp, singletons=singletons, repetitions=repetitions, old=old)

        airports_nav=[choice([n for n in self.G_nav.nodes() if self.G_nav.node[n]['sec']==sec]) for sec in self.airports]
        self.G_nav.fix_airports(airports_nav, min_dis)
        self.compute_sp_restricted(self.Nfp, silent = True)

        self.G_nav.compute_pairs_based_on_short(min_dis) 
        self.choose_short(Nsp_nav)

        if name!=None:
            self.name = name
        else:
            self.name = split(self.name,'_')[2] + str(airports[0]) + '_' + str(airports[1]) + '_' + split(self.name,'_')[-1]
            for j in range(2,len(split(self.name,'_'))):
                self.name+= '_' + split(self.name,'_')[j]

        #return G

    def check_airports_and_pairs(self):
        """
        New in 2.9: check if eveything is consistent between the airports of navpoints, sectors, and the pairs.
        """

        try:
            airports_from_sector_short = list(set([a for b in self.short.keys() for a in b]))
            try:
                print 'Checking consistency of sector airports...'
                assert airports_from_sector_short == list(self.airports)
            except AssertionError:
                print 'Airports of sectors from short and airports list are not consistent:'
                for p in airports_from_sector_short:
                    if not p in self.airports:
                        print 'Pair', p, 'is in short of sectors but not in airports.'
                for p in self.airports:
                    if not p in airports_from_sector_short:
                        print 'Pair', p, 'is in airports but not in short of sectors.'
                print 
                raise
            except:
                raise


            airports_from_navs_short = list(set([a for b in self.G_nav.short.keys() for a in b]))
            try:
                print 'Checking consistency of navpoints airports...'
                assert airports_from_navs_short == list(self.G_nav.airports)
            except AssertionError:
                print 'Airports of navpoints from short and airports list are not consistent:'
                for p in airports_from_navs_short:
                    if not p in self.G_nav.airports:
                        print 'Pair', p, 'is in short of navpoints but not in airports.'
                for p in self.G_nav.airports:
                    if not p in airports_from_navs_short:
                        print 'Pair', p, 'is in airports but not in short of navpoints.'
                print
                raise
            except:
                raise

            airports_sec_from_navs = list(set([self.G_nav.node[a]['sec'] for b in self.G_nav.short.keys() for a in b]))

            try:
                print 'Checking consistency between sector and navpoints airports...'
                assert airports_sec_from_navs == list(self.airports)
            except AssertionError:
                print 'Airports of sectors infered from short of navpoints is not consistent with list of sector airports:'
                for p in airports_sec_from_navs:
                    if not p in self.airports:
                        print 'Pair', p, 'is in airports of sectors based on navpoints but not in airports of sectors.'
                for p in self.airports:
                    if not p in airports_sec_from_navs:
                        print 'Pair', p, 'is in airports of sectors but not in airports of sectors based on navpoints.'  
                print          
                raise
            except:
                raise
        except:
            raise

    def check_repair_sector_airports(self):
        pairs_sec_from_navs = list(set([(self.G_nav.node[p1]['sec'], self.G_nav.node[p2]['sec']) for (p1,p2) in self.G_nav.short.keys()]))
        for (s1,s2) in self.short.keys():
            if not (s1,s2) in pairs_sec_from_navs:
                del self.short[(s1,s2)]
                print 'I remove', s1, s2, 'from the airports of sectors because there is no corresponding pairs in navpoints.'

    def check_all_real_flights_are_legitimate(self, flights_selected):
        """
        Changed in 2.9.8: flights_selected is an external argument.
        """
        for f in flights_selected:
            try:
                assert (self.G_nav.idx_navs[f['route_m1'][0][0]], self.G_nav.idx_navs[f['route_m1'][-1][0]]) in self.G_nav.short.keys()
            except AssertionError:
                print 'A flight has a ouple source/destination not in the list of pairs of the network.'
                print f
                raise
            except:
                raise

            navpoints = set([self.G_nav.idx_navs[p[0]] for p in f['route_m1']])
            try:
                assert navpoints.issubset(set(self.G_nav.nodes()))
            except AssertionError:
                print 'A flight has navpoints not existing in the network:'
                print f
                print 'Navpoints not in network:', navpoints.difference(set(self.G_nav.nodes()))
                raise
            except:
                raise

            edges = set([(self.G_nav.idx_navs[f['route_m1'][i][0]], self.G_nav.idx_navs[f['route_m1'][i+1][0]]) for i in range(len(f['route_m1']) -1)])

            for e1, e2 in edges:
                try:
                    assert e2 in self.G_nav.neighbors(e1)
                except AssertionError:
                    print 'A flight is going from one point to the another while they are not connected:'
                    print f
                    print 'Edges not in network:', e1, e2
                    raise
                except:
                    raise

class NavpointNet(Net):
    """
    Dedicated class for navpoint networks.
    New in 2.8.2.
    """
    def __init__(self):
        super(NavpointNet, self).__init__()
        
    def build_nodes(self,N, sector_list=[],navpoints_borders=[], shortcut=0.):
        """
        Build the nodes.
        """
        j=0
        for i,cc in enumerate(navpoints_borders):
            self.add_node(j,coord=cc)
            j+=1
        self.navpoints_borders=range(j)
        for i,cc in enumerate(sector_list):
            self.add_node(j,coord=cc)
            j+=1
        for i in range(N):
            cc=[uniform(-1.,1.),uniform(-1.,1.)]
            if shortcut!=0.:
                not_too_close=True
                k=0
                while not_too_close and k<len(self.nodes()):
                    pcc=self.node[self.nodes()[k]]['coord']
                    not_too_close=np.linalg.norm(np.array(cc) - np.array(pcc))>shortcut
                    k+=1
            if shortcut==0. or not_too_close:
                self.add_node(j,coord=cc)
                j+=1
       
    def build(self, N, nairports, min_dis,generation_of_airports=True,Gtype='D', sigma=1.,mean_degree=6,  sector_list=[],navpoints_borders=[], shortcut=0.):
        """
        Build a graph of the given type. Build also the corresponding graph used for ksp_yen.
        Build_nodes externalized in 2.8.2
        """
        print 'Building random network of type ', Gtype
        
        self.weighted=sigma==0.
        
        if Gtype=='D' or Gtype=='E':
            self.build_nodes(N, sector_list=sector_list,navpoints_borders=navpoints_borders, shortcut=shortcut)
            self.build_net(N, Gtype=Gtype, mean_degree=mean_degree)
            
        elif Gtype=='T':
            xAxesNodes=np.sqrt(N/float(1.4))
            self=build_triangular(xAxesNodes)  

        if generation_of_airports:
            self.generate_airports(nairports,min_dis) 
        
    def clean_borders(self):
        """
        Remove all links between border points. Remover all links between navpoints in two different sectors
        which are both non border points.
        """
        for e in self.edges():
            if e[0] in self.navpoints_borders and e[1] in self.navpoints_borders:
                #print "I am removing edge", e, "between two points on the borders"
                self.remove_edge(*e)
                
            if self.node[e[0]]['sec']!=self.node[e[1]]['sec']:
                if (not e[0] in self.navpoints_borders) and (not e[1] in self.navpoints_borders):
                    print "I am removing edge", e, "because node", e[0], "is in sector", self.node[e[0]]['sec'],\
                     "but node", e[1], "is in sector", self.node[e[1]]['sec'], "and they are not on the borders."
                    self.remove_edge(*e)  
                       
    # def compute_pairs_based_on_short(self, min_dis):
    #     """
    #     New in 2.9: compute the pairs of airports for which at least Nfp shortest paths have been found.
    #     """
    #     for pair, paths in self.short.items():
    #         if len(paths)<self.Nfp:
    #             print 'I cut paths from pair', pair, 'because there are only', len(paths), 'paths.'
    #             del self.short[pair]
    #     self.fix_airports(list(set([p[0] for p in self.short.keys()] + [p[1] for p in self.short.keys()])), min_dis, pairs=self.short.keys()) 

    def convert_path(self, path):
        """
        New in 2.9: used to convert a path of navigation points into a path of sectors. Replace the method of class Net.
        """
        path_sec=[]
        #print path[0], self.node[path[0]]
        j,sec=0,self.node[path[0]]['sec']
        while j<len(path):
            path_sec.append(sec)
            while j<len(path) and self.node[path[j]]['sec']==sec:
                j+=1
            if j<len(path):
                sec=self.node[path[j]]['sec']
        return path_sec

    def infer_airports_from_sectors(self, airports_sec, min_dis):
        airports_nav=[choice([n for n in self.nodes() if self.node[n]['sec']==sec]) for sec in airports_sec]
        self.fix_airports(airports_nav, min_dis)# add pairs !!!!!!!!!!!!!!!!1

def utility(par, Lsp, t0sp, L, t0):

    (alpha,betha1,betha2)=par
    
    """
    the inputs of this function are all supposed to be NumPy arrays
       
    Call: U=UTILITY(ALPHA,BETHA1,BETHA2,LSP,T0SP,L,T0);
    the function utility.m computes the utility function value, comparing two
    paths on a graph;

    INPUTS

    alpha, betha1, betha2 -> empirically assigned weight parameters,
    ranging from 0 to 1;

    Lsp -> length of the shortest path;

    t0sp -> departure time of the motion along the shortest path;

    L -> length of the path which one wants to compare to the shortest
    one;

    t0 -> depature time of the motion on the path used in the
    coparison with the shortes one;

    OUTPUT

    U -> is the value of the utility function for the given choise of paths;

    """
    
    return np.dot(alpha,L)+np.dot(betha1,np.absolute(t0+L-(t0sp+Lsp)))+np.dot(betha2,np.absolute(t0-t0sp))
   
