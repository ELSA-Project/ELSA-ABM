#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(1, '..')
sys.path.insert(1, '../libs/YenKSP')

import networkx as nx
import sys

from random import sample, uniform, gauss, shuffle, choice
import numpy as np
from numpy.random import lognormal
import matplotlib.delaunay as triang
from math import sqrt, log
import pickle
import os
import matplotlib.pyplot as plt
from utilities import draw_network_map

from string import split
import copy
from os.path import join

from libs.general_tools import counter, silence, build_triangular
from libs.YenKSP.graph import DiGraph
from libs.YenKSP.algorithms import ksp_yen

version='2.9.10'

class NoShortH(Exception):
    """ 
    Exception just to detect problems in shortest path computation.
    """
    pass        

class Network_Manager:
    """
    Class Network_Manager 
    =====================
    The network manager receives flight plans from air companies and tries to
    fill the best ones on the network, by increasing order of cost. If the 
    flight plan does not overreach the capacity of any sector, it is 
    allocated to the network and the sector loads are updated. 
    The network manager can also inform the air companies if a shock occurs on
    the network, i.e if some sectors are shut. It asks for a new bunch of 
    flights plans from the airlines impacted.

    Notes
    -----
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
        Initialize loads for network G. If the NM is new style, it creates for each node a 
        list of length length_day which represents loads. Load is increased by one when a 
        flights crosses the airspace in the corresponding hour. Note that there is no explicit
        dates. If the NM is old style, the load is represented by a list of the 2-lists. The
        is a float representing the time at which the load changes, the second element is the
        load of the sector starting from the first element to the first element of the next 
        2-list.

        The load of normal sectors and airport sectors are tracked separately.

        Examples:
        ---------
        This sequence:
        G.node[n]['load_old'] = [[0, 0], [10., 1], [10.5, 2], [15.5, 1], [20., 0], [10**6,0]]
        Means that there is no flight between 0. and 10., one flight between 10. and 10.5, 
        two between 10.5 and 15.5, one again between 15.5 and 20. and no flight afterwards.

        Parameters
        ----------
        G : Net object (or networkx)
        length_day : int
            Number of hours tracked by the Network Manager.

        Notes
        -----
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

        Parameters
        ----------
        ACs : a list of AirCompany objects 
            The AirCompanys must have their flights and flight plans computed. 

        Returns
        -------
        queue : a list of objects flights
            This is the priority list for the allocation of flights.

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
        well as the first sector overloaded (bottleneck), and the flight plan selected.

        Parameters
        ----------
        G : Net Object
            (Sector) Network on which on which the flights will be allocated. 
            Needs to have an attribute G_nav which is the network of navpoints.
        flight : Flight object
            The flight which has to allocated to the network.
        storymode : bool, optional
            Used to print very descriptive output.

        Notes
        -----
        Changed in 2.2: using intervals.
        Changed in 2.7: sectors of airports are checked independently.
        Changed in 2.9: airports are not checked anymore.
        Changed in 2.9.3: airports are checked again :).

        """

        i=0
        found=False
        while i<len(flight.FPs) and not found:
            fp=flight.FPs[i]
            self.compute_flight_times(G, fp)
            path, times=fp.p, fp.times

            if storymode:
                print "     FP no", i, "tries to be allocated with trajectory (sectors):"
                print fp.p
                print "and crossing times:"
                print fp.times

            #first=1 ###### ATTENTION !!!!!!!!!!!
            first=0 ###### ATTENTION !!!!!!!!!!!
            #last=len(path)-1 ########## ATTENTION !!!!!!!!!!!
            last=len(path) ########## ATTENTION !!!!!!!!!!!
            
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
                self.allocate(G, fp, storymode=storymode, first=first, last=last)
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
        Compute the entry times and exit times of each sector for the trajectory of the given flight
        plan. Store them in the 'times' attribute of the flight plan.

        Parameters
        ----------
        G : hybrid network.
        fp : FlightPlan object.
        
        Notes
        -----
        Changed in 2.8: based on navpoints.

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

        Parameters
        ----------
        G : hybrid network
            Unmodified.
        n : int or string
            sector to check.
        (t1, t2) : (float, float)
            times of entry and exit of the flight in the sector n.
        
        Returns
        -------
        overload : boolean,
            True if the sector would be overloaded with the allocation of this flight plan.

        Notes
        -----
        Changed in 2.9: it does not check anymore if the maximum number of flights have 
        overreached the capacity, but if the total number of flights during an hour 
        (counting those already there at the beginning and those still there at the end)
        is greater than the capacity. 
        Changed in 2.9.8: added the condition h<len(G.node[n]['load']). There is now an 
        absolute reference in time and the weights of the network are in minutes.

        """

        overload = False
        h = 0 
        while float(h) <= t2/60. and h<len(G.node[n]['load']) and not overload:
            # try:
            if h+1 > t1/60. and G.node[n]['load'][h]+1>G.node[n]['capacity']:
                overload = True
            # except:
            #     print "Problem. t1/60., t2/60., h:", t1/60., t2/60., h
            #     print "G.node[n]['load']:", G.node[n]['load']
            #     raise
            h += 1
        return overload

    def overload_sector_peaks(self, G, n, (t1, t2)):
        """
        Old version (2.8.2) of previous method. Based on maximum number of 
        planes in a given sector at any time. See initialize_load method
        for more details.

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
        while float(h) <= t2/60. and h<len(G.node[n]['load']) and not overload:
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
        
    def allocate_hours(self, G, fp, storymode=False, first=0, last=-1):
        """
        Fill the network with the given flight plan. For each sector of the flight plan, 
        add one to the load for each slice of time (one hour slices) in which the flight 
        is present in the sector. 
        The 'first' and 'last' optional arguments are used if the user wants to avoid 
        to load the first and last sectors, as it was in the first versions.

        Parameters
        ----------
        G : hybrid network
            Modified in output with loads updated.
        fp : FlightPlan object
            flight plan to allocate.
        storymode : boolean, optional
            verbosity.
        first : int, optional
            position of the first sector to load in the trajectory. Deprecated.
        last : int, optional
            position of the last sector to load. Deprecated.

        Notes
        -----
        Changed in 2.9: completely changed (count number of flights per hour).
        Changed in 2.9.5: does change the load of the first and last sector.
        Changed in 2.9.8: added condition h<G.node[n]['load']

        """

        if storymode:
            print "NM allocates the flight."
        path, times = fp.p, fp.times
        #for i in range(1,len(path)-1):
        if last==-1: 
            last = len(path)
        for i in range(first, len(path)):
            n = path[i]
            t1, t2 = times[i]/60.,times[i+1]/60.
            h = 0
            while h<t2 and h<len(G.node[n]['load']):
                if h+1>t1:
                    if storymode:
                        print "Load of sector", n, "goes from",  G.node[n]['load'][h], "to", G.node[n]['load'][h]+1, "for interval", h, "--", h+1
                    G.node[n]['load'][h] += 1
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

    def deallocate_hours(self, G, fp, first=0, last=-1):
        """
        Used to deallocate a flight plan not legit anymore, for instance because one 
        sector has been shutdown.
        
        Parameters
        ----------
        G : hybrid network
            Loads are modified as output
        fp : FlightPlan object
            Flight plan to deallocate.
        first : int, optional
            position of the first sector to unload in the trajectory. Must be consistent
            with the allocation's parameters. Deprecated.
        last : int, optional
            position of the last sector to unload. Must be consistent
            with the allocation's parameters. Deprecated.

        Notes
        -----
        New in 2.5
        Changed in 2.9: completely changed, based on hour slices.

        """
        
        path,times=fp.p,fp.times
        if last==-1: 
            last = len(path)
        #for i in range(1, len(path)-1):
        for i in range(first, last):
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

        for n in sectors_to_shut:
            flights_to_reallocate = []
            flights_suppressed = []          
            sec_pairs_to_compute = []
            nav_pairs_to_compute = []
            for f in queue:
                if f.accepted:
                    path_sec = f.fp_selected.p
                    if n in path_sec:
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
            G.compute_all_shortest_paths(Nsp_nav, perform_checks=True, sec_pairs_to_compute=sec_pairs_to_compute, nav_pairs_to_compute=nav_pairs_to_compute)                
            
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

    def M0_to_M1_quick(self, G, queue, N_shocks, tau, Nsp_nav, storymode=False, sectors_to_shut=None):
        """
        Same method than previous one, but closes all sectors at the same time, then recomputes the shortest paths.
        New in 2.9.7.
        """
        if storymode:
            print "N_shocks:", N_shocks
        if sectors_to_shut==None:
            #sectors_to_shut = shock_sectors(G, N_shocks)#sample(G.nodes(), N_shocks)
            sectors_to_shut = sample(G.nodes(), int(N_shocks))
        else:
            sectors_to_shut = [sectors_to_shut]

        if sectors_to_shut!=[]:
            flights_to_reallocate = []
            flights_suppressed = []          
            sec_pairs_to_compute = []
            nav_pairs_to_compute = []
            for f in queue:
                if f.accepted:
                    path_sec = f.fp_selected.p
                    if set(sectors_to_shut).intersection(set(path_sec))!=set([]):
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
            G.compute_all_shortest_paths(Nsp_nav, perform_checks=True, sec_pairs_to_compute=sec_pairs_to_compute, nav_pairs_to_compute=nav_pairs_to_compute, verb = storymode)                
            
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
    def __init__(self, path, time, cost, ac_id, path_nav):
        """
        Parameters
        ----------
        path : list of sectors
        path_nav : lis of navpoints
        time : float
            time of departure in minutes.
        cost : float
            Cost of the nav-path given by the utility function of the company.
        ac_id : int
            Id of the Air Company.
        
        Note
        ----
        No check of consistency between sec-path and nav-path.

        """
        self.p = path # path in sectors
        self.p_nav = path_nav # path in navpoints
        self.t = time # of departure
        self.cost = cost # cost given the utility function
        self.ac_id = ac_id # id of the air company
        self.accepted = True # if the flight plan has been accepted by the NM.
        self.bottleneck = -1 # for post-processing.

    def shift_time(self, shift):
        """
        Shift the time of departure by shift (in minutes).
        """
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
    def __init__(self, Id, source, destination, pref_time, ac_id, par, Nfp):
        """
        Parameters
        ----------
        Id : int
            Identifier of the flight, relative to the AirCompany
        source : int or string
            label of origin node
        destination : int or string
            label of destination node.
        pref_time : float
            Preferred time of departure, in minutes (from time 0, beginning of the day)
        ac_id : int
            Unique Id of the AirCompany.
        par : tuple (float, float, float)
            behavioral parameter of the AirCompany for utility function.
        Nfp : int
            Maximum number of flights plans that the AirCompany is going to submit 
            for this flight. 

        Notes
        -----
        Changed in 2.9.6: added Nfp.

        """
        self.id = Id
        self.source = source
        self.destination = destination
        self.pref_time = pref_time
        self.ac_id = ac_id 
        self.par = par
        self.Nfp = Nfp

    def compute_flightplans(self, tau, G): 
        """
        Compute the flight plans for a given flight, based on Nfp, Nsp_nav and the best paths, and the utility function.
        
        Parameters
        ----------
        tau : float
            The different flight plans of the flight will be shifted by this amount (in minutes).
        G : Net object
            Used to compute cost of paths. Not modified.

        Raises
        ------
        Exception
            If some pairs in the network to not have enough shortest paths
        Exception
            If the list of flight plans in output is smaller than self.Nfp

        Notes
        -----
        Changed in 2.2: tau introduced.
        Changed in 2.8: paths made of navpoints and then converted.
        Changed in 2.9: ai and aj are navpoints, not sectors.
        Changed in 2.9.6: use the convert_path method of the Net object.
        New in 2.9.6: comes from AirCompany object.
        Changed in 2.9.7: ai and aj are source and destination.
        Changed in 2.9.9: resolved a serious bug of references on paths.
        
        """

        ai, aj = self.source, self.destination
        t0sp = self.pref_time

        # Check that all origin-destination pairs in the network
        # have a number of shortest paths exactly equal to the number 
        # of flight plans to be submitted.
        try:
            for k, v in G.G_nav.short.items():
                assert len(v)==G.Nfp
        except:
            raise Exception("OD Pair", k, "have", len(v), "shortest paths whereas", G.Nfp, "were required.")

        # For each shortest path, compute the path in sectors and the total weight of the nav-path
        SP = [(p, G.convert_path(p), G.G_nav.weight_path(p)) for p in G.G_nav.short[(ai,aj)]]

        # Compute the cost of the worst path (with desired time).
        uworst = utility(self.par, SP[0][-1], t0sp, SP[-1][-1], t0sp)
                
        # Compute the cost of all paths which have a cost smaller than uworst 
        u = [[(cp, t0sp + i*tau, utility(self.par, SP[0][-1], t0sp, c, t0sp + i*tau),p) for p,cp,c in SP] for i in range(self.Nfp)\
            if utility(self.par,SP[0][-1], t0sp, SP[0][-1],t0sp + i*tau)<=uworst]

        # Select the Nfp flight plans less costly, ordered by increasing cost.
        fp = [FlightPlan(a[0][:],a[1],a[2],self.id,a[3][:]) for a in sorted([item for sublist in u for item in sublist], key=lambda a: a[2])[:self.Nfp]]

        if len(fp)!=self.Nfp:
            raise Exception('Problem: there are', len(fp), 'flights plans whereas there should be', self.Nfp)
    
        if not G.G_nav.weighted:
            # Shuffle the flight plans with equal utility function
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
        Used for post-processing.
        Used to remember the flight plans which were overloading the network, 
        as well as the first sector to be overloaded on the trajectories.
        """
        try:
            self.flag_first = [fp.accepted for fp in self.FPs].index(True)
        except ValueError:
            self.flag_first = len(self.FPs)
            
        self.overloadedFPs = [self.FPs[n].p for n in range(0,self.flag_first)]
        self.bottlenecks = [fp.bottleneck for fp in self.FPs if fp.bottleneck!=-1]

    def shift_desired_time(self, shift):
        """
        Shift the desired time of all flight plans of the flight.
        Parameters
        ----------
        shift : float
            Amount of time in minutes.
        """
        shift = int(shift)
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
        """
        Initialize the AirCompany.

        Parameters
        ----------
        Id: integer
            unique identifier of the company.
        Nfp : integer
            Number of flights plans that flights will submit
        na: integer
            Number of flights per destination-origin operated by the Air company.
            Right now the Model supports only na=1
        pairs : list of tuple with origin-destination
            departure/arrivale point will be drawn from them if not specified in fill_FPs
        par : tuple (float, float, float)
            Parameters for the utility function

        """

        try:
            assert na==1
        except AssertionError:
            raise Exception("na!=1 is not supported by the model.")

        self.Nfp=Nfp
        self.par=par
        self.pairs=pairs 
        self.na=na
        self.id=Id
        
    def fill_FPs(self, t0spV, tau, G, pairs=[]):
        """
        Fill na flights with Nfp flight plans each, between airports given by pairs.

        Parameters
        ----------
        t0spV : iterable with floats
            Desired times of departure for each origin-destination pair.
        tau : float
            Amount of time (in seconds) used to shift the flights plans
        G : Net Object
            (Sector) Network on which on which the flights will be allocated. 
            Needs to have an attribute G_nav which is the network of navpoints.
        pairs : list of tuples (int, int), optional
            If given, it is used as the list origin destination for the flights. 
            Otherwise self.pairs is used.

        Notes
        -----

        New in 2.9.5: can specify a pair of airports.
        Changed in 2.9.6: the flight computes the flight plans itself.

        """

        if pairs==[]:
            assigned_airports = sample(self.pairs, self.na) 
        else:
            assigned_airports = pairs

        self.flights = []
        i = 0
        for (ai,aj) in assigned_airports:
            self.flights.append(Flight(i, ai, aj, t0spV[i], self.id, self.par, self.Nfp))
            self.flights[-1].compute_flightplans(tau, G)
            i += 1
        
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
        
    def import_from(self, G, numberize=False, verb=False):
        """
        Used to import the data of an already existing graph (networkx) in a Net obect.
        Weights are conserved. 

        Parameters
        ----------
        G : a networkx object
            all keys attached to nodes will be preserved. Network needs to be completely weighted, 
            or none at all.
        numberize : boolean, optional
            if True, nodes of G will not be used as is, but an index will be generated instead.
            The real name in stored in the 'name' key of the node. A dictionnary idx_nodes is 
            also attached to the network for easy (reverse) mapping.
        verb : boolean, optional
            verbosity.

        Notes
        -----
        Changed in 2.9: included case where G is empty.
        TODO: preserve attributes of edges too.

        """
        
        if verb:
            print 'Importing network...'

        if len(G.nodes())!=0:
            if not numberize:
                self.add_nodes_from(G.nodes(data=True))
                if len(G.edges())>0:
                    e1, e2 = G.edges()[0]
                    if 'weight' in G[e1][e2].keys():
                        self.add_weighted_edges_from([(e[0],e[1], G[e[0]][e[1]]['weight']) for e in G.edges()])
                    else:
                        self.add_weighted_edges_from([(e[0],e[1], 1.) for e in G.edges()])
            else:
                self.idx_nodes={s:i for i,s in enumerate(G.nodes())}
                for n in G.nodes():
                    self.add_node(self.idx_nodes[n], name=n, **G.node[n])

                e1, e2 = G.edges()[0]
                if len(G.edges())>0:
                    if 'weight' in G[e1][e2].keys():
                        for e in G.edges():
                            e1 = self.idx_nodes[e[0]]
                            e2 = self.idx_nodes[e[1]]
                            self.add_edge(e1, e2, weight=G[e1][e2]['weight'])
                    else:
                        for e in G.edges():
                            self.add_edge(self.idx_nodes[e[0]], self.idx_nodes[e[1]], weight=1.)

            if len(self.edges())>0:
                e1 = self.edges()[0]
                e2 = self.edges()[1]
                self.weighted = not (self[e1[0]][e1[1]]['weight']==self[e2[0]][e2[1]]['weight']==1.)
            else:
                print "Network has no edge!"
                self.weighted = False

            if verb:
                if self.weighted:
                    print 'Network was found weighted'
                else:
                    print 'Network was found NOT weighted'
                    if len(self.edges())>0:
                        print 'Example:', self[e1[0]][e1[1]]['weight']     
        else:
            print 'Network was found empty!'   
            
    def build_nodes(self, N, prelist=[], put_nodes_at_corners=False, small=1.e-5):
        """
        Add N nodes to the network, with coordinates taken uniformly in a square. 
        Alternatively, prelist gives the list of coordinates.

        Parameters
        ----------
        N : int
            Number of nodes to produce
        prelist : list of 2-tuples, optional
            Coordinates of nodes to add to the nodes previously generated
        put_nodes_at_corners : boolean, optional
            if True, put a nodes at each corner for the 1x1 square
        small : float, optional
            Used to be sure that the nodes in the corner are strictly within the
            square

        Notes
        -----
        Remark: the network should not have any nodes yet.
        New in 2.8.2

        """
        for i in range(N):
            self.add_node(i,coord=[uniform(-1.,1.),uniform(-1.,1.)])  
        for j,cc in enumerate(prelist):
            self.add_node(N+j,coord=cc)
        if put_nodes_at_corners:
            self.add_node(N+len(prelist), coord=[1.-small, 1.-small])
            self.add_node(N+len(prelist)+1, coord=[-1.+small, 1.-small])
            self.add_node(N+len(prelist)+2, coord=[-1.+small, -1.+small])
            self.add_node(N+len(prelist)+3, coord=[1.-small, -1.+small])

    def build_net(self, Gtype='D', mean_degree=6):
        """
        Build edges, based on Delaunay triangulation or Erdos-Renyi graph. 
        No weight is computed at this point.

        Parameters
        ----------
        Gtype : string, 'D' or 'E'
            type of graph to be generated. 'D' generates a dealunay triangulation 
            which in particular is planar and has the highest degree possible for a planar 
            graph. 'E' generates an Erdos-Renyi graph.
        mean_degree : float
            mean degree for the Erdos-Renyi graph.

        Notes
        -----
        Changed in 2.9.10: removed argument N.

        """

        if Gtype=='D':
            x,y =  np.array([self.node[n]['coord'][0] for n in self.nodes()]),np.array([self.node[n]['coord'][1] for n in self.nodes()])   
            cens, edg, tri, neig = triang.delaunay(x,y)
            for p in tri:
                self.add_edge(p[0],p[1])
                self.add_edge(p[1],p[2])
                self.add_edge(p[2],p[0])
        elif Gtype=='E':  
            N = len(self.nodes())
            prob = mean_degree/float(N-1) # <k> = (N-1)p - 6 is the mean degree in Delaunay triangulation.
            for n in self.nodes():
                for m in self.nodes():
                    if n>m:
                        if np.random.rand()<=prob:
                            self.add_edge(n,m)
            
    def build(self, N, Gtype='D', mean_degree=6, prelist=[], put_nodes_at_corners=False):
        """
        Build a graph of the given type, nodes, edges.
        Essentially gathers build_nodes and build_net methods and add the possibility 
        of building a simple triangular network (not sure it is up to date though).

        Parameters
        ----------
        N : int
            Number of nodes to produce.
        Gtype : string, 'D', 'E', or 'T'
            type of graph to be generated. 'D' generates a dealunay triangulation 
            which in particular is planar and has the highest degree possible for a planar 
            graph. 'E' generates an Erdos-Renyi graph. 'T' builds a triangular network
        mean_degree : float
            mean degree for the Erdos-Renyi graph.
        prelist : list of 2-tuples, optional
            Coordinates of nodes to add to the nodes previously generated
        put_nodes_at_corners : boolean, optional
            if True, put a nodes at each corner for the 1x1 square

        """

        print 'Building random network of type', Gtype
        
        if Gtype=='D' or Gtype=='E':
            self.build_nodes(N, prelist=prelist, put_nodes_at_corners=put_nodes_at_corners)
            self.build_net(Gtype=Gtype, mean_degree=mean_degree)  
        elif Gtype=='T':
            xAxesNodes = np.sqrt(N/float(1.4))
            self = build_triangular(xAxesNodes)  
        
    def generate_weights(self, typ='gauss', par=[1.,0.01]):
        """
        Generates weights with a gaussian distribution or given by the euclidean distance
        between nodes, tuned so that the average matches the one given as argument.

        Parameters
        ----------
        input typ: str
            should be 'gauss' or 'coords'. The first produces gaussian weights with mean 
            given by the first element of par and the deviation given by the second element
            of par. 'coords' computes the euclideian distance between nodes (based on key 
            coord) and adjust it so the average weight over all edges matches the float given
            by par.
        input par: list or float
            If typ is 'gauss', gives the mean and deviation. Otherwise, should be a float giving 
            the average weight.

        """

        self.typ_weights, self.par_weights=typ, par
        if typ=='gauss':
            mu = par[0]
            sigma = par[1]
            for e in self.edges():
                self[e[0]][e[1]]['weight'] = max(gauss(mu, sigma),0.00001)
        elif typ=='coords':
            for e in self.edges():
                #self[e[0]][e[1]]['weight']=sqrt((self.node[e[0]]['coord'][0] - self.node[e[1]]['coord'][0])**2 +(self.node[e[0]]['coord'][1] - self.node[e[1]]['coord'][1])**2)
                self[e[0]][e[1]]['weight'] = np.linalg.norm(np.array(self.node[e[0]]['coord']) - np.array(self.node[e[1]]['coord']))
            avg_weight = np.mean([self[e[0]][e[1]]['weight'] for e in self.edges()])
            for e in self.edges():
                self[e[0]][e[1]]['weight'] = par*self[e[0]][e[1]]['weight']/avg_weight
        self.typ_weights = typ
        self.weighted = True

    def fix_weights(self, weights, typ='data'):
        """
        Fix the weights from a dictionnary given as argument.

        Parameters
        ----------
        input weights : dictionnary 
            of dictionnaries whose keys are the edges and values are the weights.
        typ : string
            only used to track where the weights come from

        """

        for e, w in weights.items():
            self[e[0]][e[1]]['weight'] = w
        self.typ_weights = typ
        self.weighted = True
            
    def generate_capacities(self, typ='constant', C=5, par=[1]):
        """
        Generates capacities with different distributions.
        If typ is 'constant', all nodes have the same capacity, given by C.
        If typ is 'gauss', the capacities are taken from a normal distribution with mean C
        and standard deviation par[0]
        If typ is 'uniform', the capacities are taken from a uniform distribution, with 
        bounds C-par[0]/2.,C+par[0]/2.
        If typ is 'areas', the capacities are proportional to the square root of the area of 
        the sector, with the proportionality factor set so at to have a mean close to C. 
        This requires that each node has a key 'area'.
        If typ is 'lognormal', the capacities are taken from a lognormal distribution.

        Capacities are integers, minimum 1.

        Parameters
        ----------
        typ : string
            type of distribution, see description.
        C : int
            main parameter of distribution, see description.
        par : list of int or float
            other parameters, see description.

        Notes
        ----- 
        New in 2.7: added lognormal and areas.
        Changed in 2.9.8: removed "manual"

        """

        assert typ in ['constant', 'gauss', 'uniform', 'time', 'lognormal', 'areas']
        self.C, self.typ_capacities, self.par_capacities = C, typ, par
        if typ=='constant':
            for n in self.nodes():
                self.node[n]['capacity'] = C
        elif typ=='gauss':
            for n in self.nodes():
                self.node[n]['capacity'] = max(1, int(gauss(C,par[0])))
        elif typ=='uniform':
            for n in self.nodes():
                self.node[n]['capacity'] = max(1, int(uniform(C-par[0]/2.,C+par[0]/2.)))
        elif typ=='lognormal':
            for n in self.nodes():
                self.node[n]['capacity'] = max(1, int(lognormal(log(C),par[0])))
        elif typ=='areas':
            if par[0]=='sqrt':
                area_avg = np.mean([sqrt(self.node[n]['area']) for n in self.nodes()])
                alpha = C/area_avg
                for n in self.nodes():
                    self.node[n]['capacity'] = max(1, int(alpha*sqrt(self.node[n]['area'])))
        
    def fix_capacities(self, capacities, typ='exterior'):
        """
        Fix the capacities given with the dictionnary of capacities.
        Checks if every sector has a capacity.

        Parameters
        ----------
        capacities : dictionnary
            keys are nodes and values are capacities. Capacities are
            converted to integers.
        typ : string
            Used to keep track of how the capacities have been generated.

        Raises 
        ------
        Exception
            if at least one node does not have a capacity.

        Notes
        -----
        New in 2.9.

        """

        self.typ_capacities = typ
        for n,v in capacities.items():
            self.node[n]['capacity'] = int(v)

        try:
            for n in self.nodes():
                assert self.node[n].has_key('capacity')
        except AssertionError:
            raise Exception("One sector at least does not have a capacity!")

    def generate_airports(self, nairports, min_dis, C_airport=10):
        """
        Generates nairports airports. Builds the accessible pairs of airports for this network
        with a minimum distance min_dis. Sets the capacity of the airport with the argument C_airport.

        Parameters
        ----------
        nairports : float
            Number of airports to create.
        min_dis : int
            Topological distance (number of ndoes between origin and destination, without the ends) 
            under which a connection cannot be created.
        C_airport : int, optional
            Capacity of the sectors which are airports. They are used only by flights which are
            departing or lending in this area. It is different from the standard capacity key
            which is used for flights crossing the area, which is set to 10000.

        Notes
        -----
        The capacities of airports is not trivial, see the builder in prepare_navpoint_network.

        """

        self.airports = sample(self.nodes(),nairports)
        self.short = {(ai,aj):[] for ai in self.airports for aj in self.airports if len(nx.shortest_path(self, ai, aj))-2>=min_dis}
        
        for a in self.airports:
            self.node[a]['capacity']=100000                 # TODO: check this.
            self.node[a]['capacity_airport'] = C_airport

    def fix_airports(self, *args, **kwargs):
        """
        Used to reset the airports and then add the new airports.
        """
        if hasattr(self, "airports"):
            self.airports = []
            self.short = {}

        self.add_airports(*args, **kwargs)

    def add_airports(self, airports, min_dis, pairs=[], C_airport=10, singletons=False):
        """
        Add airports given by user. The pairs can be given also by the user, 
        or generated automatically, with minimum distance min_dis.

        Parameters
        ----------
        min_dis : int 
            minimum distance -- in nodes, excluding the airports -- betwen a pair of airports.
        pairs : list of 2-tuples, optional
            Pairs of nodes for connections. If [], all possible pairs between airports are computed 
            (given min_dis)
        C_airport : int, optional
            Capacity of the sectors which are airports. They are used only by flights which are
            departing or lending in this area. It is different from the standard capacity key
            which is used for flights crossing the area, which is set to 10000.
        singletons : boolean, optional
            If True, pairs in which the source is identical to the destination are authorized 
            (but min_dis has to be smaller or equal to 2.)
        
        Notes
        -----
        Changed in 2.9.8: changed name to add_airports. Now the airports are added
        to the existing airports instead of overwriting the list.

        """
        if not hasattr(self, "airports"):
            self.airports = airports
        else:
            self.airports = np.array(list(set(list(self.airports) + list(airports))))
            
        if not hasattr(self, "short"):
            self.short = {}

        if pairs==[]:
            for ai in self.airports:
                for aj in self.airports:
                    if len(nx.shortest_path(self, ai, aj))-2>=min_dis and ((not singletons and ai!=aj) or singletons):
                        if not self.short.has_key((ai,aj)):
                            self.short[(ai, aj)] = []
        else:
            for (ai,aj) in pairs:
                 if ((not singletons and ai!=aj) or singletons):
                    if not self.short.has_key((ai,aj)):
                        self.short[(ai, aj)] = []

        for a in airports:
            #self.node[a]['capacity']=100000                # TODO: check this.
            self.node[a]['capacity_airport'] = C_airport
    
    def infer_airports_from_navpoints(self, C_airport, singletons=False):
        """
        Detects all sectors having at least one navpoint being a source or a destination. 
        Mark them as airports, with a given capacity.
        Should only be used by hybrid networks (with attribute G_nav).

        Parameters
        ----------
        C_airport : int
            Capacity of the sectors which are airports. They are used only by flights which are
            departing or lending in this area. It is different from the standard capacity key
            which is used for flights crossing the area.
        singletons : boolean, optional
            If True, pairs in which the source is identical to the destination are authorized 
            (but min_dis has to be smaller or equal to 2.)

        """

        assert hasattr(self.G, G_nav)
        pairs = []
        for (p1,p2) in self.G_nav.short.keys():
            s1 = self.G_nav.node[p1]['sec']
            s2 = self.G_nav.node[p2]['sec']
            pairs.append((s1,s2))
        pairs = list(set(pairs))
        self.fix_airports(np.unique([self.G_nav.node[n]['sec'] for n in self.G_nav.airports]), -10, 
                C_airport=C_airport, pairs=pairs, singletons=singletons)

    def infer_airports_from_short_list(self):
        """
        Compute the list of airports from all possible pairs of source/destination.
        """
        self.airports = list(set([a for b in self.short.keys() for a in b]))

    def build_H(self): 
        """
        Build the DiGraph object used in the ksp_yen algorithm.
        """
        self.H = DiGraph()
        self.H._data = {}
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
        
    def compute_shortest_paths(self, Nfp, repetitions=True, use_sector_path=False, old=False, pairs=[], 
        verb=1, delete_pairs=True):
        """
        Pre-Build Nfp weighted shortest paths between each pair of airports. 
        If the function do not find enough paths, the corresponding source/destination pair is deleted.
        
        Parameters
        ----------
        Nfp : int
            Number of shortest paths to compute between each pair of airports.
        repetitions : boolean, optional
            If True, a path can have a given node twice or more. Otherwise, the function makes
            several iterations, considering longer and longer paths until it finds a path which doesn't have any 
            repeated sector.
        use_sector_path : boolean, optional
            If True, the nav-paths are generated so that the sector paths do not have repeated sectors. Does not 
            have any effect if repetitions is True.
        old : boolean, optional
            Should always be false. Used to compare with previous algorithm of YenKSP.
        pairs : list of 2-tuples, optional
            list of origin-destination for which the shortest paths will be computed. If [], all shortest paths
            will be computed.
        verb : int, optional
            verbosity
        delete_pairs : boolean, optional
            if True, all pairs for which not enough shortest paths have been found are deleted.

        Notes
        -----
        Changed in 2.9: added singletons option. Added repetitions options to avoid repeated sectors in paths.
        Changed in 2.9.4: added procedure to have always 10 distinct paths (in sectors).
        Changed in 2.9.7: modified the location of the not enough_path loop to speed up the process. Added
        pairs_to_compute, so that it does not necesseraly recompute every shortest paths.
        Changed in 2.9.8: Added n_tries in case of use_sector_path.
        Changed in 2.9.10: added option to remove pairs which do not have enough paths. If disabled, the last paths 
        is directed until Nfp is reached.

        """

        if use_sector_path:
            try:
                assert self.type == 'sec'
            except:
                Exception('use_sector_path should only be used with hybrid networks.')

        if pairs==[]:
            pairs = self.short.keys()[:]

        assert not old
        
        deleted_pairs = []
        if repetitions:
            for (a,b) in pairs:
                enough_paths = False
                Nfp_init = Nfp
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

                            for path in duplicates:
                                paths.remove(path)

                            try:
                                assert len(paths)==Nfp and len(duplicates)==previous_duplicates
                                enough_paths=True
                                paths = [list(vvv) for vvv in set([tuple(vv) for vv in paths])][:Nfp_init]
                                if len(paths) < Nfp_init:
                                    enough_paths = False
                                    print 'Not enough paths, doing another round (' + str(Nfp +1 - Nfp_init), 'additional path(s)).'
                                Nfp += 1
                            except AssertionError:
                                #print 'a:', a, 'b:', b, 'len(self.short[(a,b)]):', len(self.short[(a,b)])
                                print "kspyen can't find enough paths (only " + str(len(paths)) + ')', "for the pair", a, b,
                                #print 'Number of duplicates:', len(duplicates)
                                #print 'Number of duplicates:', len(duplicates)
                                #print 'Number of paths with duplicates:', len(paths_init)
                                if delete_pairs:
                                    print "I delete this pair."
                                    deleted_pairs.append((a,b))
                                    del self.short[(a,b)]
                                    break
                                else:
                                    print

                        Nfp = Nfp_init
                        if self.short.has_key((a,b)):
                            self.short[(a,b)] = paths[:]      

                        if not delete_pairs:
                            if  len(self.short[(a,b)])<Nfp:
                                print  "Pair", (a,b), "do not have enough path, I duplicate the last one..."
                            while len(self.short[(a,b)])<Nfp:
                                self.short[(a,b)].append(self.short[(a,b)][-1])
                            assert len(self.short[(a,b)])==Nfp
                    else:
                        self.short[(a,b)] = [[a] for i in range(Nfp)]
            else:
                for it, (a,b) in enumerate(pairs):
                    #if verb:
                    #    counter(it, len(pairs), message='Computing shortest paths...')
                    if a!=b:
                        enough_paths=False
                        Nfp_init=Nfp
                        i=0
                        while not enough_paths:
                            enough_paths=True
                            paths_nav = self.kshortestPath(a, b, Nfp, old=old)
                            paths = [self.convert_path(p) for p in paths_nav]
                            previous_duplicates=1
                            duplicates=[]
                            duplicates_nav=[]
                            n_tries = 0
                            while len(duplicates)!=previous_duplicates and n_tries<10:
                                if n_tries!=0: print "I don't have enough paths, I make another turn. n_tries=", n_tries
                                previous_duplicates=len(duplicates)
                                duplicates=[]
                                duplicates_nav=[]
                                for j,sp in enumerate(paths): # Check the repetitions on the sec-path
                                    if len(np.unique(sp))<len(sp):
                                        duplicates.append(sp)
                                        duplicates_nav.append(paths_nav[j])
                                
                                if len(duplicates)!=previous_duplicates:
                                    paths_nav=self.kshortestPath(a, b, Nfp+len(duplicates))
                                    paths = [self.convert_path(p) for p in paths_nav]

                                n_tries += 1

                            for path in duplicates_nav:
                                paths_nav.remove(path)

                            try:
                                assert len(paths_nav)==Nfp and len(duplicates)==previous_duplicates
                                enough_paths=True
                                paths_nav = [list(vvv) for vvv in set([tuple(vv) for vv in paths_nav])][:Nfp_init]
                                if len(paths_nav) < Nfp_init:
                                    enough_paths = False
                                    print 'Not enough paths, doing another round (' + str(Nfp - Nfp_init), 'additional paths).'
                                Nfp += 1
                            except AssertionError:
                                #print 'a:', a, 'b:', b, 'len(self.short[(a,b)]):', len(self.short[(a,b)])
                                print "kspyen can't find enough paths (only " + str(len(paths)) + ')', "for the pair", a, b,
                                #print 'Number of duplicates:', len(duplicates)
                                if delete_pairs:
                                    print "I delete this pair."
                                    deleted_pairs.append((a,b))
                                    del self.short[(a,b)]
                                    raise
                                else:
                                    print
                                
                            i+=1
                        Nfp = Nfp_init
                        if self.short.has_key((a,b)):
                            #assert len(paths_nav) == Nfp_init
                            self.short[(a,b)] = paths_nav[:]     

                        if not delete_pairs: 
                            if  len(self.short[(a,b)])<Nfp:
                                print  "Pair", (a,b), "do not have enough path, I duplicate the last one..."
                            while len(self.short[(a,b)])<Nfp:
                                self.short[(a,b)].append(self.short[(a,b)][-1])  


        return list(set(deleted_pairs))

    def compute_sp_restricted(self, Nfp, silent=True, pairs=[], delete_pairs=True):
        """
        Computes the k shortest nav-paths restricted to shortest sec-paths. Note that this method
        is used by the hybrid network, not by the navpoint network directly.

        Parameters
        ----------
        Nfp : int
            Number of shortest nav-paths per shortest sec-path.
        silent : boolean, optional
            verbosity
        pairs : list of 2-tuples, optional
            list of origin-destination for which the shortest paths will be computed. If [], all shortest paths
            will be computed.
        delete_pairs : boolean, optional
            if True, all pairs for which not enough shortest paths have been found are deleted.

        Notes
        -----
        Changed in 2.9.1: save all Nfp shortest paths of navpoints for each paths of sectors (Nfp also).
        Changed in 2.9.7: added pairs kwarg in order to compute less paths.
        Changed in 2.9.10: added option to remove pairs which do not have enough paths. If disabled, the last paths 
        is duplicated until Nfp is reached.
        Note: this method should NOT be used by a navpoint network.
        Note: this method should not touch self.G_nav.short because the actual shortest paths for 
        the navpoints network are chosen in method choose_short.
        Note: this is black magic. Don't make any change unless you know what you are doing (and you probably don't).

        """

        if not hasattr(self, 'short_nav'):
            self.short_nav={}

        if pairs ==[]:
            pairs = self.G_nav.short.keys()[:]
        else:
            pairs = [p for p in pairs if p in self.G_nav.short.keys()]

        for idx,(p0,p1) in enumerate(pairs):
            #print "Computing restricted paths for pair", (p0, p1)
            #counter(idx, len(pairs), message='Computing shortest paths (navpoints)...')
            s0,s1 = self.G_nav.node[p0]['sec'], self.G_nav.node[p1]['sec']
            self.G_nav.short[(p0,p1)] = []
            self.short_nav[(p0,p1)] = {}
            try:
                assert len(self.short[(s0,s1)]) == self.Nfp
            except:
                print "s0, s1:", s0, s1
                print "len(self.short[(s0,s1)]), self.Nfp", len(self.short[(s0,s1)]), self.Nfp
                raise

            try:
                assert len(self.short[(s0, s1)]) == self.Nfp
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
                                # Cette partie est psychédélique... Ces deux boucles semblent etre des checks seulement.
                                # for i in range(len(sp)-1):
                                #     found=False
                                #     ss1, ss2=sp[i], sp[i+1]
                                #     for n1, n2 in self.G_nav.edges():
                                #         if (self.G_nav.node[n1]['sec']==ss1 and self.G_nav.node[n2]['sec']==ss2) or (self.G_nav.node[n1]['sec']==ss2 and self.G_nav.node[n2]['sec']==ss1):
                                #             found=True
                                            # if ss1==0 and ss2==31:
                                            #     print n1, n2, self.G_nav.node[n1]['sec'], self.G_nav.node[n2]['sec'], H_nav.has_node(n1), H_nav.has_node(n1), H_nav.has_edge(n1,n2)
                                    # if found==False:
                                    #     print 'Problem: sectors', ss1, 'and', ss2, 'are not adjacent in terms of navpoints.'
                                    # else:
                                    #     print 'sectors', ss1 , 'and', ss2, 'are adjacent.'
                                # for i in range(len(sp)-1):
                                #     found=False
                                #     ss1, ss2=sp[i], sp[i+1]
                                #     for n1, n2 in H_nav.edges():
                                #         if (self.G_nav.node[n1]['sec']==ss1 and self.G_nav.node[n2]['sec']==ss2) or (self.G_nav.node[n1]['sec']==ss2 and self.G_nav.node[n2]['sec']==ss1):
                                #             found=True
                                    # if found==False:
                                    #     print 'Problem: sectors', ss1, 'and', ss2, 'are not adjacent in terms of navpoints (H).'
                                    # else:
                                    #     print 'sectors', ss1 , 'and', ss2, 'are adjacent (H).'

                                # Compute shortest paths on restrictied network.
                                H_nav.fix_airports([p0,p1], 0.)
                                H_nav.build_H()
                                try:
                                    H_nav.compute_shortest_paths(Nfp, repetitions=False, use_sector_path=True, old=False, verb = 0, delete_pairs=delete_pairs)
                                except AssertionError:
                                    raise NoShortH('')
                                # try:
                                #     for v in H_nav.short.values():
                                #         assert len(v) == Nfp
                                # except AssertionError:
                                #     print len(v), Nfp
                                #     raise
                                # except:
                                #     raise

                                # for k, p in enumerate(H_nav.short[(p0,p1)]):
                                #     if self.convert_path(p)!=sp:
                                #         print 'Alert: discrepancy between theoretical path of sectors and final one!'
                                #         print 'Path number', k
                                #         print 'Theoretical one:', sp, self.weight_path(sp)
                                #         print 'Actual one:',  self.convert_path(p), self.weight_path(self.convert_path(p))        
                                #         raise Exception("Problem")                        


                                shorts=[p for p in  H_nav.short[(p0,p1)] if self.convert_path(p)==sp]
                                assert len(shorts)==self.Nfp
                                #self.G_nav.short[(p0,p1)] = self.G_nav.short.get((p0,p1),[]) + shorts
                                # This list stores all nav-shortest paths, organized by sec-shortest paths.
                                # So it should have Nfp entries of length Nfp.
                                self.short_nav[(p0,p1)][tuple(sp)] = shorts 
                                assert len(self.short_nav[(p0,p1)][tuple(sp)])==self.Nfp
                            except nx.NetworkXNoPath:
                                print 'No restricted shortest path between' ,p0, 'and', p1
                                cc=nx.connected_components(H_nav)
                                print 'Composition of connected components (sectors):'
                                for c in cc:
                                    print np.unique([self.G_nav.node[n]['sec'] for n in c])
                                raise
                        else:
                            print 'The subgraph was empty, I carry on.'

                #try:
                #    assert len(self.short_nav[(p0, p1)]) == self.Nfp
                #except:    
                #    print "len(self.short_nav[(p0, p1)]), self.Nfp", len(self.short_nav[(p0, p1)]), self.Nfp
                #    raise
            except AssertionError:
                print 'There is a problem with the number of shortest paths.'
                raise
            except NoShortH:
                #self.G_nav.short[(p0,p1)]
                print "I can't find enough paths for this pair", 
                if delete_pairs:
                    print "so I delete it."
                    del self.G_nav.short[(p0,p1)]
                    del self.short_nav[(p0,p1)]     
                else:
                    print


            # try:
            #     assert len(self.short_nav[(p0,p1)]) == self.Nfp
            # except:
            #     print len(self.short_nav[(p0,p1)])
            #     raise
            # if self.G_nav.short.has_key((p0,p1)):
            #     self.G_nav.short[(p0,p1)] = sorted(list(set([tuple(o) for o in G.G_nav.short[(p0,p1)]])), key= lambda p: G.G_nav.weight_path(p))[:Nfp]
 
        #for p in pairs:
        #    if self.G_nav.short.has_key(p) and self.G_nav.short[p]==[]:
        #        del self.G_nav.short[p]
        #        print 'I deleted pair', p ,'because no path was computed.'
            #elif self.G_nav.short[p] < Nfp and not delete_pairs:
            #    print  "Pair", p, "do not have enough path, I duplicate the last one."
            #    while len(self.G_nav.short[p])<Nfp:
            #        self.G_nav.short[p].append(self.short[p][-1])  

        #for p in self.G_nav.short.keys():
        #    if self.G_nav.short[p] < Nfp and not delete_pairs:
        
                # print  "Pair", p, "do not have enough path, I duplicate the last one."
                # while len(self.G_nav.short[p])<Nfp:
                #     self.G_nav.short[p].append(self.short[p][-1])
            # try:
            #     assert len(self.G_nav.short[p])==Nfp    
            # except:
            #     print len(self.G_nav.short[p]), Nfp
            #     raise     
                   
    def compute_all_shortest_paths(self, Nsp_nav, perform_checks = False, sec_pairs_to_compute = [], nav_pairs_to_compute = [], verb = True):
        """
        Gather several methods to ensure a consistency between navpoints and sectors.
        Obsolete ? TODO
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
        Return the k weighted shortest paths on the network thanks to YenKSP algorithm. Uses the DiGraph,
        computed by build_H.

        Parameters
        ----------
        i : int
            origin
        j : int
            destination
        k : int
            Number of shortest paths to compute
        old : boolean, optional
            Should set to False. Used to compare with previous algorithm YenKSP

        Raises
        ------
        Exception
            if old is set to True. 

        Notes
        -----
        TODO : remove old.

        """
        if not old:
            spath = [a['path'] for a in ksp_yen(self.H, i, j, k)]
        else:
            raise Exception("You asked for old ksp_yen which is not avalaible anymore")

        spath = sorted(spath, key=lambda a:self.weight_path(a))
        
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
        Used to choose Nfp shortest paths such as there are Nsp_nav nav-shortest paths for each sector-shortest path.
        Needs to have all sec and nav paths computed. Usable only by hybrid network.
        
        Parameters
        ----------
        Nsp_nav : int
            Number of nav-paths per sec-path.
        pairs : list of 2-tuples, optional
            list of origin-destination (navs) for which the shortest paths will be computed. If [], all shortest paths
            will be computed. 

        Notes
        -----
        New in 2.9
        Changed in 2.9.10: repetitive shortest paths in sector supported.
        
        """
        
        if pairs == []:
            pairs = self.G_nav.short.keys()
        else:
            pairs = [p for p in pairs if p in self.G_nav.short.keys()]
        self.Nsp_nav = Nsp_nav
        assert self.type=='sec'
        for (p0,p1) in pairs:
            self.G_nav.short[(p0,p1)] = sorted([path for paths in self.short_nav[(p0,p1)].values() for path in paths[:Nsp_nav]],\
                key= lambda p: self.G_nav.weight_path(p))[:self.Nfp]

            while len(self.G_nav.short[(p0,p1)])<self.Nfp:
                print "There is not enough shortest paths for", (p0, p1), "so I duplicate the last one."
                self.G_nav.short[(p0,p1)].append(self.G_nav.short[(p0,p1)][-1])

    def convert_path(self, path):
        """
        New in 2.8: used to convert a path of navigation points into a path of sectors.
        """
        path_sec = []
        j,sec = 0, self.G_nav.node[path[0]]['sec']
        while j<len(path):
            path_sec.append(sec)
            while j<len(path) and self.G_nav.node[path[j]]['sec']==sec:
                j+=1
            if j<len(path):
                sec = self.G_nav.node[path[j]]['sec']
        return path_sec
            
    def basic_statistics(self, rep='.'):
        """
        Computes basic stats on degree, weights and capacities. 

        Parameters
        ----------
        rep : string
            directory in which the stats are saved.

        Notes
        -----
        TODO: expand this.

        """
        os.system('mkdir -p ' + rep)
        with open(join(rep, 'basic_stats_net.txt'),'w') as f:
            print >>f, 'Mean/std degree:', np.mean([self.degree(n) for n in self.nodes()]), np.std([self.degree(n) for n in self.nodes()])
            print >>f, 'Mean/std weight:', np.mean([self[e[0]][e[1]]['weight'] for e in self.edges()]), np.std([self[e[0]][e[1]]['weight'] for e in self.edges()])
            print >>f, 'Mean/std capacity:', np.mean([self.node[n]['capacity'] for n in self.nodes() if not n in self.airports]),\
                np.std([self.node[n]['capacity'] for n in self.nodes() if not n in self.airports])
        
    def shut_sector(self, n):
        """
        Shut the sector, i.e. remove the sec-node, all corresponding nav-nodes, and all the 
        corresponding origin-destinations in sectors and navpoints. 

        Parameters
        ----------
        n : int
            Sector to shut. 

        Returns
        -------
        delete_pairs : list of 2-tuples
            List of nav-origin-destinations deleted.

        Notes
        -----
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
        """
        Print the network.

        TODO : fix stack and colors.
        """
        draw_network_map(self, title=self.name, load=False, generated=True, airports=hasattr(self,'airports'))

    def check_airports_and_pairs(self):
        """
        Check if everything is consistent between the airports of navpoints, sectors, and the pairs, i.e.:
            -- consistency of short list and airport list for sectors,
            -- consistency of short list and airport list for navpoints,
            -- consistency of sec-short list inferred for nav-short list and nav-airports.
        
        Notes
        -----
        New in 2.9
        """

        try:
            airports_from_sector_short = set([a for b in self.short.keys() for a in b])
            try:
                print 'Checking consistency of sector airports...'
                assert airports_from_sector_short == set(self.airports)
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


            airports_from_navs_short = set([a for b in self.G_nav.short.keys() for a in b])
            try:
                print 'Checking consistency of navpoints airports...'
                assert airports_from_navs_short == set(self.G_nav.airports)
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

            airports_sec_from_navs = set([self.G_nav.node[a]['sec'] for b in self.G_nav.short.keys() for a in b])

            try:
                print 'Checking consistency between sector and navpoints airports...'
                assert airports_sec_from_navs == set(self.airports)
            except AssertionError:
                print 'Airports of sectors infered from short of navpoints is not consistent with list of sector airports:'
                for p in airports_sec_from_navs:
                    if not p in self.airports:
                        print 'Pair', p, 'is in airports of sectors based on navpoints but not in airports of sectors.'
                for p in self.airports:
                    if not p in airports_sec_from_navs:
                        print 'Pair', p, 'is in airports of sectors but not in airports of sectors based on navpoints.' 
                print "airports_sec_from_navs:", airports_sec_from_navs
                print "list(self.airports)", list(self.airports)
                print          
                raise
            except:
                raise
        except:
            raise

    def check_repair_sector_airports(self):
        """
        Remove all pairs of sec-airports which do not have at least one connection in nav-airports.
        Only intended for hybrid network (do not use for simple sector networks).
        """
        pairs_sec_from_navs = list(set([(self.G_nav.node[p1]['sec'], self.G_nav.node[p2]['sec']) for (p1,p2) in self.G_nav.short.keys()]))
        for (s1,s2) in self.short.keys():
            if not (s1,s2) in pairs_sec_from_navs:
                del self.short[(s1,s2)]
                print 'I remove', s1, s2, 'from the airports of sectors because there is no corresponding pairs in navpoints.'

    def check_all_real_flights_are_legitimate(self, flights_selected, repair=False):
        """
        Check if the trajectories from Distance library are compatible with 
        the present network. The method checks for each flight:
         - If the entry/exit are in the list of short,
         - if all navpoints are in the list of nodes,
         - if all segments are in the list of edges.
        If 'repair' is True, the non-legit flights are eliminated.
        Changed in 2.9.8: flights_selected is an external argument.
        TODO: this method should not be used only by sector networks to check 
        the navpoint network...
        """
        fl_s = flights_selected[:]

        for f in fl_s:
            try:
                assert (self.G_nav.idx_nodes[f['route_m1'][0][0]], self.G_nav.idx_nodes[f['route_m1'][-1][0]]) in self.G_nav.short.keys()
            except AssertionError:
                if repair:
                    print "Deleting a flight because its entry/exit pair is not in the list of the network:", (self.G_nav.idx_nodes[f['route_m1'][0][0]], self.G_nav.idx_nodes[f['route_m1'][-1][0]])
                    flights_selected.remove(f)
                    continue
                else:
                    print
                    print 'A flight has a couple source/destination not in the list of pairs of the network.'
                    print f
                    raise

            navpoints = set([self.G_nav.idx_nodes[p[0]] for p in f['route_m1']])
            try:
                assert navpoints.issubset(set(self.G_nav.nodes()))
            except AssertionError:
                if repair:
                    print "Deleting a flight because there is at least one navpoint in its trajectory which is not in the list of nodes."
                    flights_selected.remove(f)
                    continue
                else:
                    print
                    print 'A flight has navpoints not existing in the network:'
                    print f
                    print 'Navpoints not in network:', navpoints.difference(set(self.G_nav.nodes()))
                    raise

            edges = set([(self.G_nav.idx_nodes[f['route_m1'][i][0]], self.G_nav.idx_nodes[f['route_m1'][i+1][0]]) for i in range(len(f['route_m1']) -1)])

            for e1, e2 in edges:
                try:
                    assert e2 in self.G_nav.neighbors(e1)
                except AssertionError:
                    if repair:
                        print "Deleting a flight because a segment of its trajectory is not in the list of edges."
                        flights_selected.remove(f)
                        break
                    else:
                        print
                        print 'A flight is going from one point to the another while they are not connected:'
                        print f
                        print 'Edges not in network:', e1, e2
                        raise
        return flights_selected

    def connections(self):
        """
        New in 2.9.8: returns the possible connections between airports.
        """
        return self.short.keys()

    def get_airports(self):
        """
        New in 2.9.8: returns the airports based on connections.
        """
        return set([e for ee in self.connections() for e in ee])

    def stamp_airports(self):
        """
        New in 2.9.8: compute the list of airports based on short.
        """
        self.airports = self.get_airports()

class NavpointNet(Net):
    """
    Dedicated class for navpoint networks.
    New in 2.8.2.
    """
    def __init__(self):
        super(NavpointNet, self).__init__()

    def remove_node(self, n):
        """
        Remove node on the border or in the bulk transparently.
        """ 
        if hasattr(self, "navpoints_borders"):
            if n in self.navpoints_borders:
               self.navpoints_borders.remove(n)
        super(NavpointNet, self).remove_node(n)
        
    def build_nodes(self, N, sector_list=[], navpoints_borders=[], cutoff=0.):
        """
        Overwrite method of class Net. Build the nodes.

        Begins by adding the nodes on the border. Then add nodes from sector_list.
        Finally, add N nodes randomly in the 1x1 square.

        Parameters
        ----------
        N : int
            Number of nodes to produce randomly in teh 1x1 square
        sector_list : list of 2-tuples, optional
            List of coordinates of nodes to add.
        navpoints_borders : list of 2-tuples, optional
            List of coordinates of nodes to add to navpoin_borders.
        cutoff : float
            Used to avoid navpoints too close to each other. No navpoints can be closer 
            than cutoff according to the euclidean distance.

        """
        
        j=0
        for i,cc in enumerate(navpoints_borders):
            self.add_node(j, coord=cc)
            j+=1
        self.navpoints_borders=range(j)
        for i, cc in enumerate(sector_list):
            self.add_node(j, coord=cc)
            j+=1
        for i in range(N):
            cc=[uniform(-1.,1.),uniform(-1.,1.)]
            if cutoff!=0.:
                not_too_close=True
                k=0
                while not_too_close and k<len(self.nodes()):
                    pcc=self.node[self.nodes()[k]]['coord']
                    not_too_close=np.linalg.norm(np.array(cc) - np.array(pcc))>cutoff
                    k+=1
            if cutoff==0. or not_too_close:
                self.add_node(j, coord=cc)
                j+=1
       
    def build(self, N, nairports, min_dis, generation_of_airports=True, Gtype='D', sigma=1., mean_degree=6, sector_list=[], navpoints_borders=[], shortcut=0.):
        """
        Build a graph of the given type. Build also the corresponding graph used for ksp_yen.
        

        Parameters
        ----------
        N : int
            Number of nodes to produce.
        Gtype : string, 'D', 'E', or 'T'
            type of graph to be generated. 'D' generates a dealunay triangulation 
            which in particular is planar and has the highest degree possible for a planar 
            graph. 'E' generates an Erdos-Renyi graph. 'T' builds a triangular network
        mean_degree : float
            mean degree for the Erdos-Renyi graph.
        prelist : list of 2-tuples, optional
            Coordinates of nodes to add to the nodes previously generated
        put_nodes_at_corners : boolean, optional
            if True, put a nodes at each corner for the 1x1 square

        Notes
        -----
        Changed in 2.8.2: Build_nodes externalized.
        Changed in 2.9.10: don't check weights.

        """
        
        print 'Building random network of type ', Gtype
        
        #self.weighted=sigma==0.
        
        if Gtype=='D' or Gtype=='E':
            self.build_nodes(N, sector_list=sector_list,navpoints_borders=navpoints_borders, shortcut=shortcut)
            self.build_net(Gtype=Gtype, mean_degree=mean_degree)
            
        elif Gtype=='T':
            xAxesNodes=np.sqrt(N/float(1.4))
            self=build_triangular(xAxesNodes)  

        if generation_of_airports:
            self.generate_airports(nairports,min_dis) 
        
    def clean_borders(self):
        """
        Remove all links between border points. 

        Notes
        -----
        Previously: Remove all links between navpoints in two different sectors
        which are both non border points.
        Changed in 2.9.8: don't do the second operation anymore. 
        """
        for e in self.edges():
            if e[0] in self.navpoints_borders and e[1] in self.navpoints_borders:
                self.remove_edge(*e)

    def convert_path(self, path):
        """
        Used to convert a path of navigation points into a path of sectors. Replace the method of class Net.
        
        Parameters
        ----------
        path : list of nav-nodes

        Returns
        -------
        path_sec : list of sec-nodes

        Notes
        -----
        New in 2.9

        """
        path_sec=[]
        j,sec=0,self.node[path[0]]['sec']
        while j<len(path):
            path_sec.append(sec)
            while j<len(path) and self.node[path[j]]['sec']==sec:
                j+=1
            if j<len(path):
                sec=self.node[path[j]]['sec']
        return path_sec

    def infer_airports_from_sectors(self, airports_sec, min_dis):
        """
        Choose exactly one navpoint in each sector of airports_sec and fix it as airport 
        of the navpoint network.
        
        Parameters
        ----------
        airports_sec : list of sec-node
        min_dis : int
            minimum topological distance under which a connectoin cannot be created
            (for the navpoint network)

        Notes
        -----
        TODO : add pairs to fix_airports?
        """
        airports_nav=[choice([n for n in self.nodes() if self.node[n]['sec']==sec]) for sec in airports_sec]
        self.fix_airports(airports_nav, min_dis)# add pairs !!!!!!!!!!!!!!!! TODO Not important?

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
   
