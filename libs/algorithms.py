#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  algorithms.py
#  
#  Copyright 2012 Kevin R <KRPent@gmail.com>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#
# 
from operator import itemgetter
from prioritydictionary import priorityDictionary
from graph import DiGraph


## @package YenKSP
# Computes K-Shortest Paths using Yen's Algorithm.
#
# Yen's algorithm computes single-source K-shortest loopless paths for a graph 
# with non-negative edge cost. The algorithm was published by Jin Y. Yen in 1971
# and implores any shortest path algorithm to find the best path, then proceeds 
# to find K-1 deviations of the best path.

## Computes K paths from a source to a sink in the supplied graph.
#
# @param graph A digraph of class Graph.
# @param start The source node of the graph.
# @param sink The sink node of the graph.
# @param K The amount of paths being computed.
#
# @retval [] Array of paths, where [0] is the shortest, [1] is the next 
# shortest, and so on.
#
# class SimplifiedData:
#     ## An edge with this cost signifies that it has been removed from the graph.
#     # This value implies that any edge in the graph must be very small in 
#     # comparison.
#     INFINITY = 10000
    
#     ## Represents a NULL predecessor.
#     UNDEFINDED = None
    
#     ## The dictionary of the graph. Each key represents a node and the value of
#     # the associated node is a dictionary of all the edges. The key of the edges
#     # dictionary is the node the edge terminates at and the value is the cost of
#     # the edge.
#     _data = {}

#     _name='poeut'
#     ## Gets the edges of a specified node.
#     #
#     # @param self The object pointer.
#     # @param node The node whose edges are being queried.
#     # @retval {} A dictionary of the edges and thier cost if the node exist 
#     # within the graph or None if the node is not in the graph.
#     #
#     def __init__(self, name=None):
#         if name:
#             self._name = name
#     def __getitem__(self, node):
#         if self._data.has_key(node):
#             return self._data[node]
#         else:
#             return None

#     ## Iterator for the digraph object.
#     #
#     # @param self The object pointer.
#     # @retval iter An iterator that can be used to process each node of the 
#     # graph.
#     #
#     def __iter__(self):
#         return self._data.__iter__()

#     ## Adds a node to the graph.
#     #
#     # @param self The object pointer.
#     # @param node The name of the node that is to be added to the graph.
#     # @retval bool True if the node was added or False if the node already 
#     # existed in the graph.
#     #
    
#     def import_from_graph(self,G):
#         for n in G.nodes():
#             self.add_node(n)
#         for e in G.edges():
#             self.add_edge(e[0],e[1],G[e[0]][e[1]]['weight'])
        
#     def add_node(self, node):
#         if self._data.has_key(node):
#             return False

#         self._data[node] = {}
#         return True

#     ## Adds a edge to the graph.
#     #
#     # @post The two nodes specified exist within the graph and their exist an
#     # edge between them of the specified value.
#     #
#     # @param self The object pointer.
#     # @param node_from The node that the edge starts at.
#     # @param node_to The node that the edge terminates at.
#     # @param cost The cost of the edge, if the cost is not specified a random
#     # cost is generated from 1 to 10.
#     #
#     def add_edge(self, node_from, node_to, cost):
#         self.add_node(node_from)
#         self.add_node(node_to)
        
#         self._data[node_from][node_to] = cost
#         return

#     ## Removes an edge from the graph.
#     #
#     # @param self The object pointer.
#     # @param node_from The node that the edge starts at.
#     # @param node_to The node that the edge terminates at.
#     # @param cost The cost of the edge, if the cost is not specified all edges
#     # between the nodes are removed.
#     # @retval int The cost of the edge that was removed. If the nodes of the 
#     # edge does not exist, or the cost of the edge was found to be infinity, or 
#     # if the specified edge does not exist, then -1 is returned.
#     #
#     def remove_edge(self, node_from, node_to, cost=None):
#         if not self._data.has_key(node_from):
#             return -1
        
#         if self._data[node_from].has_key(node_to):
#             if not cost:
#                 cost = self._data[node_from][node_to]
                
#                 if cost == self.INFINITY:
#                     return -1
#                 else:
#                     self._data[node_from][node_to] = self.INFINITY
#                     return cost
#             elif self._data[node_from][node_to] == cost:
#                 self._data[node_from][node_to] = self.INFINITY
                
#                 return cost
#             else:
#                 return -1
#         else:
#             return -1
            
            
        
def ksp_yen(graph, node_start, node_end, max_k=2):
    distances, previous = dijkstra(graph, node_start)
    
    A = [{'cost': distances[node_end], 
          'path': path(previous, node_start, node_end)}]
    B = []
    
    #print 'Best shortest path: ', A
    #print
    if not A[0]['path']: return A
    
    for k in range(1, max_k):
        #print 'Finding shortest path of order', k
        if k>1:
            distances=make_distances(graph, A[-1]['path'])
        for i in range(0, len(A[-1]['path']) - 1):
            node_spur = A[-1]['path'][i]
            path_root = A[-1]['path'][:i+1]
            
            edges_removed = []
            for path_k in A:
                curr_path = path_k['path']
                if len(curr_path) > i and path_root == curr_path[:i+1]:
                    cost = graph.remove_edge(curr_path[i], curr_path[i+1])
                    if cost == -1:
                        continue
                    edges_removed.append([curr_path[i], curr_path[i+1], cost])
            
            path_spur = dijkstra(graph, node_spur, node_end)
            
            if path_spur['path']:
                path_total = path_root[:-1] + path_spur['path']
                dist_total = distances[node_spur] + path_spur['cost']  # Distance not updated!
                #print node_spur, path_root[:-1], path_spur['path']
                #print node_spur, distances[node_spur], path_spur['cost']
                potential_k = {'cost': dist_total, 'path': path_total}
            
                if not (potential_k in B):
                    B.append(potential_k)
            
            for edge in edges_removed:
                graph.add_edge(edge[0], edge[1], edge[2])
        
        if len(B):
            #print 'Potential:', B
            B = sorted(B, key=itemgetter('cost'))
            #print 'Best:', B[0]
            A.append(B[0])
            B.pop(0)
        else:
            break
        #print
    #print
    return A

def ksp_yen_old(graph, node_start, node_end, max_k=2):
    distances, previous = dijkstra(graph, node_start)
    
    A = [{'cost': distances[node_end], 
          'path': path(previous, node_start, node_end)}]
    B = []
    
    #print 'Best shortest path: ', A
    #print
    if not A[0]['path']: return A
    
    for k in range(1, max_k):
        #print 'Finding shortest path of order', k
        # if k>1:
        #     distances=make_distances(graph, A[-1]['path'])
        for i in range(0, len(A[-1]['path']) - 1):
            node_spur = A[-1]['path'][i]
            path_root = A[-1]['path'][:i+1]
            
            edges_removed = []
            for path_k in A:
                curr_path = path_k['path']
                if len(curr_path) > i and path_root == curr_path[:i+1]:
                    cost = graph.remove_edge(curr_path[i], curr_path[i+1])
                    if cost == -1:
                        continue
                    edges_removed.append([curr_path[i], curr_path[i+1], cost])
            
            path_spur = dijkstra(graph, node_spur, node_end)
            
            if path_spur['path']:
                path_total = path_root[:-1] + path_spur['path']
                dist_total = distances[node_spur] + path_spur['cost']  # Distance not updated!
                #print node_spur, path_root[:-1], path_spur['path']
                #print node_spur, distances[node_spur], path_spur['cost']
                potential_k = {'cost': dist_total, 'path': path_total}
            
                if not (potential_k in B):
                    B.append(potential_k)
            
            for edge in edges_removed:
                graph.add_edge(edge[0], edge[1], edge[2])
        
        if len(B):
            #print 'Potential:', B
            B = sorted(B, key=itemgetter('cost'))
            #print 'Best:', B[0]
            A.append(B[0])
            B.pop(0)
        else:
            break
        #print
    #print
    return A
    
#def ksp_yen_bk(G, node_start, node_end, max_k=2):
#    graph=SimplifiedData()
#    graph.import_from_graph(G)
#
#    print graph._data
#
#    distances, previous = dijkstra_bk(graph, node_start)
#    print distances, previous
#    A = [{'cost': distances[node_end], 
#          'path': path_bk(previous, node_start, node_end, graph.INFINITY)}]
#    B = []
#    
#    if not A[0]['path']: return A
#    
#    for k in range(1, max_k):
#        for i in range(0, len(A[-1]['path']) - 1):
#            node_spur = A[-1]['path'][i]
#            path_root = A[-1]['path'][:i+1]
#            
#            edges_removed = []
#            for path_k in A:
#                curr_path = path_k['path']
#                if len(curr_path) > i and path_root == curr_path[:i+1]:
#                    cost =graph.remove_edge(curr_path[i], curr_path[i+1])
#                    if cost == -1:
#                        continue
#                    edges_removed.append([curr_path[i], curr_path[i+1], cost])
#            #print graph[39][86]['weight']
#            path_spur = dijkstra_bk(graph, node_spur, node_end)
#            
#            if path_spur['path']:
#                path_total = path_root[:-1] + path_spur['path']
#                dist_total = distances[node_spur] + path_spur['cost']
#                potential_k = {'cost': dist_total, 'path': path_total}
#            
#                if not (potential_k in B):
#                    B.append(potential_k)
#            
#            for edge in edges_removed:
#                graph.add_edge(edge[0], edge[1], edge[2])
#        
#        if len(B):
#            B = sorted(B, key=itemgetter('cost'))
#            A.append(B[0])
#            B.pop(0)
#        else:
#            break
#    
#    return A


## Computes the shortest path from a source to a sink in the supplied graph.
#
# @param graph A digraph of class Graph.
# @param node_start The source node of the graph.
# @param node_end The sink node of the graph.
#
# @retval {} Dictionary of path and cost or if the node_end is not specified,
# the distances and previous lists are returned.
#
def dijkstra(graph, node_start, node_end=None):
    distances = {}      
    previous = {}       
    Q = priorityDictionary()
    
    for v in graph:
        distances[v] = graph.INFINITY
        previous[v] = graph.UNDEFINDED
        Q[v] = graph.INFINITY
    
    distances[node_start] = 0
    Q[node_start] = 0
    
    for v in Q:
        if v == node_end: break

        for u in graph[v]:
#            if graph[v][u]==None:
#                print 'BOUH', distances[v], graph[v][u]
            cost_vu = distances[v] + graph[v][u]
            if cost_vu < distances[u]:
                #if node_start==
                #print 'dijkstra', v,u,graph[v][u]
                distances[u] = cost_vu
                Q[u] = cost_vu
                previous[u] = v

    if node_end!=None:
    #if node_end:
        return {'cost': distances[node_end], 
                'path': path(previous, node_start, node_end)}
    else:
        return (distances, previous)
        
#def dijkstra_bk(graph, node_start, node_end=None):
#    distances = {}      
#    previous = {}       
#    Q = priorityDictionary()
#    
#    for v in graph:
#        distances[v] = graph.INFINITY
#        previous[v] = graph.UNDEFINDED
#        Q[v] = graph.INFINITY
#    
#    distances[node_start] = 0
#    Q[node_start] = 0
#    
#    for v in Q:
#        if v == node_end: break
#
#        for u in graph[v]:
#            if graph[v][u]==None:
#                print 'BOUH', v,u,distances[v], graph[v][u]
#            cost_vu = distances[v] + graph[v][u]
#            print 'cost_vu', u,v, cost_vu, distances[u], cost_vu<distances[u]
#            if cost_vu < distances[u]:
#                distances[u] = cost_vu
#                Q[u] = cost_vu
#                previous[u] = v
#    print
#    print previous
#    print
#    if node_end!=None:
#        return {'cost': distances[node_end], 
#                'path': path_bk(graph, previous, node_start, node_end, graph.INFINITY)}
#    else:
#        return (distances, previous)

## Finds a paths from a source to a sink using a supplied previous node list.
#
# @param previous A list of node predecessors.
# @param node_start The source node of the graph.
# @param node_end The sink node of the graph.
#
# @retval [] Array of nodes if a path is found, an empty list if no path is 
# found from the source to sink.
#
def path(previous, node_start, node_end):
    route = []

    node_curr = node_end    
    while True:
        route.append(node_curr)
        if previous[node_curr] == node_start:
            route.append(node_start)
            break
        elif previous[node_curr] == DiGraph.UNDEFINDED:
            return []
        
        node_curr = previous[node_curr]
    
    route.reverse()
    return route


## Returns the cumulative distance along the given path.
#
# @param graph A digraph of class Graph.
# @param path a list of nodes of the graph.
#
# @retval [] Array of cumulative cost.
#
def make_distances(graph, path):
    distances = {path[0]:0.}
    cost = 0.
    for i in range(len(path)-1):
        cost += graph[path[i]][path[i+1]]
        distances[path[i+1]] = cost 
    return distances
        
#def path_bk(previous, node_start, node_end, inf=100000):
#    route = []
#    print previous
#    node_curr = node_end    
#    while True:
#        route.append(node_curr)
#        print node_curr
#        if previous[node_curr] == node_start:
#            route.append(node_start)
#            break
#        elif previous[node_curr] == inf:
#            return []
#        
#        node_curr = previous[node_curr]
#    
#    route.reverse()
#    return route
