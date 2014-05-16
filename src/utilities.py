# -*- coding: utf-8 -*-
"""
Created on Thu May 23 11:24:00 2013

@author: earendil

Utilies for the ABM.
"""

from mpl_toolkits.basemap import Basemap
from math import sqrt, cos, sin, pi
import numpy as np
import matplotlib.gridspec as gridspec
from descartes import PolygonPatch
import matplotlib.pyplot as plt
#from random import shuffle, seed
import networkx as nx

version='2.9.0'

#seed(3)
_colors=['Blue','BlueViolet','Brown','CadetBlue','Crimson','DarkMagenta','DarkRed','DeepPink','Gold','Green','OrangeRed','Red']

#shuffle(_colors)

def draw_network_map(G, title='Network map', trajectories=[], rep='./',airports=True, load=True, generated=False, add_to_title='', polygons=[], numbers=False, show=True):
    x_min=min([G.node[n]['coord'][0]/60. for n in G.nodes()])-0.5
    x_max=max([G.node[n]['coord'][0]/60. for n in G.nodes()])+0.5
    y_min=min([G.node[n]['coord'][1]/60. for n in G.nodes()])-0.5
    y_max=max([G.node[n]['coord'][1]/60. for n in G.nodes()])+0.5
    

    #(x_min,y_min,x_max,y_max),G,airports,max_wei,zone_geo = rest
    fig=plt.figure(figsize=(9,6))#*(y_max-y_min)/(x_max-x_min)))#,dpi=600)
    gs = gridspec.GridSpec(1, 2,width_ratios=[6.,1.])
    ax = plt.subplot(gs[0])
    ax.set_aspect(1./0.8)

    
    if generated:
        def m(a,b):
            return a,b
        y,x=[G.node[n]['coord'][0] for n in G.nodes()], [G.node[n]['coord'][1] for n in G.nodes()]
    else:
        m=draw_zonemap(x_min,y_min,x_max,y_max,'i')
        x,y=split_coords(G,G.nodes(),r=0.08)
    
    for i,pol in enumerate(polygons):
        patch = PolygonPatch(pol,alpha=0.5, zorder=2, color=_colors[i%len(_colors)])
        ax.add_patch(patch) 
       # for n in self.nodes():
       #     print self.node[n]['load']
       #     print [self.node[n]['load'][i][1] for i in range(len(self.node[n]['load'])-1)]
    if load:
        sze=[(np.average([G.node[n]['load'][i][1] for i in range(len(G.node[n]['load'])-1)],\
        weights=[(G.node[n]['load'][i+1][0] - G.node[n]['load'][i][0]) for i in range(len(G.node[n]['load'])-1)])
        /float(G.node[n]['capacity'])*800 + 5) for n in G.nodes()]
    else:
        sze=10
        
    coords={n:m(y[i],x[i]) for i,n in enumerate(G.nodes())}
    
    ax.set_title(title)
    #sca = ax.scatter([self.node[n]['coord'][0] for n in self.nodes()],[self.node[n]['coord'][0] for n in self.nodes()],marker='o',zorder=6,s=sze)#,s=snf,lw=0,c=[0.,0.45,0.,1])
    sca=ax.scatter([coords[n][0] for n in G.nodes()],[coords[n][1] for n in G.nodes()],marker='o',zorder=6,s=sze,c='b')#,s=snf,lw=0,c=[0.,0.45,0.,1])
    if airports:
        scairports=ax.scatter([coords[n][0] for n in G.airports],[coords[n][1] for n in G.airports],marker='o',zorder=6,s=20,c='r')#,s=snf,lw=0,c=[0.,0.45,0.,1])
       # scaa=ax.scatter(x_a,y_a,marker='s',zorder=5,s=sna,c=[0.7,0.133,0.133,1],edgecolor=[0,0,0,1],lw=0.7)
       # scat = ax.scatter(x_m1t,y_m1t,marker='d',zorder=6,s=snt,lw=0,c=[0.,0.45,0.,1])

    if 1:
        for e in G.edges():
            # if G.node[e[0]]['m1'] and G.node[e[1]]['m1']:
            #     print e,width(G[e[0]][e[1]]['weight'])
            #        xe1,ye1=m(self.node[e[0]]['coord'][1]/60.,self.node[e[0]]['coord'][0]/60.)
            #        xe2,ye2=m(self.node[e[1]]['coord'][1]/60.,self.node[e[1]]['coord'][0]/60.)
                plt.plot([coords[e[0]][0],coords[e[1]][0]],[coords[e[0]][1],coords[e[1]][1]],'k-',lw=0.5)#,lw=width(G[e[0]][e[1]]['weight'],max_wei),zorder=4)
          
    weights={n:{v:0. for v in G.neighbors(n)} for n in G.nodes()}
    for path in trajectories:
        try:
            #path=f.FPs[[fpp.accepted for fpp in f.FPs].index(True)].p
            for i in range(0,len(path)-1):
                print path[i], path[i+1]
                weights[path[i]][path[i+1]]+=1.
        except ValueError:
            pass
        except:
            raise
    
    max_w=np.max([w for vois in weights.values() for w in vois.values()])
    
    print 'max_w', max_w
    
    for n,vois in weights.items():
        for v,w in vois.items():
           # if G.node[n]['m1'] and G.node[v]['m1']:
                plt.plot([coords[n][0],coords[v][0]],[coords[n][1],coords[v][1]],'r-',lw=w/max_w*4.)#,lw=width(G[e[0]][e[1]]['weight'],max_wei),zorder=4)

    if numbers:
        for n in G.nodes():
            plt.text(G.node[n]['coord'][0], G.node[n]['coord'][1], ster(n))
       # if 0:
       #     patch=PolygonPatch(adapt_shape_to_map(zone_geo,m),facecolor='grey', edgecolor='grey', alpha=0.08,zorder=3)#edgecolor='grey', alpha=0.08,zorder=3)
       #     ax.add_patch(patch)
           
       # if 0:
       #     patch=PolygonPatch(adapt_shape_to_map(expand(zone_geo,0.005),m),facecolor='brown', edgecolor='black', alpha=0.1,zorder=3)#edgecolor='grey', alpha=0.08,zorder=3)
       #     ax.add_patch(patch)
    plt.savefig(rep + 'network_flights' + add_to_title + '.png',dpi=300)
    if show:
        plt.show()


def split_coords(G,nodes,r=0.04):
    lines=[]
    for n in G.nodes():
        if n in nodes:
            added=False
            for l in lines:
                if sqrt((G.node[n]['coord'][0] - G.node[l[0]]['coord'][0])**2 + (G.node[n]['coord'][1] - G.node[l[0]]['coord'][1])**2)<1.: #nodes closer than 0.1 degree
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
            theta=2.*pi*float(l.index(n))/float(len(l))
            x.append(G.node[n]['coord'][0]/60. + r*cos(theta))
            y.append(G.node[n]['coord'][1]/60. + r*sin(theta))
    return x,y
    
def draw_zonemap(x_min,y_min,x_max,y_max,res):
    m = Basemap(projection='gall',lon_0=0.,llcrnrlon=y_min,llcrnrlat=x_min,urcrnrlon=y_max,urcrnrlat=x_max,resolution=res)
    m.drawmapboundary(fill_color='white') #set a background colour
    m.fillcontinents(color='white',lake_color='white')  # #85A6D9')
    m.drawcoastlines(color='#6D5F47', linewidth=0.8)
    m.drawcountries(color='#6D5F47', linewidth=0.8)
    m.drawmeridians(np.arange(-180, 180, 5), color='#bbbbbb')
    m.drawparallels(np.arange(-90, 90, 5), color='#bbbbbb')
    return 

def restrict_to_connected_components(G):
    """
    And keeping all other attributes.
    """
    CC=nx.connected_component_subgraphs(G)[0]
    for n in G.nodes():
        if not n in CC.nodes():
            G.remove_node(n)
    return G
