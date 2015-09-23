#!/usr/bin/env python

import sys
sys.path.insert(1, '..')
from os.path import join as jn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.pylab import *
from matplotlib.collections import PatchCollection
import datetime as dt
from mpl_toolkits.basemap import Basemap
from collections import deque
from descartes import PolygonPatch
from shapely.geometry import Polygon

from abm_strategic.utilities import read_trajectories_for_tact, draw_zonemap
from libs.general_tools import nice_colors, _colors as colors, simple_color_map_function

time_res = 30# seconds
y_min, y_max = 6, 15
x_min, x_max = 36, 47

 #altitudes_colors = {alt:colors[i%len(colors)] for i, alt in enumerate(range(200, 450, 10))}
color_map = simple_color_map_function((1., 0., 0.), (0., 0., 1.), min_value=240, max_value=420)

altitudes_colors = {alt:color_map(alt) for alt in range(240, 420, 10)}

m = Basemap(projection='gall', lon_0=0., llcrnrlon=y_min, llcrnrlat=x_min, urcrnrlon=y_max, urcrnrlat=x_max, resolution='i')

# Conversion for nautical miles
# 1 nautical mile is 1 minute of arc along a meridian.
x, y = m([(y_max-y_min)/2., (y_max-y_min)/2.], [(x_max-x_min)/2., (x_max-x_min)/2.+1./60.])
p1, p2 = list(zip(x,y))
# Approximate distance corresponding to 1 nautical mile in the projection of m:
d_NM = np.linalg.norm(np.array(p1) - np.array(p2))

def make_time(it, date=dt.datetime(2010, 5, 6, 0, 0, 0)):
	current_date = date + dt.timedelta(seconds=time_res*it)
	return current_date

class Flight:
	def __init__(self, traj, ax, len_trail=10):
		self.ax = ax # Axis

		self.traj = traj
		self.p0 = None
		self.p1 = None

		self.pos = [[y, x] for x, y, z, t, s in traj]
		# Transform the coordinates with Basemap projection
		self.pos = [np.array(p) for p in zip(*m(*zip(*self.pos)))]
		self.t = [dt.datetime(*t) for x, y, z, t, s in traj]

		self.len_trail = len_trail

		#self.prev_r = deque([(0., 0.)]*len_trail)
		#self.update(self.t[0])
		#self.prev_r = deque([self.r]*len_trail)
		self.prev_r = deque([])

		self.color = altitudes_colors[int(self.traj[0][2])]

	def update(self, date):
		# Find the points just before and just after date.
		idx = 1
		while self.t[idx]<date:
			idx += 1
		if self.p1!=idx:
			self.p0 = idx-1
			self.p1 = idx

			# Compute position vectors before and after
			self.r0 = self.pos[self.p0]
			self.r1 = self.pos[self.p1]

			# Compute velocity on this segment
			self.v = (self.r1 - self.r0)/(self.t[self.p1] - self.t[self.p0]).total_seconds()
		
		self.compute_position(date)

		self.alt = int(self.traj[self.p0][2])
		self.color = altitudes_colors[self.alt]

	def compute_position(self, date):
		# Extrapolate position at time "date":
		self.r = self.r0 + self.v*(date - self.t[self.p0]).total_seconds()
		if len(self.prev_r)==self.len_trail:
			self.prev_r.pop()
		self.prev_r.appendleft(self.r)

	def is_active(self, date):
		return self.t[0] <= date < self.t[-1]

	def is_passed(self, date):
		return date > self.t[-1]

	def plot(self, plot_alt=True):
		patches = []
		others = []

		# Five nautical miles safety area
		circle = Circle(self.r, radius=5*d_NM, alpha=.6, color=self.color)
		patches.append(circle)

		if plot_alt:
			# Altitude
			alt = self.ax.annotate(str(self.alt), self.r+ 5*np.array([d_NM, d_NM]))# textcoords='figure fraction')
			others.append(alt) 

		# trail
		coords = list(zip(*self.prev_r))
		trail = plot(coords[0], coords[1], '-', color=self.color)
		others += trail
		
		# Put all the patches in the first coordinates, all the lines in the second
		#image = [circle], trail + [alt]
		return patches, others


def prepare_flights(trajs, ax, len_trail=5):
	flights = []
	for i, traj in enumerate(trajs):
		flights.append(Flight(traj, ax, len_trail=len_trail))

	return flights

def init():
	objs = []
	for line in plot([40], [10], 'o', c='b'):
		objs.append(line)	

	#draw_zonemap(8, 35, 15, 45, 'i', sea_color=background_color, continents_color=background_color, lake_color=background_color,\
    #                                                lw=lw_map, draw_mer_par=draw_mer_par)

	
	#for ob in m.drawmapboundary(fill_color='white'):
	#set a background colour
	#	objs.append(ob)

	#objs.append(m.drawmapboundary(fill_color='white'))
	#objs.append(m.fillcontinents(color='white', lake_color='white'))  # #85A6D9')
	objs.append(m.drawcoastlines(color='#6D5F47', linewidth=0.8))
	#objs.append(pouet)
	objs.append(m.drawcountries(color='#6D5F47', linewidth=0.8))
	#objs.append(m.drawmeridians(np.arange(-180, 180, 5), color='#bbbbbb'))
	#m.drawparallels(np.arange(-90, 90, 5), color='#bbbbbb')

	return objs

def animate(it, flights, ttl, ax, paras, starting_date=dt.datetime(2010, 5, 6, 0, 0, 0)):
	date = make_time(it, date=starting_date)
	ttl.set_text(str(it))
	t = ax.annotate(date.strftime('%m/%d/%Y %H:%M'),(0.3, 0.11), textcoords='figure fraction')
	#title('Iteration:' + str(it))
	print it
	objs = [t]

	for fl in (f for f in flights if f.is_active(date)):
		fl.update(date)
		patches, lines = fl.plot(plot_alt=paras['plot_alt'])

		for patch in patches:
			ax.add_patch(patch)
			objs.append(patch)

		for line in lines:
			objs.append(line)				

	return objs

if __name__=='__main__':
	if len(sys.argv)>=2:
		if sys.argv[1]!='test':
			trajectories_file = sys.argv[1]
			if len(sys.argv)==3:
				area_file = sys.argv[2]
			else:
				area_file = None
		else:
			# for test
			#trajectories_file = '/home/earendil/Documents/ELSA/ABM/results/trajectories/M1/trajs_Real_LI_v5.8_Strong_EXTLIRR_LIRR_2010-5-6+0_d2_cut240.0_directed_rej0.02_new_0.dat'
			trajectories_file = '/media/earendil/Lothlorien/Downloads/Deconflict_Sim/DECONFLICT/inputABM_n-0_Eff-0.9728914418_Nf-2000.dat'
			area_file = '/home/earendil/Documents/ELSA/ABM/ABM_FINAL/abm_tactical/config/bound_latlon_LIRR.dat'

	else:
		Exception("Please provide a file name.")

	paras ={'trail':30,
			'plot_alt':True,
			'area_file':area_file,
			'result_dir':'/home/earendil/Documents/ELSA/ABM/results'}

	trajectories = read_trajectories_for_tact(trajectories_file)#, fmt_out='(x, y, z, t)')[:]

	#starting_date = dt.datetime(2010, 5, 6, 12, 0, 0)
	starting_date = dt.datetime(2014, 8, 8, 8, 0, 0)

	fig = plt.figure(figsize=(12, 15))
	ax = plt.axes()

	flights = prepare_flights(trajectories, ax, len_trail=paras['trail'])
	#ax.set_xlim([0,2*2*np.pi])
	#ttl = ax.set_title('',animated=True)
	ttl = ax.text(-0.5, 1.05, '', transform=ax.transAxes, va='center')

	# Plot area
	if area_file!=None:
		with open(area_file, 'r') as f:
			lines = f.readlines()
		lines[0].split('\t')
		pts = [(float(l.split('\t')[0]), float(l.split('\t')[1])) for l in lines]
		x, y = list(zip(*pts))
		pol = Polygon(zip(*m(y, x)))
		patch = PolygonPatch(pol, alpha=0.05, zorder=-1, color='k')
        ax.add_patch(patch) 
		
	ani = animation.FuncAnimation(fig, animate, frames=100, interval=200, blit=True, repeat=False, repeat_delay=3000,
									 fargs=(flights, ttl, ax, paras, starting_date),
									 init_func=init)
	#ani.save(jn(paras['result_dir'], 'test_movie.mp4'), metadata={'artist':'earendil'})
	ani.save('pouet.mp4', writer='ffmpeg', metadata={'artist':'earendil'})
	show()