#!/usr/bin/env python

import sys
sys.path.insert(1, '..')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.pylab import *
import datetime as dt
from mpl_toolkits.basemap import Basemap

from abm_strategic.utilities import read_trajectories_for_tact, draw_zonemap
from libs.general_tools import nice_colors, _colors as colors

time_res = 30# seconds
y_min, y_max = 6, 15
x_min, x_max = 36, 47

m = Basemap(projection='gall', lon_0=0., llcrnrlon=y_min, llcrnrlat=x_min, urcrnrlon=y_max, urcrnrlat=x_max, resolution='i')

def make_time(it, date=dt.datetime(2010, 5, 6, 0, 0, 0)):
	current_date = date + dt.timedelta(seconds=time_res*it)
	return current_date

def find_active_flights(trajs, date):
	return 

class Flight:
	def __init__(self, color, traj):
		self.color = color
		self.traj = traj
		self.p0 = None
		self.p1 = None

		self.pos = [[y, x] for x, y, z, t, s in traj]
		# Transform the coordinates with Basemap projection
		self.pos = [np.array(p) for p in zip(*m(*zip(*self.pos)))]
		self.t = [dt.datetime(*t) for x, y, z, t, s in traj]

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

	def compute_position(self, date):
		# Extrapolate position at time "date":
		self.r = self.r0 + self.v*(date - self.t[self.p0]).total_seconds()

	def is_active(self, date):
		return self.t[0] <= date < self.t[-1]

	def is_passed(self, date):
		return date > self.t[-1]

	def plot(self):
		return plot([self.r[0]], [self.r[1]], 'o', c=self.color)


def prepare_flights(trajs):
	flights = []
	for i, traj in enumerate(trajs):
		flights.append(Flight(colors[i%len(colors)], traj))

	return flights

def init(it, flights, ttl, starting_date=dt.datetime(2010, 5, 6, 0, 0, 0)):
	date = make_time(it, date=starting_date)
	ttl.set_text(str(it))
	#title('Iteration:' + str(it))
	print it
	#active_flights = find_active_flights(trajs, date)
	objs = []
	#for j, traj in enumerate(trajs):
	for fl in (f for f in flights if f.is_active(date)):
		fl.update(date)
		
		for line in fl.plot():
		#for line in plot([0], [1], 'o', c='b'):
			objs.append(line)

	#for line in plot([40], [10], 'o', c='b'):
	#	objs.append(line)		

	return objs

def init():
	objs = []
	for line in plot([40], [10], 'o', c='b'):
		objs.append(line)	

	#draw_zonemap(8, 35, 15, 45, 'i', sea_color=background_color, continents_color=background_color, lake_color=background_color,\
    #                                                lw=lw_map, draw_mer_par=draw_mer_par)

	
	#for ob in m.drawmapboundary(fill_color='white'):
	#set a background colour
	#	objs.append(ob)
	#m.fillcontinents(color='white', lake_color='white')  # #85A6D9')
	pouet = m.drawcoastlines(color='#6D5F47', linewidth=0.8)
	objs.append(pouet)
	#m.drawcountries(color='#6D5F47', linewidth=0.8)
	#m.drawmeridians(np.arange(-180, 180, 5), color='#bbbbbb')
	#m.drawparallels(np.arange(-90, 90, 5), color='#bbbbbb')

	return objs

def animate(it, flights, ttl, starting_date=dt.datetime(2010, 5, 6, 0, 0, 0)):
	date = make_time(it, date=starting_date)
	ttl.set_text(str(it))
	t = ax.annotate(date.strftime('%m/%d/%Y %H:%M'),(0.3,0.11), textcoords='figure fraction')
	#title('Iteration:' + str(it))
	print it
	objs = [t]

	for fl in (f for f in flights if f.is_active(date)):
		fl.update(date)
		
		for line in fl.plot():
			objs.append(line)

	#for line in plot([40], [10], 'o', c='b'):
	#	objs.append(line)		

	return objs


# def _blit_draw(self, artists, bg_cache):
# 	# found here: http://stackoverflow.com/questions/17558096/animated-title-in-matplotlib

# 	# Handles blitted drawing, which renders only the artists given instead
# 	# of the entire figure.
# 	updated_ax = []
# 	for a in artists:
# 		# If we haven't cached the background for this axes object, do
# 		# so now. This might not always be reliable, but it's an attempt
# 		# to automate the process.
# 		if a.axes not in bg_cache:
# 			# bg_cache[a.axes] = a.figure.canvas.copy_from_bbox(a.axes.bbox)
# 			# change here
# 			bg_cache[a.axes] = a.figure.canvas.copy_from_bbox(a.axes.figure.bbox)
# 		a.axes.draw_artist(a)
# 		updated_ax.append(a.axes)

# 	# After rendering all the needed artists, blit each axes individually.
# 	for ax in set(updated_ax):
# 		# and here
# 		# ax.figure.canvas.blit(ax.bbox)
# 		ax.figure.canvas.blit(ax.figure.bbox)

# # MONKEY PATCH!!
# matplotlib.animation.Animation._blit_draw = _blit_draw

if __name__=='__main__':
	if len(sys.argv)==2:
		trajectories_file = sys.argv[1]
	else:
		Exception("Please provide a file name.")

	trajectories = read_trajectories_for_tact(trajectories_file)[:]

	# for traj in trajectories:
	# 	print traj
	# 	print
	# print

	starting_date = dt.datetime(2010, 5, 6, 12, 0, 0)

	flights = prepare_flights(trajectories)

	fig = plt.figure()
	ax = plt.axes()
	#ax.set_xlim([0,2*2*np.pi])
	#ttl = ax.set_title('',animated=True)
	ttl = ax.text(-0.5, 1.05, '', transform=ax.transAxes, va='center')
	#ylim((8, 15))
	#xlim((35, 45))
	ani = animation.FuncAnimation(fig, animate, frames=1000, interval=200, blit=True, repeat_delay=3000,
									 fargs=(flights, ttl, starting_date),
									 init_func=init)
	#im_ani.save('im.mp4', metadata={'artist':'Guido'})
	show()