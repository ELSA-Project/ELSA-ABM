#!/usr/bin/env python

import sys
sys.path.insert(1, '..')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.pylab import *

from abm_strategic.utilities import read_trajectories_for_tact
from libs.general_tools import nice_colors

c = 'k'
lw = 1.

def init():
	"""initialize animation"""
	#line.set_data([], [])
	points, edges = simple_fraction(10, 1.)

	objs = []
	for n1, n2 in edges:
		for line in plot(*zip(*[points[n1], points[n2]]), c=c, lw=lw):
			objs.append(line)

	return objs

def animate(x, trajs):
	objs = []
	for j, traj in enumerate(trajs):
		xx, yy = [], []
		for i in range(min(x, len(traj))):
			xx.append(traj[i][0])
			yy.append(traj[i][1])

		for line in plot(xx, yy, '-o', c=nice_colors[j%len(nice_colors)]):
			objs.append(line)

	return objs



if __name__=='__main__':
	if len(sys.argv)==2:
		trajectories_file = sys.argv[1]
	else:
		Exception("Please provide a file name.")

	trajectories = read_trajectories_for_tact(trajectories_file)[:20]

	for traj in trajectories:
		print traj
		print
	print

	fig = plt.figure()
	ani = animation.FuncAnimation(fig, animate, frames=10, interval=200, blit=True, repeat_delay=3000, fargs=(trajectories, ))# init_func=init
	#im_ani.save('im.mp4', metadata={'artist':'Guido'})
	show()