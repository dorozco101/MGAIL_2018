#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 14:36:15 2018

@author: nick
"""

import numpy as np
import matplotlib.pyplot as plt

def save_fig(itrs,means,stds, filepath="./graph_rewards.png",title = 'rewards vs itr',
             x_label="iteration", y_label="avg reward", x_range=(0, 1), y_range=(0,1), color="blue",  grid=True):
  fig = plt.figure()
  #ax = fig.add_subplot(111, autoscale_on=False, xlim=x_range, ylim=y_range)
  ax = fig.add_subplot(111, autoscale_on=True, xlim=x_range, ylim=y_range)
  ax.grid(grid)
  ax.set_xlabel(x_label)
  ax.set_ylabel(y_label)
  ax.set_title(title)
  ax.plot(itrs,means, color,  alpha=1.0)
  ax.fill_between(itrs,means+stds, means-stds, facecolor=color, alpha=0.5)
  fig.savefig(filepath)
  fig.clear()
  plt.close(fig)
  

itrs, means,stds = np.load('reward_itrs_ip.npy'),np.load('reward_means_ip.npy'),np.load('reward_stds_ip.npy')

save_fig(itrs,means,stds,'./graph_HalfCheetah.png','HalfCheetah','iteration','avg_reward')