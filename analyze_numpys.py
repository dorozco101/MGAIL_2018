#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 21:22:24 2018

@author: nick
"""

'Hopper-v1_good_rewards.npy'
'Hopper-v1_med_rewards.npy'
'Hopper-v1_mixed_rewards.npy'

import numpy as np

rew = np.load('expert_numpys/Hopper-v1_good_rewards.npy')
done = np.load('expert_numpys/Hopper-v1_good_terminals.npy')
rews = []
idx_done = 1
rew1 = 0
count = 0
for i in range(done.shape[0]):
    done1 = done[i]
    rew1+=rew[i]
    if done1:
        print(rew1)
        rew1 = 0
        count+=1
print(count)
    #print(np.sum(rew[:idx_done]))
    #rews.append(np.sum(rew[:idx_done]))
    #done = done[idx_done+1:]
    #rew = rew[idx_done+1:]
    
