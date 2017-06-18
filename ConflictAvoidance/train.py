# coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
#from sklearn.externals import joblib
from chainer import cuda, Variable, FunctionSet,\
    optimizers, Chain, serializers
import chainer.functions as F
import cv2

#import sys
import datetime
#import os
from collections import deque
import copy

import Visualize

import Forward_Kaihi

if __name__ == '__main__':
    action_list = ['up', 'right', 'left']
    n_act = len(action_list)
    env = Forward_Kaihi.Forward_Enviroment2(True)
    agent = Forward_Kaihi.Agent(n_act, True)

    step = 1
    n_episode = 1333
    meiroset = deque()
    for i_episode in range(n_episode):
        st = env.reset()
        meiroset.append(st)
        ep_end = False

        sum_reward = 0

        while not ep_end:

            act_i_array = agent.get_action_train(st)

            act_i = act_i_array.argmax()
            action = action_list[act_i]
            observation, reward, ep_end = env.step(action)
            st_dash = observation
            agent.stock_experience(st, act_i, reward, st_dash, ep_end)
            step += 1
            st = observation
            meiroset.append(st)

            sum_reward += reward

        print "----------------------------------------------"
        if reward < 0:
            print u'衝突!!'
        else:
            print u'回避!!'
        print u'sum_reward = ', sum_reward
        print "----------------------------------------------"
        agent.gakushu(step)
    agent.save_model()
