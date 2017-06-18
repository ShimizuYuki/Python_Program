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
    # model=['model_8_forward_MLP']
    # model=['model_8_forward_CNN','model_8_forward_CNN_burebure','model_8_forward_CNN_manyUP']
    model = ['model_8_forward_CNN_burebure']
    for model_name in model:
        print "----------------------------------------------"
        print u'MODEL : ', model_name
        print "----------------------------------------------"

        n_act = len(action_list)
        env = Forward_Kaihi.Forward_Enviroment(False)
        agent = Forward_Kaihi.Agent(n_act, False)
        agent.load_model(model_name)
        step = 1
        n_episode = 50
        meiroset = deque()
        shoutotu = 0
        kaihi = 0
        up = 0
        right = 0
        left = 0
        for i_episode in range(n_episode):
            st = env.reset()
            meiroset.append(st)
            ep_end = False
            while not ep_end:
                act_i_array = agent.get_action_test(st)
                act_i = act_i_array.argmax()
                #act_i = np.random.randint(0,n_act)
                #act_i = 0
                action = action_list[act_i]
                if action == 'left':
                    left += 1
                elif action == 'right':
                    right += 1
                elif action == 'up':
                    up += 1
                observation, reward, ep_end = env.step(action)
                st = observation
                meiroset.append(st)
            if reward < 0:
                shoutotu += 1
                # print u'衝突!!'
            else:
                kaihi += 1
                # print u'回避!!'
            # print "----------------------------------------------"

        print u"テスト回数 = %d" % (n_episode)
        print u"衝突回数 = %d , 回避回数 = %d" % (shoutotu, kaihi)
        kaihiritu = float(kaihi) / n_episode * 100
        print u"回避率 = " + str(kaihiritu) + "%"
        print u"UP :%d回 , RIGHT :%d回 , LEFT :%d回\n" % (up, right, left)
        # Visualize.Make_animation(list(meiroset)[0:50*8])
