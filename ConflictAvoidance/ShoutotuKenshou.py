# coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
#from sklearn.externals import joblib
from chainer import cuda, Variable, FunctionSet,\
   optimizers,Chain,serializers
import chainer.functions  as F
import cv2

#import sys
import datetime
#import os
from collections import deque
import copy

import Visualize

import Forward_Kaihi
if __name__ == '__main__':
  action_list=['up','right','left']
  #model_name = 'model_8_forward_CNN'
  #shoutotu_kenshyou(action_list,model_name)
  def shoutotu_kenshyou(action_list,model_name):
    n_act = len(action_list)
    env=Forward_Kaihi.Forward_Enviroment2(False)
    agent=Forward_Kaihi.Agent(n_act,False)
    agent.load_model(model_name)
    step=1
    shoutotuset=deque()
    shoutotu=0
    kaihi=0
    up=0
    right=0
    left=0
    while True:
      st=env.reset()
      meiroset=deque()
      meiroset.append(st)
      ep_end = False
      while not ep_end:
        act_i_array=agent.get_action_test(st)
        act_i= act_i_array.argmax()
        action=action_list[act_i]

        if action == 'left':
          left+=1
        elif action == 'right':
          right+=1
        elif action == 'up':
          up+=1
        observation,reward,ep_end=env.step(action)
        st = observation
        meiroset.append(st)
      print "----------------------------------------------"
      if reward < 0:
        shoutotu+=1
        print u'衝突!!'
        shoutotuset.extend(meiroset)
      else:
        kaihi+=1
        print u'回避!!'
      print "----------------------------------------------"
      if shoutotu == 50 or len(meiroset)==10000:
        break
    print u"テスト回数 = %d"%(shoutotu+kaihi)
    print u"衝突回数 = %d , 回避回数 = %d"%(shoutotu,kaihi)
    kaihiritu = float(kaihi)/(shoutotu+kaihi)
    print u"回避率 = "+str(kaihiritu)+"%"
    print u"LEFT :%d回 , UP :%d回 , RIGHT :%d回"%(left,right,up)
    Visualize.Make_animation(shoutotuset)
