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

class Forward_Enviroment():
  def __init__(self,Train):
    # parameters
    #self.name = os.path.splitext(os.path.basename(__file__))[0]
    self.screen_n_y = 8
    self.screen_n_x = 8
    self.player_length = 1
    #self.frame_rate = 5
    self.Train=Train
    if not Train:
      self.player_x = np.random.randint(self.screen_n_x - self.player_length)

  def reset(self):
    # reset player position
    self.player_y = self.screen_n_y - 1
    if self.Train:
      self.player_x = np.random.randint(self.screen_n_x - self.player_length)
    #self.player_x=7

    # reset goal position
    self.goal_y = 0
    self.goal_x = np.random.randint(self.screen_n_x)
    #self.goal_x = 2
    # reset other variables
    self.reward = 0
    self.ep_end = False
    self.draw()
    return self.screen

  def step(self, action):
    # update player position
    if action == 'left':
      # move left
      self.player_x = max(0, self.player_x - 1)
    elif action == 'right':
      # move right
      self.player_x = min(self.player_x + 1, self.screen_n_x - self.player_length)
    elif action == 'up':
      # do nothing
      pass
    # update goal position
    self.goal_y += 1
    # xlision detection
    self.reward = 0
    self.ep_end = False
    if self.goal_y == self.player_y:
        self.ep_end = True
        # self.player_x == self.goal_xと同じ意味
        if self.player_x == self.goal_x :
            # goal
            self.reward = 10
        else:
            # 衝突
            self.reward = -10
    self.draw()
    return self.screen, self.reward, self.ep_end
  def draw(self):
      # reset screen
      self.screen = np.zeros((self.screen_n_y, self.screen_n_x))
      kabe = np.ones((self.screen_n_x))/2.0
      self.screen[self.goal_y]=kabe
      # draw goal
      self.screen[self.goal_y, self.goal_x] = 0
      # draw player
      self.screen[self.player_y, self.player_x:self.player_x + self.player_length] = 1
  def reset_kokuhuku(self):
    # reset player position
    self.player_y = self.screen_n_y - 1
    if self.Train:
      self.player_x = np.random.randint(self.screen_n_x - self.player_length)
    self.player_x = 7

    # reset goal position
    self.goal_y = 0
    #self.goal_x = np.random.randint(self.screen_n_x)
    self.goal_x = np.random.randint(0,3)
    #self.goal_x = 0
    # reset other variables
    self.reward = 0
    self.ep_end = False
    self.draw()
    return self.screen

#直進を評価する環境
class Forward_Enviroment2():
  def __init__(self,Train):
    # parameters
    #self.name = os.path.splitext(os.path.basename(__file__))[0]
    self.screen_n_y = 8
    self.screen_n_x = 8
    self.player_length = 1
    #self.frame_rate = 5
    self.Train=Train
    if not Train:
      self.player_x = np.random.randint(self.screen_n_x - self.player_length)

  def reset(self):
    # reset player position
    self.player_y = self.screen_n_y - 1
    if self.Train:
      self.player_x = np.random.randint(self.screen_n_x - self.player_length)
    #self.player_x = 6
    # reset goal position
    self.goal_y = 0
    self.goal_x = np.random.randint(self.screen_n_x)
    # reset other variables
    self.reward = 0
    self.ep_end = False
    self.draw()
    return self.screen
  def step(self, action):
    # update goal position
    self.goal_y += 1
    # xlision detection
    self.reward = 0
    self.ep_end = False
    # update player position
    if action == 'left':
      # move left
      self.player_x = max(0, self.player_x - 1)
    elif action == 'right':
      # move right
      self.player_x = min(self.player_x + 1, self.screen_n_x - self.player_length)
    elif action == 'up':
      self.reward = 10
    if self.goal_y == self.player_y:
        self.ep_end = True
        # self.player_x == self.goal_xと同じ意味
        if self.player_x == self.goal_x :
            # goal
            self.reward += 100
        else:
            # 衝突
            self.reward = -100
    self.draw()
    return self.screen, self.reward, self.ep_end
  def draw(self):
      # reset screen
      self.screen = np.zeros((self.screen_n_y, self.screen_n_x))
      kabe = np.ones((self.screen_n_x))/2.0
      self.screen[self.goal_y]=kabe
      # draw goal
      self.screen[self.goal_y, self.goal_x] = 0
      # draw player
      self.screen[self.player_y, self.player_x:self.player_x + self.player_length] = 1

class DQN_NN(object):
  def __init__(self,n_act):
    self.N_input = 64
    N_output = n_act
    #N_unit = (self.N_input-1)*2
    N_unit = 64
    self.model = FunctionSet(
      l1=F.Linear(self.N_input,N_unit),
      #l2=F.Linear(N_unit, N_unit),
      #l3=F.Linear(N_unit, N_unit),
      l4=F.Linear(N_unit, N_output,initialW=np.zeros((N_output, N_unit), dtype=np.float32)))
  def Q_func(self,x):
    N,h,w=x.shape
    x=x.reshape(N,h*w)
    x = Variable(x)
    h = F.leaky_relu(self.model.l1(x))
    #h = F.leaky_relu(self.model.l2(h))
    #h = F.leaky_relu(self.model.l3(h))
    y = self.model.l4(h)
    return y
class DQN_CNN(object):
  def __init__(self,n_act):
    N_output = n_act
    self.model = FunctionSet(
      conv1=F.Convolution2D(1, 16, 3, pad=1),
      conv2=F.Convolution2D(16, 16, 3, pad=1),
      l1=F.Linear(256, 256),
      l2=F.Linear(256, N_output))
  def Q_func(self,x):
    N,h,w=x.shape
    x=x.reshape(N,1,h,w)
    x = Variable(x)
    h = F.relu(self.model.conv1(x))
    h = F.max_pooling_2d(F.relu(self.model.conv2(h)), 2)
    h = F.relu(self.model.l1(h))
    y = self.model.l2(h)
    return y

class Agent():
  def __init__(self,n_act,Train):
    self.n_act = n_act
    #self.NN = DQN_NN(n_act)
    self.NN = DQN_CNN(n_act)
    self.target_NN = copy.deepcopy(self.NN)
    self.optimizer = optimizers.Adam()
    self.optimizer.setup(self.NN.model)
    self.mem_size = 1000 # Experience Replayのために覚えておく経験の数
    self.memory = deque(maxlen=self.mem_size)
    self.gamma = 0.99 # 割引率
    self.batch_size = 32 # Experience Replayの際のミニバッチの大きさ
    #self.train_freq = 100 # ニューラルネットワークの学習間隔
    self.target_update_freq = 1 # ターゲットネットワークの同期間隔
    # ε-greedy
    self.epsilon_min = 0.1 # εの最小値
    if Train:
      self.epsilon = 1.0# εの初期値->小さいほどgreedy
      self.epsilon_decay = 0.001 # εの減衰値
    else:
      self.epsilon = 0 # εの初期値->小さいほどgreedy
      self.epsilon_decay = 0 # εの減衰値

    #self.exploration = 1000 # εを減衰し始めるまでのステップ数(今回はメモリーが貯まるまで)

  def get_action_test(self, st):
    #各行動の価値が含まれた(n_act,)のarrayを返す
    if np.random.rand() < self.epsilon:
      return np.random.rand(self.n_act)
      #return np.array([np.random.rand(), 0, 6.0/7])
    else:
      #-----NN用のデータ準備[float32->reshape(N,1,h,w)]
      st = np.array([st], dtype=np.float32)
      #st = st.reshape(1,9)
      Q = self.NN.Q_func(st)
      return Q.data[0]

  def get_action_train(self, st):
    #各行動の価値が含まれた(n_act,)のarrayを返す
    h , w = st.shape
    if st[h-1,w-1]==1:
      Max = np.max(st[0:h-1 , w-1])
      if Max == 0:
        return np.array([1,0,0])
      else:
        return np.array([0,0,1])
    elif st[h-1,0]==1:
      Max = np.max(st[0:h-1 , 0])
      if Max == 0:
        return np.array([1,0,0])
      else:
        return np.array([0,1,0])
    else:
      if np.random.rand() < self.epsilon:
        #return np.random.rand(self.n_act)
        #return np.array([np.random.rand(), 0, 6.0/7])
        return np.array([0,0,1])
      else:
        #-----NN用のデータ準備[float32->reshape(N,1,h,w)]
        st = np.array([st], dtype=np.float32)
        #st = st.reshape(1,9)
        Q = self.NN.Q_func(st)
        return Q.data[0]

  def stock_experience(self, st, act, r, st_dash, ep_end):
    self.memory.append((st, act, r, st_dash, ep_end))
  #-----学習-----
  def gakushu(self,step):
    self.experience_replay()
    print u"学習"
    if step % self.target_update_freq == 0:
      self.target_NN = copy.deepcopy(self.NN)

    if len(self.memory) >= self.mem_size:
      self.reduce_epsilon()
      print u"ε = ",self.epsilon

  def experience_replay(self):
    #シャッフル
    mem = np.random.permutation(list(self.memory))
    for i in xrange(0, len(mem), self.batch_size):
      batch = mem[i:i+self.batch_size]
      st_batch,act_batch,r_batch,st_dash_batch,ep_end_batch = [], [], [], [], []
      #print batch
      for j in xrange(len(batch)):
        st_batch.append(batch[j][0])
        act_batch.append(batch[j][1])
        r_batch.append(batch[j][2])
        st_dash_batch.append(batch[j][3])
        ep_end_batch.append(batch[j][4])
      # 勾配を初期化
      self.optimizer.zero_grads()
      # 順伝播させて誤差と精度を算出
      loss = self.calc_loss(st_batch,act_batch,r_batch,st_dash_batch,ep_end_batch)
      # 誤差逆伝播で勾配を計算
      loss.backward()
      self.optimizer.update()
  def calc_loss(self, st_batch,act_batch,r_batch,st_dash_batch,ep_end_batch):
    #-----NN用のデータ準備[float32->reshape(N,1,h,w)]
    st_batch = np.array(st_batch, dtype=np.float32)
    act_batch = np.array(act_batch, dtype=np.int8)
    r_batch = np.array(r_batch, dtype=np.float32)
    st_dash_batch = np.array(st_dash_batch, dtype=np.float32)
    ep_end_batch = np.array(ep_end_batch, dtype=np.bool)

    Q = self.NN.Q_func(st_batch)
    tmp = self.target_NN.Q_func(st_dash_batch)
    max_Q_dash = np.array(map(np.max,tmp.data), dtype=np.float32)
    #target = np.asanyarray(copy.deepcopy(Q.data), dtype=np.float32)
    target = np.copy(Q.data)
    for j in xrange(len(st_batch)):
      if ep_end_batch[j]:
        target[j, act_batch[j]] = r_batch[j]
      else:
        target[j, act_batch[j]] = r_batch[j] + (self.gamma * max_Q_dash[j])
    loss = F.mean_squared_error(Q, Variable(target))
    return loss
  def reduce_epsilon(self):
    if self.epsilon > self.epsilon_min:
      self.epsilon = max(self.epsilon_min ,self.epsilon - self.epsilon_decay)
  def save_model(self):
    modelname = "model_%s"%(datetime.datetime.now().strftime("%m-%d-%H-%M"))
    serializers.save_npz("./modelkeep/%s"%(modelname),self.NN.model)
    print u"モデルを保存しました(Model_Name=%s)"%(modelname)
    print "----------------------------------------------"
  def load_model(self,model_name):
    serializers.load_npz('./modelKeep/'+model_name,self.NN.model)
