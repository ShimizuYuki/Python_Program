# coding:utf-8
import numpy as np
from sklearn.externals import joblib
from chainer import cuda, Variable, FunctionSet,optimizers,Chain,serializers
import chainer.functions  as F

import time
import datetime
import os

import CNN
#np.random.seed(0)


def unpickleImg():
  print u"unpickle開始"
  nL =["color_x","s_labelset","gray_x","x_train","x_test","y_train","y_test"]
  unpickle_num = len(nL)
  aL = []
  for i in range(unpickle_num):
    filename = nL[i]+'.pkl'
    fo = open('./Dataset/'+Hozondir+'/'+filename,'rb')
    d=joblib.load(fo)
    fo.close()
    aL.append(d)
    print nL[i]+u"のunpickle完了"
    print nL[i]+".shape : ",aL[i].shape
  print u"全てのunpickle完了"
  print "----------------------------------------------"
  return aL


if __name__ == '__main__':
  StartTime = time.clock()
  # 学習の繰り返し回数
  n_epoch   = 100
  # 確率的勾配降下法で学習させる際の１回分のバッチサイズ
  batchsize = 10

  N_train = 60000
  N_test = 10000
  #-----データロード-------------------------------------
  color_x,s_labelset,gray_x,x_train,x_test,y_train,y_test=unpickleImg()
  color_x = color_x.astype(np.float32)
  gray_x = gray_x.astype(np.float32)
  x_train = x_train.astype(np.float32)
  x_test = x_test.astype(np.float32)
  y_train = y_train.astype(np.int32)
  y_test = y_test.astype(np.int32)
  x_train = x_train[0:N_train]
  x_test  = x_test[0:N_test]
  y_train = y_train[0:N_train]
  y_test  = y_test[0:N_test]
  print u"実際にNNに突っ込む訓練,テストデータのShapes"
  print "x_train.shape : ",x_train.shape
  print "x_test.shape : ",x_test.shape
  print "y_train.shape : ",y_train.shape
  print "y_test.shape : ",y_test.shape
  print "----------------------------------------------"
  #------NN構築----------------------------------------------------------------
  CNN = CNN.cnn()
  CNN.set_Model_Opti()
  #-----トレーニング-----------------------------------------------------------
  #CNN.train_loop(n_epoch,batchsize,x_train,y_train,x_test,y_test)
  N_train = len(x_train)
  N_test = len(x_test)
  N = N_train+N_test
  train_loss = []
  train_acc  = []
  test_loss = []
  test_acc  = []
  # Learning loop
  for epoch in xrange(1, n_epoch+1):
    '''
    訓練
    '''
    sum_trainloss = 0
    sum_trainacc = 0
    # 0〜Nまでのデータをバッチサイズごとに使って学習
    for i in xrange(0, N_train, batchsize):
      x_batch = x_train[i:i+batchsize]
      y_batch = y_train[i:i+batchsize]
      # 勾配を初期化
      CNN.optimizer.zero_grads()
      # 順伝播させて誤差と精度を算出
      loss,acc = CNN.calc_loss(x_batch,y_batch)
      # 誤差逆伝播で勾配を計算
      loss.backward()
      CNN.optimizer.update()
      train_loss.append(loss.data)
      train_acc.append(acc.data)
      sum_trainloss += loss.data * batchsize
      sum_trainacc  += acc.data  * batchsize
      print 'epoch:%d'%(epoch)+u"のTrain %f"%((i+batchsize)*100.0/N_train)+u"% 終了"
    mean_trainloss = sum_trainloss / N_train
    mean_trainacc = sum_trainacc / N_train
    print "----------------------------------------------"
    #モデルの保存
    CNN.save_model()
    '''
    テスト
    '''
    #テストデータで誤差と正解精度を算出
    sum_testloss = 0
    sum_testacc     = 0
    for i in xrange(0, N_test, batchsize):
      x_batch = x_test[i:i+batchsize]
      y_batch = y_test[i:i+batchsize]
      # 順伝播させて誤差と精度を算出
      loss,acc = CNN.calc_loss(x_batch,y_batch,train=False)
      test_loss.append(loss.data)
      test_acc.append(acc.data)
      sum_testloss += loss.data * batchsize
      sum_testacc  += acc.data * batchsize
      print 'epoch:%d'%(epoch)+u"のTest %f"%((i+batchsize)*100.0/N_test)+u"% 終了"
      mean_testloss = sum_testloss / N_test
      mean_testacc  = sum_testacc  / N_test
      print "----------------------------------------------"
      print "epoch ",epoch,u"終了"
      # 訓練データの誤差と、正解精度を表示
      print 'train mean loss=%f'%(mean_trainloss),\
            'train mean acc=%f'%(mean_trainacc)
      # テストデータの誤差と、正解精度を表示
      print 'test mean loss=%f'%(mean_testloss),\
            'test mean acc=%f'%(mean_testacc)
      print ('Time = %f'%(time.clock()-StartTime))
      print "----------------------------------------------"
