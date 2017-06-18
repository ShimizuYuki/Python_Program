# coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
from sklearn.externals import joblib
from chainer import cuda, Variable, FunctionSet,\
     optimizers,Chain,serializers
import chainer.functions  as F
import cv2

import sys
import time
import datetime
import os

import CNN
import MakeDataSet
import Draw
#np.random.seed(0)

def draw_config(gx,img_list,tate,yoko):
  img_num = tate*yoko
  if len(gx)<img_num:
    print u"len(gx)<img_num\nつまりConfigするデータ数<img_numになってるよ"
    print  "len(gx)=", len(gx)
    print  "img_num=", img_num
    sys.exit()
  gx = gx[0:img_num]
  img_list = img_list[0:img_num]
  #認識した画像のクラスのナンバーリスト
  config_number = CNN.config(gx)
  title_list = []
  for i in range(img_num):
    recog = cate[config_number[i]]
    title = "recog=%s"%(recog)
    title_list.append(title)
  Draw.draw_imgSet(img_list,title_list,tate,yoko)

def config(x):
  y = CNN.forward(x,False)
  #認識した画像のクラスのナンバーリスト
  config_number = recog_array.data.argmax(axis=1)
  return config_number


if __name__ == '__main__':
  cate=["Fat","NotFat"]
  model_name = "Fat_10-05-06-09"
  folder_list = ['TestData']
  print u"%sフォルダ内の画像をConfig"%(folder_list[0])
  print "----------------------------------------------"

  #Configフォルダの中の画像数によってimg_numは変わる
  img_num=4
  tate = 4
  yoko = 5

  data_list,label_list = MakeDataSet.makeDataList(folder_list)
  dataset,labelset = MakeDataSet.gattaiResize(data_list,label_list)
  color_x , gray_x = MakeDataSet.preparNNdata(dataset)

  CNN=CNN.cnn()
  CNN.set_Model_Opti()
  serializers.load_npz('./ModelKeep/'+model_name,CNN.model)

  imgCNN.draw_config(gray_x,data_list[0],cate,img_num,tate,yoko)
  plt.show()

  #--------------------------------------------------------------------
