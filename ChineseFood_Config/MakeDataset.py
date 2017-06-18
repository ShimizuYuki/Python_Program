# coding:utf-8
import numpy as np
from sklearn.externals import joblib
import cv2

import sys
import os
import random

import imgKanren
import Dataload
#np.random.seed(0)
def Flip(img):
  new_img = np.zeros_like(img)
  new_img[:] = img[:]
  row,col,ch= new_img.shape
  hflip_img = cv2.flip(new_img, 1)
  vflip_img = cv2.flip(new_img, 0)
  return hflip_img,vflip_img
def SPnoise(img):
  new_img = np.zeros_like(img)
  new_img[:] = img[:]
  row,col,ch= new_img.shape

  s_vs_p = 0.5
  amount = 0.004
  sp_img = new_img.copy()

  # 塩モード
  num_salt = np.ceil(amount * new_img.size * s_vs_p)
  coords = [np.random.randint(0, i-1 , int(num_salt)) for i in new_img.shape]
  sp_img[coords[:-1]] = (255,255,255)

  # 胡椒モード
  num_pepper = np.ceil(amount* new_img.size * (1. - s_vs_p))
  coords = [np.random.randint(0, i-1 , int(num_pepper)) for i in new_img.shape]
  sp_img[coords[:-1]] = (0,0,0)
  return sp_img
def noise(img):
  new_img = np.zeros_like(img)
  new_img[:] = img[:]
  row,col,ch= new_img.shape
  mean = 0
  sigma = 15
  gauss = np.random.normal(mean,sigma,(row,col,ch))
  gauss = gauss.reshape(row,col,ch)
  gauss_img = new_img + gauss

  return gauss_img
def blur(img):
  new_img = np.zeros_like(img)
  new_img[:] = img[:]
  blured = cv2.blur(new_img, (10, 10))
  return blured
def gamma(img):
  new_img = np.zeros_like(img)
  new_img[:] = img[:]

  # ガンマ変換ルックアップテーブル
  gamma1 = 0.75
  gamma2 = 1.5

  LUT_G1 = np.arange(256, dtype = 'uint8' )
  LUT_G2 = np.arange(256, dtype = 'uint8' )

  for i in range(256):
    LUT_G1[i] = 255 * pow(float(i) / 255, 1.0 / gamma1)
    LUT_G2[i] = 255 * pow(float(i) / 255, 1.0 / gamma2)
  # 変換
  new_img=new_img.astype(np.uint8)
  high_cont_img = cv2.LUT(new_img, LUT_G1)
  low_cont_img = cv2.LUT(new_img, LUT_G2)
  return high_cont_img,low_cont_img
def contrast(img):
  new_img = np.zeros_like(img)
  new_img[:] = img[:]
  # ルックアップテーブルの生成
  min_table = 50
  max_table = 205
  diff_table = max_table - min_table

  LUT_HC = np.arange(256, dtype = 'uint8' )
  LUT_LC = np.arange(256, dtype = 'uint8' )

  # ハイコントラストLUT作成
  for i in range(0, min_table):
    LUT_HC[i] = 0
  for i in range(min_table, max_table):
    LUT_HC[i] = 255 * (i - min_table) / diff_table
  for i in range(max_table, 255):
    LUT_HC[i] = 255

  # ローコントラストLUT作成
  for i in range(256):
    LUT_LC[i] = min_table + i * (diff_table) / 255

  # 変換
  new_img=new_img.astype(np.uint8)
  high_cont_img = cv2.LUT(new_img, LUT_HC)
  low_cont_img = cv2.LUT(new_img, LUT_LC)
  return high_cont_img,low_cont_img
#-------------------------------------------------------------------------
#folder_listにある全てのフォルダから画像セットを作りラベルを付与し1つのデータリストに
def makeDataList(folder_list,read_num=100):
  class_num = len(folder_list)
  data_list = []
  label_list = []
  for i in xrange(class_num):
    imgset , label = makeImgSet(read_num,folder_list[i], i)
    data_list.append(imgset)
    label_list.append(label)
  print u"全フォルダー内の画像読み込み完了"
  print "len(data_list) : ",len(data_list)
  for i in xrange(class_num):
    print "len(data_list[%d]) : "%(i),len(data_list[i])
    print "len(label_list[%d]) : "%(i),len(label_list[i])
  print "data_list[0][0].shape : ",data_list[0][0].shape
  print "----------------------------------------------"
  return data_list , label_list
#1つのフォルダーから画像セットを作りラベルを付与する
def makeImgSet(read_num,FolderName,label):
  try :
    imgset = []
    img_list = os.listdir('./imgFolder/'+FolderName+'/')
    random.shuffle(img_list)
  except OSError:
    print "\n\n"+FolderName+u"フォルダが見つかりません!!!\n\n"
    raise

  #print img_list
  print FolderName,u'フォルダ'
  val = 0
  for img_name in (img_list):
    if '.jpg' in img_name or '.jpeg' in img_name \
        or '.png' in img_name or '.JPG' in img_name:
      img =cv2.imread('./imgFolder/'+FolderName+'/'+img_name)
      val+=1
      imgset.append(img)
      if val==read_num:
        break
  print len(imgset),u'枚画像読み込み完了'
  print "----------------------------------------------"
  labelset=np.zeros((len(imgset)))+label
  return imgset,labelset
#-------------------------------------------------------------------------
def gattaiResize(data_list,label_list,h=50,w=50):
  class_num = len(data_list)
  for i in xrange(class_num):
    imgset=[0]*len(data_list[i])
    imgset[:]=data_list[i][:]
    label = label_list[i]
    for j in xrange(len(imgset)):
      imgset[j]=cv2.resize(imgset[j],(w,h))
    imgset = np.array(imgset)
    print i+1,u"つ目のフォルダの画像resize完了"
    print "imgset.shape : ",imgset.shape
    print "----------------------------------------------"
    if i== 0:
      dataset = imgset
      g_label = label
    elif i >= 1:
      dataset = np.vstack((dataset,imgset))
      g_label = np.hstack((g_label,label))
  print u"全ての画像gattai完了"
  print "dataset.shape : ",dataset.shape
  print "----------------------------------------------"
  return dataset,g_label
#-------------------------------------------------------------------------
#resizeされたデータを水増し
def dataSetMizumasi(dataset,N_array):
  print u"datasetの水増し開始，画像数は%dです"%(len(dataset))
  dataset = dataset.astype(np.uint8)
  i=0
  for img in dataset:
    mizumasi72 = imgMizumasi(img)
    mizumasi72 = np.array(mizumasi72)
    if i== 0:
      m_dataset = mizumasi72
    elif i >= 1:
      m_dataset = np.vstack((m_dataset,mizumasi72))
    i+=1
    print i,u"枚目の水増し完了"
  print u"datasetの水増し完了"
  for j in xrange(len(N_array)):
    m_label=np.zeros((N_array[j]))+j
    if j== 0:
      m_labelset = m_label
    elif j >= 1:
      m_labelset = np.hstack((m_labelset,m_label))
  print u"labelsetの水増し完了"
  print "m_dataset.shape : ",m_dataset.shape
  print "m_labelset.shape : ",m_labelset.shape
  print "----------------------------------------------"
  return m_dataset , m_labelset
def imgMizumasi(img):
  mizumasi = [img]
  f_img , vf_img=  Flip(mizumasi[0])
  mizumasi.append(f_img)
  #mizumasi.append(vf_img)
  repeat = len(mizumasi)
  for val in range(repeat):
    sp_img =  SPnoise(mizumasi[val])
    mizumasi.append(sp_img)
  repeat = len(mizumasi)
  for val in range(repeat):
    n_img =  noise(mizumasi[val])
    mizumasi.append(n_img)
  repeat = len(mizumasi)
  for val in range(repeat):
    h_img , l_img = gamma(mizumasi[val])
    mizumasi.append(h_img)
    mizumasi.append(l_img)
  repeat = len(mizumasi)
  for val in range(repeat):
    h_img , l_img = contrast(mizumasi[val])
    mizumasi.append(h_img)
    mizumasi.append(l_img)
  '''
  repeat = len(mizumasi)
  for val in range(repeat):
    b_img =  blur(mizumasi[val])
    mizumasi.append(b_img)
  '''
  return mizumasi
#------------------------------------------------------------------------
def shuffle(dataset,g_label):
  print u"Shuffle開始"
  print "g_label[0:10] : ",g_label[0:10]
  N,h,w,c = dataset.shape
  s_dataset = np.zeros_like(dataset)
  s_dataset[:]=dataset[:]
  s_label = np.zeros_like(g_label)
  s_label[:]=g_label[:]
  s_dataset = s_dataset.reshape(N,h*w*c)
  if s_label.shape[0] != N:
    print u"ラベルとdatasetのサイズが不一致"
  dataset_label = np.column_stack((s_dataset,s_label))
  np.random.shuffle(dataset_label)
  s_dataset = dataset_label[:,:h*w*c]
  s_label = dataset_label[:,h*w*c:]
  s_dataset = s_dataset.reshape(N,h,w,c)
  s_label = s_label.ravel()
  print u"Shuffle完了"
  print "s_label[0:10] : ",s_label[0:10]
  print "----------------------------------------------"
  return s_dataset,s_label
#------------------------------------------------------------------------
def preparNNdata(dataset):
  N,h,w = dataset.shape[0:3]
  #プロット用のカラー画像準備
  color_x = np.zeros_like(dataset)
  color_x[:] = dataset[:]
  color_x=color_x.astype(np.uint8)
  #グレースケール画像準備
  gray_x=np.zeros((N,h,w))
  for i in xrange(N):
    gray_x[i]=cv2.cvtColor(color_x[i],cv2.COLOR_BGR2GRAY)
  gray_x  /= 255.0
  color_x  /= 255.0
  print u"NNに入れるデータ準備完了"
  print "color_x.shape : ",color_x.shape
  print "gray_x.shape : ",gray_x.shape
  print "----------------------------------------------"
  return color_x , gray_x
#------------------------------------------------------------------------
def TrainTestBunkatu(N,N_train,x,label):
  x_train, x_test , x_notUse = np.split(x, [N_train,N])
  y_train, y_test , y_notUse = np.split(label, [N_train,N])
  print u"トレーニングとテストデータ準備完了"
  print "x_train.shape : ",x_train.shape
  print "x_test.shape : ",x_test.shape
  print "y_train.shape : ",y_train.shape
  print "y_test.shape : ",y_test.shape
  print "----------------------------------------------"
  return x_train,x_test,y_train,y_test
#------------------------------------------------------------------------
def dump(aL,nL):
  Hozondir = 'Dataset'
  print u"dump開始"
  dump_num = len(aL)
  if Hozondir not in os.listdir('./') :
    os.mkdir('./'+Hozondir)
  for i in xrange(dump_num):
    filename = nL[i]+'.pkl'
    fo = open('./'+Hozondir+'/'+filename,'wb')
    joblib.dump(aL[i], fo )
    fo.flush()
    fo.close()
    print nL[i]+u"のdump完了"
    print nL[i]+".shape : ",aL[i].shape
  print u"全てのdump完了"
  print "----------------------------------------------"

if __name__ == '__main__':
  folder_list=["Annindo-fu","Charhan","Gyouza","Gomadango","Ra-men",\
  "Ma-bo-do-fu","Nikuman","Ebimayo","Tinjaoro-su","Tenshinhan"]
  cate=folder_list
  #1つのフォルダから何枚の画像を読み込むか(フォルダ内の画像数より大きいとフォルダ内の画像すべてが読み込まれる)
  read_num = 100
  N = read_num*len(folder_list)*72
  N_train = (N*7)/10
  N_test = N-N_train

  data_list,label_list = makeDataList(folder_list,read_num)

  N_array = []
  for i in xrange(len(N_array)):
    N_array.append(len(data_list[i]))
  N_array = N_array*72

  dataset,labelset = gattaiResize(data_list,label_list)
  dataset,labelset = dataSetMizumasi(dataset,N_array)
  s_dataset,s_labelset = shuffle(dataset,labelset)
  color_x , gray_x = preparNNdata(s_dataset)

  if N > len(color_x):
    print 'read_num > 実際にロードされた画像数'
    N=len(color_x)
    N_train = (N*7)/10

  x_train,x_test,y_train,y_test=TrainTestBunkatu(N,N_train,gray_x,s_labelset)
  nL =["color_x","s_labelset","gray_x","x_train","x_test","y_train","y_test"]
  aL = [color_x, s_labelset, gray_x,  x_train , x_test , y_train , y_test]
  dump(aL,nL)
