#coding:utf-8
#%matplotlib nbagg
#%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import datetime
import cv2

#np.random.seed(0)
#-------------------------------------------------------------------------
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
#-------------------------------------------------------------------------
def to_plot(img):
  return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
def to_grayscale(img):
  grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  return grayed
#-------------------------------------------------------------------------
def draw_imgSet(imgset,title_list,tate=5,yoko=4):
  plt.figure(figsize=(10,10))
  #r = np.random.permutation(len(tate*yoko))
  for idx in xrange(tate*yoko):
    #if random==True:
      #img = imgset[r[idx]]
      #title = title_list[r[idx]]
    #ランダムに画像をピックアップしない場合
    #elif random==False:
    img = imgset[idx]
    title = title_list[idx]
    plt.subplot(tate, yoko, idx+1)
    draw_img(img,title)
def draw_img(img,title):
  if np.max(img) <= 1.0:
    img*=255
  img=img.astype(np.uint8)
  plt.title(title,size=10)
  plt.tick_params(labelbottom="off")
  plt.tick_params(labelleft="off")
  if len(img.shape)==3 :
    if img.shape[0]==1:
      h,w=img.shape[1:]
      img=img.reshape(h,w)
      draw_img(img,title)
    elif img.shape[0]==3:
      img = img.transpose(1,2,0)
      draw_img(img,title)
    else:
      plt.imshow(to_plot(img))
  else:
      plt.gray()
      plt.imshow(img)
#-------------------------------------------------------------------------



if __name__ == '__main__':
  a =2
