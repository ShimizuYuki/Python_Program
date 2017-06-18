# coding:utf-8
%matplotlib inline
print "Hello World"

import sys, codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout)
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys
import time
import datetime
import os
# np.random.seed(0)
def to_plot(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
def to_grayscale(img):
    grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return grayed
def draw_img(img, title):
    if np.max(img) <= 1.0:
        img *= 255
    img = img.astype(np.uint8)
    plt.title(title, size=10)
    plt.tick_params(labelbottom="off")
    plt.tick_params(labelleft="off")
    if len(img.shape) == 3:
        if img.shape[0] == 1:
            h, w = img.shape[1:]
            img = img.reshape(h, w)
            draw_img(img, title)
        elif img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
            draw_img(img, title)
        else:
            plt.imshow(to_plot(img))
    else:
        plt.gray()
        # img=img[::-1,:]
        plt.imshow(img)

def draw_img_set(imgset, titleset, yoko=8, tate=5):
    plt.figure(figsize=(9,9))
    for idx in xrange(len(imgset)):
        img = imgset[idx]
        title = titleset[idx]
        plt.subplot(tate, yoko, idx + 1)
        draw_img(img, title)

def load(Hozondir):
    print 'Load START'
    nL = ["x_train", "x_test", "label_train", "label_test"]
    #data_path = './Dataset/'+Hozondir+'/'
    data_path \
    ='/Users/okuyamatakashi/pyworks/MakeImageDataset/Dataset/'+Hozondir+'/'

    aL = []
    for i in range(len(nL)):
        filename = nL[i] + '.npy'
        d = np.load(data_path+filename)
        aL.append(d)
        print nL[i] + ": load complete"
        print nL[i] + ".shape : ", d.shape
    print "All load complete"
    print "----------------------------------------------"
    return aL

if __name__ == '__main__':
    Hozondir = '2017-05-10-14-17'
    x_train, x_test, label_train, label_test = load(Hozondir)
    draw_img_set(x_train[:25], label_train[:25], 5, 5)
    #plt.savefig('/Users/okuyamatakashi/desktop/result.jpg')
    #plt.show()
    draw_img_set(x_test[:25], label_test[:25], 5, 5)
    #plt.show()
