# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib.animation as animation
import datetime
import cv2


print "Hello World"
# ----------------------画像出力---------------------


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
    plt.figure(figsize=(11, 11))
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
    minisetsize = tate * yoko
    for val in range(0, len(imgset), minisetsize):
        img_mini_set = list(imgset)[val:val + minisetsize]
        title_mini_set = list(titleset)[val:val + minisetsize]
        # tate=int(len(img_mini_set)/yoko)
        plt.figure(figsize=(10, 10))
        for idx in xrange(len(img_mini_set)):
            img = img_mini_set[idx]
            title = title_mini_set[idx]
            plt.subplot(tate, yoko, idx + 1)
            draw_img(img, title)
        plt.show()

def Make_animation(imgset):
    fig = plt.figure()
    ims = []
    for idx in xrange(len(imgset)):
        img = imgset[idx]
        if np.max(img) <= 1.0:
            img *= 255
        img = img.astype(np.uint8)
        if len(img.shape) == 3:
            im = [plt.imshow(to_plot(img), interpolation="none")]
        else:
            plt.gray()
            im = [plt.imshow(img, interpolation="none")]
        ims.append(im)
    ani = animation.ArtistAnimation(fig, ims)
    hiduke = datetime.datetime.now().strftime("%m-%d-%H-%M")
    print u"gifを保存中"
    ani.save("GIF/%s.gif" % (hiduke), writer="imagemagick")


# -----------------------等高線---------------------
x1 = np.arange(5)
x2 = np.arange(5)
X1, X2 = np.meshgrid(x1, x2)
X1
X2
Z1 = np.array([[24, 32, 12, 16, 21],
               [23, 24, 25, 26, 27],
               [43, 36, 32, 26, 25],
               [30, 32, 25, 21, 20],
               [20, 32, 23, 20, 14]])
plt.autumn()
plt.pcolor(X1, X2, Z1)

# 塗りつぶしあり
plt.hot()
plt.contourf(X1, X2, Z1, 1)

# 塗りつぶしなし
plt.contour(X1, X2, Z1, 1)

# cmapの種類
# autumn,bone,cool,copper,flag,gray,hot,hsv,jet,pink,prism,spring,summer,winter,spectral


def f(x, y):
    return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)


n = 5
x = np.linspace(-3, 3, n)
y = np.linspace(-3, 3, n)
X, Y = np.meshgrid(x, y)
X
# 10 -> 等高線の数
plt.contourf(X, Y, f(X, Y), 10, cmap=plt.cm.spectral)
plt.contour(X, Y, f(X, Y), cmap=plt.cm.spectral)





x = np.load('./Dataset/2_class/x.npy')
label = np.load('./Dataset/2_class/label.npy')
plt.scatter(x[:,0], x[:,1],c=label, cmap=plt.cm.Spectral)
