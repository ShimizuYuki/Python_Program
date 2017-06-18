# coding:utf-8
#%matplotlib nbagg
#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import datetime
import cv2


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
    plt.figure(figsize=(11,11))
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
