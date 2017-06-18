# coding:utf-8
%matplotlib inline
print 'Hello World'
# --------------------配列の保存、ロード----------------------------------------
import numpy as np
import pickle
A = np.arange(100).reshape(10, 10)
print A
# Save
f = open('./hogehoge/a.pickle', 'w')
pickle.dump(A, f)
# この行がないと、'insecure string pickle'というエラーが出る
f.close()
# Load
f = open('./hogehoge/a.pickle', 'r')
B = pickle.load(f)
print B
# Numpy配列の保存とロード
np.save('./hogehoge/a.npy', A)  # バイナリで保存
C = np.load('./hogehoge/a.npy')
print C
np.savetxt('./hogehoge/a.data', A)  # テキストで保存
D = np.loadtxt('./hogehoge/a.data')
print D
# --------------------Numpy----------------------------------------
import numpy as np
x = np.linspace(0,1,10)
x
A = np.array((np.arange(16) + 1).reshape(4, 4))
A
print np.mean(A)
# 縦の平均
print np.mean(A, axis=0)
# 横の平均
print np.mean(A, axis=1)
mean = 0
sigma = 1
n = np.random.normal(mean,sigma,(4,3,2))
print n.shape
print n
# --------------------matlotlib----------------------------------------
x = np.linspace(0,1,10)
x
 x1 = np.arange(5)
 x2 = np.arange(5)
X1,X2 = np.meshgrid(x1 , x2)
X1
X2
Z = np.array([[24, 32, 12, 16, 21],
           [23, 24, 25, 26, 27],
           [43, 36, 32, 26, 25],
           [30, 32, 25, 21, 20],
           [20, 32, 23, 20, 14]])
z1 = np.ones((12))
z2 = np.zeros((13))
Z = np.hstack((z1,z2)).reshape(5,5)
Z

plt.autumn()
plt.pcolor(X1,X2,Z)
plt.hot()
# cmapの種類
    #autumn,bone,cool,copper,flag,gray,hot,hsv,jet,pink,prism,spring,summer,winter,spectral
# 塗りつぶしあり
plt.contourf(X1,X2,Z,1)
# 塗りつぶしなし
plt.contour(X1,X2,Z,1)
def f(x, y):
    return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 -y ** 2)
n = 5
x = np.linspace(-3, 3, n)
y = np.linspace(-3, 3, n)
X,Y = np.meshgrid(x, y)
X
#plt.axes([0.025, 0.025, 0.95, 0.95])

# 10 -> 等高線の数
plt.contourf(X, Y, f(X,Y),10,cmap=plt.cm.spectral)
plt.contour(X, Y, f(X,Y),cmap=plt.cm.pink)

# --------------------sklearn----------------------------------------
import sklearn.datasets
import matplotlib.pyplot as plt
%matplotlib inline
x , label = sklearn.datasets.make_classification(n_features=2, n_samples=300, n_redundant=0, n_informative=2,n_clusters_per_class=1, n_classes=3)
x.shape
label.shape

# 引数c -> 色を直接指定するか、色と対応づける配列を与える
plt.scatter(x[:, 0], x[:, 1],c=label,cmap=plt.cm.jet)
plt.scatter(x[:, 0], x[:, 1],c="#e70a3c",cmap=plt.cm.jet)
#np.save('x.npy', x)  # バイナリで保存
#np.save('label.npy', label)  # バイナリで保存


X, y = sklearn.datasets.make_moons(400, noise=0.2)
X.shape
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
np.save('x.npy', X)  # バイナリで保存
np.save('label.npy', y)  # バイナリで保存
# --------------------os----------------------------------------
zimport os
path = './Portfolio/MakeImageDataset/'
List = os.listdir(path)
print List
# File only
List_file = [f for f in List if os.path.isfile(os.path.join(path, f))]
print(List_file)
# Directry only
List_dir = [f for f in List if os.path.isdir(os.path.join(path, f))]
print(List_dir)
os.mkdir('./hogehoge')
# --------------------DateTime----------------------------------------
import datetime
today = datetime.date.today()
print today
print today.year
print today.month
print today.day
print today.isoformat()
import datetime
today_detail = datetime.datetime.today()
print today_detail.strftime("%Y-%m-%d-%H-%M-%S")
print today_detail
print today_detail.year
print today_detail.month
print today_detail.day
print today_detail.hour
print today_detail.minute
print today_detail.second
print today_detail.microsecond
print today_detail.isoformat()
# --------------------Time----------------------------------------
import time
start_time = time.time()
elapsed_time = time.time() - start_time
print elapsed_time
print u"実行時間 : %d日%d時間%d分%d秒%f"\
    % (elapsed_time / (60 * 60 * 24), elapsed_time / (60 * 60), elapsed_time / 60, elapsed_time % 60,
       elapsed_time - int(elapsed_time))
# --------------------sys----------------------------------------
import sys
a = 3
if a == 3:
    print "終了します"
    sys.exit()
print "Hello World"

# --------------------matlotlib,OpenCV----------------------------------------
import matplotlib.pyplot as plt
%matplotlib inline
import cv2
img = cv2.imread('/Users/okuyamatakashi/desktop/charhan.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
# plt.show()
plt.savefig('/Users/okuyamatakashi/desktop/result.jpg')
def Flip(img):
    new_img = np.zeros_like(img)
    new_img[:] = img[:]
    row, col, ch = new_img.shape
    hflip_img = cv2.flip(new_img, 1)
    vflip_img = cv2.flip(new_img, 0)
    return hflip_img, vflip_img
def SPnoise(img):
    new_img = np.zeros_like(img)
    new_img[:] = img[:]
    row, col, ch = new_img.shape

    s_vs_p = 0.5
    amount = 0.004
    sp_img = new_img.copy()

    # 塩モード
    num_salt = np.ceil(amount * new_img.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in new_img.shape]
    sp_img[coords[:-1]] = (255, 255, 255)

    # 胡椒モード
    num_pepper = np.ceil(amount * new_img.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in new_img.shape]
    sp_img[coords[:-1]] = (0, 0, 0)
    return sp_img
def noise(img):
    new_img = np.zeros_like(img)
    new_img[:] = img[:]
    row, col, ch = new_img.shape
    mean = 0
    sigma = 15
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    gauss_img = new_img + gauss
    gauss_img = gauss_img.astype(np.uint8)
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

    LUT_G1 = np.arange(256, dtype='uint8')
    LUT_G2 = np.arange(256, dtype='uint8')

    for i in range(256):
        LUT_G1[i] = 255 * pow(float(i) / 255, 1.0 / gamma1)
        LUT_G2[i] = 255 * pow(float(i) / 255, 1.0 / gamma2)
    # 変換
    new_img = new_img.astype(np.uint8)
    high_cont_img = cv2.LUT(new_img, LUT_G1)
    low_cont_img = cv2.LUT(new_img, LUT_G2)
    return high_cont_img, low_cont_img
def contrast(img):
    new_img = np.zeros_like(img)
    new_img[:] = img[:]
    # ルックアップテーブルの生成
    min_table = 50
    max_table = 205
    diff_table = max_table - min_table

    LUT_HC = np.arange(256, dtype='uint8')
    LUT_LC = np.arange(256, dtype='uint8')

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
    new_img = new_img.astype(np.uint8)
    high_cont_img = cv2.LUT(new_img, LUT_HC)
    low_cont_img = cv2.LUT(new_img, LUT_LC)
    return high_cont_img, low_cont_img
# 反転
plt.figure(figsize=(11,11))
plt.tick_params(labelbottom="off")
plt.tick_params(labelleft="off")
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
for i in range(2):
    plt.subplot(1, 3, i+2)
    plt.imshow(cv2.cvtColor(Flip(img)[i], cv2.COLOR_BGR2RGB))
# SPnoise
plt.figure(figsize=(11,11))
plt.tick_params(labelbottom="off")
plt.tick_params(labelleft="off")
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(SPnoise(img), cv2.COLOR_BGR2RGB))
# noise
plt.figure(figsize=(11,11))
plt.tick_params(labelbottom="off")
plt.tick_params(labelleft="off")
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(noise(img), cv2.COLOR_BGR2RGB))
# Blur
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(blur(img), cv2.COLOR_BGR2RGB))
# ガンマ変換
plt.figure(figsize=(11,11))
plt.tick_params(labelbottom="off")
plt.tick_params(labelleft="off")
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
for i in range(2):
    plt.subplot(1, 3, i+2)
    plt.imshow(cv2.cvtColor(gamma(img)[i], cv2.COLOR_BGR2RGB))
# コントラスト
plt.figure(figsize=(11,11))
plt.tick_params(labelbottom="off")
plt.tick_params(labelleft="off")
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
for i in range(2):
    plt.subplot(1, 3, i+2)
    plt.imshow(cv2.cvtColor(contrast(img)[i], cv2.COLOR_BGR2RGB))
