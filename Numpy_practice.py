# coding:utf-8
print 'Hello World'
# --------配列の保存、ロード----------------------------
import numpy as np

np.save('./hogehoge/a.npy', A)  # バイナリで保存

x = np.load('./NN_without_Framework/Dataset/3_class/x.npy')
print x.shape
print x[:10]
label = np.load('./NN_without_Framework/Dataset/3_class/label.npy')
print label.shape
print label[:10]


np.savetxt('./hogehoge/a.data', A)  # テキストで保存
D = np.loadtxt('./hogehoge/a.data')
print D
# --------色々-------------------------------------------
mean = 0
sigma = 1
n = np.random.normal(mean,sigma,(4,3,2))
print n.shape
print n
x = np.linspace(0,1,11)
-x
1 - x
x = np.random.randint(0,3,16).reshape(4,4)-1
x
x * (x > 0)

# --------Stack---------------------------------------------------
one = np.ones((4,4))
zero = np.zeros((4,4))
np.vstack((one,zero))
np.r_[one, zero]
np.hstack((one,zero))
np.c_[one, zero]
np.column_stack((one,zero))

one = np.ones((4))
zero = np.zeros((4))
np.vstack((one,zero))
np.r_[one, zero]
np.hstack((one,zero))
np.c_[one, zero]
np.column_stack((one,zero))

# --------Random---------------------------------------------------
print np.random.rand()      # 0〜1の乱数を1個生成
print np.random.rand(100)   # 0〜1の乱数を100個生成
print np.random.rand(10,10) # 0〜1の乱数で 10x10 の行列を生成
print np.random.rand(100) * 40 + 30 # 30〜70の乱数を100個生成
""" 標準正規分布。いわゆるガウシアン。標準正規分布ならば randn()
平均・分散を指定したい場合は normal() """
print np.random.randn()         # 標準正規分布 (平均0, 標準偏差1)
print np.random.randn(10)       # 標準正規分布を10個生成
print np.random.randn(10,10)    # 標準正規分布による 10x10 の行列
print np.random.normal(50,10)   # 平均50、標準偏差10の正規分布

print np.random.randint(100)          #  0〜99 の整数を1個生成
print np.random.randint(30,70)        # 30〜69 の整数を1個生成
print np.random.randint(0,2,20)     #  0〜99 の整数を20個生成
print np.random.randint(0,100,(5,5))  #  0〜99 の整数で5x5の行列を生成
print np.random.random_integers(100)  # 1〜100 の整数を1個生成
print np.random.random_integers(30,70)# 30〜70 の整数を1個生成

print np.random.city = ["Sapporo","Sendai","Tokyo","Nagoya","Kyoto","Osaka","Fukuoka"]
print np.random.choice(city)     # 1個をランダム抽出
print np.random.choice(city,10)  # 10個をランダム抽出（重複あり）
# 5個をランダム抽出（重複なし)
print np.random.choice(city,5,replace=False)

# 確率を重み付けする場合
weight = [0.1, 0.1, 0.3, 0.1, 0.1, 0.2, 0.1]
print np.random.choice(city, p=weight)


# ----------Axis------------------------------------------------
A = np.array((np.arange(16) + 1).reshape(4, 4))
A
print -np.mean(A)
print np.max(A)
print np.min(A)
# 縦
print np.mean(A, axis=0)
print np.max(A, axis=0)
print np.min(A, axis=0)
print np.argmax(A, axis=0)
# 横
B =  np.sum(A, axis=1)
print B
print np.mean(B)
print np.max(A, axis=1)
print np.min(A, axis=1)
print np.argmax(A, axis=1)



A = np.array((np.arange(12,dtype=np.float32) + 1).reshape(4, 3))
A
np.sum(A, axis=1)
(A.T/np.sum(A, axis=1)).T


# ----------空集合------------------------------------------------
arr = np.array([])
arr = np.append(arr, np.array([1, 2, 3]))
arr = np.append(arr, np.array([4, 5]))
arr

arr = np.empty((0,3), int)
arr = np.append(arr, np.array([[1, 2, 3]]), axis=0)
arr = np.append(arr, np.array([[4, 5, 0]]), axis=0)
arr

# ----------対数----------------------------------------
A
np.log(A)

# ---------四捨五入-------------------------------------
# 配列 [-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0] を作成
a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
# 四捨五入 (小数点以下 .5 以上は繰上げ、.5未満は切捨て)
np.round(a)
# 切り捨て (小数部分を取り除く)
np.trunc(a)
# 切り捨て (小さい側の整数に丸める)
np.floor(a)
# 切り上げ (大きい側の整数に丸める)
np.ceil(a)
# ゼロに近い側の整数に丸める
np.fix(a)
