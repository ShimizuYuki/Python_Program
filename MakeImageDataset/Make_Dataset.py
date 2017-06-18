# coding:utf-8
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
#--------------------------------------------------------------------------
def to_plot(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
def to_grayscale(img):
    grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return grayed
#--------------------------------------------------------------------------
def makeDataList(N_train, N_test):
    img_path = './RowImage/'
    #img_path = '/Users/okuyamatakashi/pyworks/MakeImageDataset/RowImage/'
    List = os.listdir(img_path)
    Folderlist = [f for f in List if os.path.isdir(os.path.join(img_path, f))]
    class_num = len(Folderlist)
    train_data_list = []
    train_label_list = []
    test_data_list = []
    test_label_list = []
    for i in xrange(class_num):
        print Folderlist[i], u'フォルダから画像読み込み中'
        #file_list = os.listdir('/RowImage/'+Folderlist[i]+'/')
        file_list = os.listdir(img_path+Folderlist[i]+'/')
        #random.shuffle(file_list)
        for File in file_list:
            if '.jpg' not in File and '.jpeg' not in File and '.png' not in File and '.JPG' not in File:
                file_list.remove(File)

        if len(file_list) < N_train:
            print "%sフォルダ内の画像数が指定された訓練データ数(%d枚)より少ないです"%(Folderlist[i], N_train)
            sys.exit()
        if len(file_list) < N_test:
            print "%sフォルダ内の画像数が指定されたテストデータ数(%d枚)より少ないです"%(Folderlist[i], N_test)
            sys.exit()

        train_img_list = []
        for j in range(N_train):
            img_name = file_list[j]
            print img_name
            img = cv2.imread(img_path + Folderlist[i] + '/' + img_name)
            train_img_list.append(img)

        test_img_list = []
        for j in range(N_train, N_train+N_test):
            img_name = file_list[j]
            #print img_name
            img = cv2.imread(img_path + Folderlist[i] + '/' + img_name)
            test_img_list.append(img)

        print Folderlist[i], u'フォルダから'
        print len(train_img_list), u'枚の訓練データ'
        print len(test_img_list), u'枚のテストデータ\n読み込み完了'
        print "----------------------------------------------"
        train_label = np.zeros((len(train_img_list))) + i
        test_label = np.zeros((len(test_img_list))) + i

        train_data_list.append(train_img_list)
        test_data_list.append(test_img_list)
        train_label_list.append(train_label)
        test_label_list.append(test_label)

    print u"全フォルダー内の画像読み込み完了"
    print "len(train_data_list) : ", len(train_data_list)
    print "len(test_data_list) : ", len(test_data_list)
    print "----------------------------------------------"
    return train_data_list, train_label_list, test_data_list, test_label_list, class_num

def gattaiResize(data_list, label_list, h, w):
    class_num = len(data_list)
    for i in xrange(class_num):
        imgset = [0] * len(data_list[i])
        imgset[:] = data_list[i][:]
        label = label_list[i]
        for j in xrange(len(imgset)):
            imgset[j] = cv2.resize(imgset[j], (w, h))
        imgset = np.array(imgset)
        if i == 0:
            dataset = imgset
            g_label = label
        elif i >= 1:
            dataset = np.vstack((dataset, imgset))
            g_label = np.hstack((g_label, label))
    print u"合体完了"
    return dataset, g_label

def dataSetMizumasi(dataset, N, class_num):
    print u"水増し開始，画像数は%dです" % (len(dataset))
    dataset = dataset.astype(np.uint8)
    i = 0
    for img in dataset:
        mizumasi = imgMizumasi(img)
        mizumasi = np.array(mizumasi)
        if i == 0:
            m_dataset = mizumasi
        elif i >= 1:
            m_dataset = np.vstack((m_dataset, mizumasi))
        i += 1
    print u"datasetの水増し完了"
    for j in xrange(class_num):
        m_label = np.zeros((N*72)) + j
        if j == 0:
            m_labelset = m_label
        elif j >= 1:
            m_labelset = np.hstack((m_labelset, m_label))
    print u"labelsetの水増し完了"
    return m_dataset, m_labelset
def imgMizumasi(img):
    mizumasi = [img]
    f_img, vf_img = Flip(mizumasi[0])
    mizumasi.append(f_img)
    # mizumasi.append(vf_img)
    repeat = len(mizumasi)
    for val in range(repeat):
        sp_img = SPnoise(mizumasi[val])
        mizumasi.append(sp_img)

    repeat = len(mizumasi)
    for val in range(repeat):
        n_img = noise(mizumasi[val])
        mizumasi.append(n_img)

    '''
    repeat = len(mizumasi)
    for val in range(repeat):
        b_img = blur(mizumasi[val])
        mizumasi.append(b_img)
    '''

    repeat = len(mizumasi)
    for val in range(repeat):
        h_img, l_img = gamma(mizumasi[val])
        mizumasi.append(h_img)
        mizumasi.append(l_img)

    repeat = len(mizumasi)
    for val in range(repeat):
        h_img, l_img = contrast(mizumasi[val])
        mizumasi.append(h_img)
        mizumasi.append(l_img)
    return mizumasi

def shuffle(dataset, label):
    print u"Shuffle開始"
    print "label[0:10] : ", label[0:10]
    N, h, w, c = dataset.shape
    s_dataset = np.zeros_like(dataset)
    s_dataset[:] = dataset[:]
    s_label = np.zeros_like(label)
    s_label[:] = label[:]

    s_dataset = s_dataset.reshape(N, h * w * c)
    if s_label.shape[0] != N:
        print u"ラベルとdatasetのサイズが不一致"
        sys.exit()
    dataset_label = np.column_stack((s_dataset, s_label))
    np.random.shuffle(dataset_label)
    s_dataset = dataset_label[:, :h * w * c]
    s_label = dataset_label[:, h * w * c:]

    s_dataset = s_dataset.reshape(N, h, w, c)

    s_label = s_label.ravel()
    print u"Shuffle完了"
    return s_dataset, s_label

def save(aL):
    print u"保存開始"
    nL = ["x_train", "x_test", "label_train", "label_test"]
    data_path = './Dataset/'
    #data_path = '/Users/okuyamatakashi/pyworks/MakeImageDataset/Dataset/'
    hiduke = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M")
    os.mkdir(data_path+hiduke)
    for i in xrange(len(aL)):
        filename = nL[i] + '.npy'
        np.save(data_path+hiduke+'/'+filename, aL[i])
        print nL[i] + u"のdump完了"
        print nL[i] + ".shape : ", aL[i].shape
    print u"全てのdump完了"
    print "----------------------------------------------"

#-------------------------------------------------------------------------
def make(N_train, N_test, h, w):
    print u"画像の読み込み"
    train_data_list,train_label_list,test_data_list,test_label_list,class_num\
    = makeDataList(N_train, N_test)
    for i in xrange(class_num):
        print "len(train_data_list[%d]) : %d" % (i, len(train_data_list[i]) )
        print "len(train_label_list[%d]) : %d" % (i, len(train_label_list[i]) )
        print "len(test_data_list[%d]) : %d" % (i, len(test_data_list[i]) )
        print "len(test_label_list[%d]) : %d" % (i, len(test_label_list[i]) )
        print "----------------------------------------------"
    print u"訓練データのresize、合体"
    train_data , train_label \
    = gattaiResize(train_data_list, train_label_list, h, w)
    print "train_data.shape : ", train_data.shape
    print "----------------------------------------------"
    print u"テストデータのresize、合体"
    test_data , test_label \
    = gattaiResize(test_data_list, test_label_list, h, w)
    print "test_data.shape : ", test_data.shape
    print "----------------------------------------------"
    print u"訓練データの水増し"
    m_train_data, m_train_label = dataSetMizumasi(train_data,N_train,class_num)
    print "m_train_data.shape : ", m_train_data.shape
    print "m_train_label.shape : ", m_train_label.shape
    print "----------------------------------------------"
    print u"テストデータの水増し"
    m_test_data, m_test_label = dataSetMizumasi(test_data,N_test,class_num)
    print "m_test_data.shape : ", m_test_data.shape
    print "m_test_label.shape : ", m_test_label.shape
    print "----------------------------------------------"
    print u"訓練データのシャフル"
    s_train_data, s_train_label = shuffle(m_train_data,m_train_label)
    print "s_train_label[0:10] : ", s_train_label[0:10]
    print "----------------------------------------------"
    print u"テストデータのシャフル"
    s_test_data, s_test_label = shuffle(m_test_data,m_test_label)
    print "s_test_label[0:10] : ", s_test_label[0:10]
    print "----------------------------------------------"
    print u"データの保存"
    s_train_data = s_train_data.astype(np.float32)
    s_test_data = s_test_data.astype(np.float32)
    s_train_label = s_train_label.astype(np.uint8)
    s_test_label = s_test_label.astype(np.uint8)
    aL = [s_train_data,s_test_data,s_train_label,s_test_label]
    save(aL)

if __name__ == '__main__':
    N_train = 3
    N_test = 2
    h = 100
    w = 100
    make(N_train, N_test, h, w)
