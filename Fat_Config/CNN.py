class cnn(object):
  def __init__(self):
    self.StartTime = time.clock()
  #(入力のチャネル数,出力のチャネル数,フィルタサイズ)
  def set_Model_Opti(self):
    N_output = 10
    self.model = FunctionSet(
      conv1=F.Convolution2D(1, h, 3, pad=1),
      conv2=F.Convolution2D(h, h, 3, pad=1),
      conv3=F.Convolution2D(h, h, 3, pad=1),
      conv4=F.Convolution2D(h, h, 3, pad=1),
      conv5=F.Convolution2D(h, h, 3, pad=1),
      l1=F.Linear(5250, 5000),
      l2=F.Linear(5000, N_output))
    #Optimaizerの設定
    self.optimizer = optimizers.Adam()
    self.optimizer.setup(self.model)
  def calc_loss(self,x_data, y_data, train=True):
    t = Variable(y_data)
    y = forward(x_data,train)
    loss = F.mean_squared_error(Q, Variable(target))
    return F.softmax_cross_entropy(y, t), F.accuracy(y, t)
  def forward(self,x_data, y_data, train=True):
    N,h,w=x.shape
    x = x.reshape(N,1,h,w)
    x = Variable(x_data)
    h = F.relu(self.model.conv1(x))
    h = F.max_pooling_2d(F.relu(self.model.conv2(h)), 2)
    h = F.relu(self.model.conv3(h))
    h = F.max_pooling_2d(F.relu(self.model.conv4(h)), 2)
    h = F.max_pooling_2d(F.relu(self.model.conv5(h)), 2)
    h = F.dropout(F.relu(self.model.l1(h)), train=train)
    y = self.model.l2(h)
    return y
  #--------------------------------------------------------------------------
  def save_model(self):
    modelname = "Fat_Config%s"%(datetime.datetime.now().strftime("%m-%d-%H-%M"))
    serializers.save_npz("./modelkeep/%s"%(modelname))
    print u"モデルを保存しました(Model_Name=%s)"%(modelname)
    print "----------------------------------------------"
  #--------------------------------------------------------------------------
  def draw_answerChack(self,gx,y,cx,cate,img_num,tate=4,yoko=4):
    print '-----DrawAnswerCheckStart------'
    if len(gx)<tate*yoko:
      print u"len(gx)<tate*yokoになってるよ"
      print  "len(gx)=", len(gx)
      print  "tate*yoko=", tate*yoko
      sys.exit()
    gx = gx[0:img_num]
    cx = cx[0:img_num]
    ans_array = y[0:img_num]
    #認識した画像のクラスのナンバーリスト
    recog_array = self.config(gx,ans_array)[0]
    title_list = []
    for i in range(img_num):
      ans   = cate[ans_array[i]]
      recog = cate[recog_array[i]]
      #title = "ans=%s,recog=%s"%(ans,recog)
      title = "This is %s"%(ans)
      title_list.append(title)
    print "----------------------------------------------"
    imgKanren.draw_imgSet(cx,title_list,img_num,tate,yoko)


  def answerCheck(self,gx,y,n_predict):
    print '-----AnswerCheckStart------'
    if len(gx)<n_predict:
      print u"len(gx)<n_predictになってるよ"
      print  "len(gx)=", len(gx)
      print  "n_predict=", n_predict
      sys.exit()
    gx = gx[0:n_predict]
    ans_array = y[0:n_predict]
    recog_array,acc = self.config(gx,ans_array)
    #認識した画像のクラスのナンバーリスト
    recog_array,acc = self.config(gx,ans_array)
    for idx in range(n_predict):
      print "ans:%d,predict:%d"%(ans_array[idx],recog_array[idx])
    print "acc of predict = %f"%(acc.data)
    print "----------------------------------------------"
    return recog_array

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
