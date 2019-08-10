# -*- Coding: utf-8 -*-
#  Donkey COZMO
#    Convolution network training
#  Copyright (C) RC30-popo,2019


# OS
import os
# OpenCV
import cv2
# Numpy
import numpy as np
# Chainer
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Chain,optimizers,Variable
from chainer import serializers
from chainer import cuda
# random
import random

xp = cuda.cupy

from cozmo_dnn3 import czCnn


# Files
DONKEY_COZMO_DATAFILE = 'data3/donkey_cozmo.dat'
DONKEY_COZMO_MDLFILE = 'donkey_cozmo_mdl3.npz'
DONKEY_COZMO_OPTFILE = 'donkey_cozmo_opt.npz'

img_x = 160
img_y = 120

x_train_data = []
t_train_data = []

# Read data and convert to vector
with open(DONKEY_COZMO_DATAFILE, mode='r', encoding='utf-8') as f:
    for line in f:
        imgfile,label = line[:-1].split(',')
        print('imgfile = '+imgfile+',label = '+label)
        img = cv2.imread(imgfile)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,(img_x,img_y),interpolation=cv2.INTER_AREA)
#        print('img shape = ',img.shape)
        img_gs = img.flatten()

        x_train_data.append(img_gs)
        t_train_data.append(int(label))

x_train = xp.array(x_train_data,dtype=xp.float32)
x_train /= 255
t_train = xp.array(t_train_data,dtype=xp.int32)
total_datacount = len(x_train)
print('Total number of training data = ',total_datacount)

# Initialize Neural Network
model = czCnn()
chainer.config.train = True
optimizer = optimizers.Adam()
optimizer.setup(model)

# GPU setup
gpu_device = 0
cuda.get_device(gpu_device).use()
model.to_gpu(gpu_device)

# number of epochs
n_epoch = 15

# batch size
batch_size = 50

# Training main loop
for epoch in range(n_epoch):
    sum_loss = 0
    perm = np.random.permutation(total_datacount)

    for i in range(0,total_datacount,batch_size):
        if i + batch_size <= total_datacount:
            local_batch_size = batch_size
        else:
            local_batch_size = total_datacount - i
        x = Variable(x_train[perm[i:i+local_batch_size]])
        t = Variable(t_train[perm[i:i+local_batch_size]])

        y = model.forward(x,ratio=0.3)
        model.cleargrads()
        loss = F.softmax_cross_entropy(y, t)
        loss.backward()
        optimizer.update()
#        sum_loss += loss.data*local_batch_size
        sum_loss += float(cuda.to_cpu(loss.data)) * local_batch_size

    print("epoch: {0}, mean loss: {1}".format(epoch,sum_loss/total_datacount))

# Check accuracy
perm = np.random.permutation(total_datacount)
x = Variable(x_train[perm[0:100]])
t = t_train[perm[0:100]]
chainer.config.train = False
y = model.forward(x)
cnt = 0
for i in range(100):
    ti = t[i]
    yi = xp.argmax(y.data[i])
    print('[%d] t = %d, y = %d' % (i,ti,yi))
    if ti == yi:
        cnt += 1
# Display Result
# print(cnt)
print("accuracy: {}".format(cnt/(100)))



model.to_cpu()
serializers.save_npz(DONKEY_COZMO_MDLFILE, model)
# serializers.save_npz(DONKEY_COZMO_OPTFILE, optimizer)
