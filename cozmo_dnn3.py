# -*- Coding: utf-8 -*-
#  Donkey COZMO
#    Convolution network definition
#  Copyright (C) RC30-popo,2019

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Chain, optimizers, Variable

# chainer.config.train = False

class czCnn(Chain):
    def __init__(self):
        super(czCnn, self).__init__(
            conv1 = L.Convolution2D(in_channels=1, out_channels=16, ksize=4, stride=1, pad=1),
            conv2 = L.Convolution2D(in_channels=16, out_channels=32, ksize=4, stride=1, pad=1),
            l0 = L.Linear(None,512),
            l1 = L.Linear(None,4),
            bncv1 = L.BatchNormalization(16),
            bncv2 = L.BatchNormalization(32),
            bn0 = L.BatchNormalization(512),
        )
    def forward(self, x,ratio=0.5):
        h = F.reshape(x,(len(x),1,160,120))
        h = F.max_pooling_2d(F.relu(self.bncv1(self.conv1(h))),2)
        h = F.max_pooling_2d(F.relu(self.bncv2(self.conv2(h))),2)
        h = F.dropout(F.relu(self.bn0(self.l0(h))),ratio=ratio)
        h = self.l1(h)
        return h
