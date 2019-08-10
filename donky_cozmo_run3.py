#!/usr/bin/env python3
# -*- Coding: utf-8 -*-
#  Donkey COZMO
#    autonomous run by deep learning result 
#  Copyright (C) RC30-popo,2019



import sys
import time
import termios
import os

import cv2
from PIL import Image
import numpy as np
import cozmo
from cozmo.util import degrees, distance_mm, speed_mmps

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

# Initial settings
DONKEY_COZMO_FACE_INIT_ANGLE = 0.0 # -25.0 degree to 44.5 degree
DONKEY_COZMO_TURN_ANGLE = 10.0
DONKEY_COZMO_FORWARD_DIST = 20.0
DONKEY_COZMO_SHORT_FORWARD_DIST = 5.0
DONKEY_COZMO_SPEED = 50.0

# COZMO actions
DONKEY_COZMO_ACTION_FORWARD = 0
DONKEY_COZMO_ACTION_LEFT_TURN = 1
DONKEY_COZMO_ACTION_RIGHT_TURN = 2
DONKEY_COZMO_ACTION_HOLD = 3

DONKEY_COZMO_ACTION_SHORT_FORWARD = 99

# Files
DONKEY_COZMO_MDLFILE = 'donkey_cozmo_mdl3.npz'

# Control Options
OPT_LOGICAL_CORRECTION1 = True
OPT_LOGICAL_CORRECTION1_THRES = 2

# COZMO WORD
DONKEY_COZMO_WORD_START = 'コズモ、きどうしました。 ぜんシステムせいじょうです。　はっしんします。'
DONKEY_COZMO_WORD_HOLD_START = 'ていしひょうしきはっけん。コズモ、ていしします。'
DONKEY_COZMO_WORD_HOLD_END = 'ていしひょうしきが、かいじょされました。 はっしんします'

img_x = 160
img_y = 120

# Decide action by Deep Learning prediction from COZMO's camera image
def cozmo_decide_action(cur_img):
    global img_x
    global img_y
    global model

    img = cv2.resize(cur_img,(img_x,img_y),interpolation=cv2.INTER_AREA)
    img_gs = img.flatten()
    x_data = []
    x_data.append(img_gs)
    x = xp.array(x_data,dtype=np.float32)
    x /= 255
    x = Variable(x)
    y = model.forward(x,ratio=0.2)
    print('y = ',y.data[0])
    res = xp.argmax(y.data[0])
#    print('res = ',res)
    return res

# Hold status check
hold_status = False
def check_hold_cnt(robot,action):
    global hold_status
    if action == DONKEY_COZMO_ACTION_HOLD:
        if hold_status == False:
            robot.say_text(DONKEY_COZMO_WORD_HOLD_START).wait_for_completed()
            hold_status = True
    else:
        if hold_status == True:
            robot.say_text(DONKEY_COZMO_WORD_HOLD_END).wait_for_completed()
            hold_status = False      


# Let COZMO go forward
def cozmo_go_forward(robot: cozmo.robot.Robot):
    global DONKEY_COZMO_FORWARD_DIST
    global DONKEY_COZMO_SPEED
    robot.drive_straight(distance_mm(DONKEY_COZMO_FORWARD_DIST), speed_mmps(DONKEY_COZMO_SPEED)).wait_for_completed()

def cozmo_go_short_forward(robot: cozmo.robot.Robot):
    global DONKEY_COZMO_FORWARD_DIST
    global DONKEY_COZMO_SPEED
    robot.drive_straight(distance_mm(DONKEY_COZMO_SHORT_FORWARD_DIST), speed_mmps(DONKEY_COZMO_SPEED)).wait_for_completed()

# Let COZMO turn left
def cozmo_left_turn(robot: cozmo.robot.Robot):
    global DONKEY_COZMO_TURN_ANGLE
    robot.turn_in_place(degrees(DONKEY_COZMO_TURN_ANGLE)).wait_for_completed()

# Let COZMO turn right
def cozmo_rigth_turn(robot: cozmo.robot.Robot):
    global DONKEY_COZMO_TURN_ANGLE
    robot.turn_in_place(degrees(-1.0 * DONKEY_COZMO_TURN_ANGLE)).wait_for_completed()

# Let COZMO hold
def cozmo_hold(robot: cozmo.robot.Robot):
    pass

# Control COZMO by camera image analysis
def cozmo_donkey_run(robot: cozmo.robot.Robot):
    global DONKEY_COZMO_FACE_INIT_ANGLE

    robot.camera.image_stream_enabled = True # Enable COZMO camera capture
    lift_action = robot.set_lift_height(0.0, in_parallel=False)
    lift_action.wait_for_completed()
    head_action = robot.set_head_angle(cozmo.util.Angle(degrees=DONKEY_COZMO_FACE_INIT_ANGLE),in_parallel=False)
    head_action.wait_for_completed()
    firstTime = True
    right_left_cnt = 0
    prev_action = None
    robot.say_text(DONKEY_COZMO_WORD_START).wait_for_completed()

    try:
        while True:
            duration_s = 0.1
            latest_image = robot.world.latest_image
            if latest_image != None:
                gray_image = latest_image.raw_image.convert('L')

                cv_image = np.asarray(gray_image)
#                print('cv_image shape = ',cv_image.shape)
                if firstTime:
                    height, width = cv_image.shape[:2]
                    print('*** Start captureing COZMO camera')
                    print('image height   = ',height)
                    print('image width    = ',width)
                    firstTime = False

                cv2.imshow('frame',cv_image)
                cv2.waitKey(1)

                decided_action = cozmo_decide_action(cv_image)
                action = decided_action
                check_hold_cnt(robot,action)
                if OPT_LOGICAL_CORRECTION1:
                    if (prev_action == DONKEY_COZMO_ACTION_LEFT_TURN and decided_action == DONKEY_COZMO_ACTION_RIGHT_TURN) or (prev_action == DONKEY_COZMO_ACTION_RIGHT_TURN and decided_action == DONKEY_COZMO_ACTION_LEFT_TURN):
                        right_left_cnt = right_left_cnt + 1
                    else:
                        right_left_cnt = 0

                    if right_left_cnt >= OPT_LOGICAL_CORRECTION1_THRES:
                        action = DONKEY_COZMO_ACTION_SHORT_FORWARD
                        right_left_cnt = 0

                if action == DONKEY_COZMO_ACTION_FORWARD:
                    print('*** Go FORWARD ***')
                    cozmo_go_forward(robot)
                elif action == DONKEY_COZMO_ACTION_LEFT_TURN:
                    print('*** Turn LEFT ***')
                    cozmo_left_turn(robot)
                elif action == DONKEY_COZMO_ACTION_RIGHT_TURN:
                    print('*** Turn RIGHT ***')
                    cozmo_rigth_turn(robot)
                elif action == DONKEY_COZMO_ACTION_HOLD:
                    print('*** HOLD ***')
                    cozmo_hold(robot)
                elif action == DONKEY_COZMO_ACTION_SHORT_FORWARD:
                    print('*** Go Short FORWARD ***')
                    cozmo_go_short_forward(robot)

                prev_action = decided_action                

                
#            time.sleep(duration_s)
    except KeyboardInterrupt:
        print('Keyboard Interrupt!!')
        print('Exit Cozmo SDK')
        cv2.destroyAllWindows()
        pass


# Initialize Neural Network
model = czCnn()
#optimizer = optimizers.Adam()
#optimizer.setup(model)
chainer.config.train = False
serializers.load_npz(DONKEY_COZMO_MDLFILE, model)
# GPU setup
gpu_device = 0
cuda.get_device(gpu_device).use()
model.to_gpu(gpu_device)

# Run main code
cozmo.run_program(cozmo_donkey_run)

