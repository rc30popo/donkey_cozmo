#!/usr/bin/env python3
# -*- Coding: utf-8 -*-
#  Donkey COZMO
#    Data recorder for learning course
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

# Initial settings
DONKEY_COZMO_FACE_INIT_ANGLE = 0.0 # -25.0 degree to 44.5 degree
DONKEY_COZMO_TURN_ANGLE = 10.0
DONKEY_COZMO_FORWARD_DIST = 20.0
DONKEY_COZMO_SPEED = 50.0

# COZMO actions
DONKEY_COZMO_ACTION_FORWARD = 0
DONKEY_COZMO_ACTION_LEFT_TURN = 1
DONKEY_COZMO_ACTION_RIGHT_TURN = 2
DONKEY_COZMO_ACTION_HOLD = 3

# COZMO Key bindings
DONKEY_COZMO_KEY_FORWARD = 'l'
DONKEY_COZMO_KEY_LEFT_TURN = ','
DONKEY_COZMO_KEY_RIGTH_TURN = '.'
DONKEY_COZMO_KEY_HOLD = 'h'


# Files
DONKEY_COZMO_DATADIR = 'data3/'
DONKEY_COZMO_IMAGEPREFIX = 'img'
DONKEY_COZMO_DATAFILE = 'donkey_cozmo.dat'
DONKEY_COZMO_SEQNOFILE = 'donkey_cozmo.seq'

# Data Sequense number management
def getSeqNo(filepath):
    if os.path.exists(filepath):
        with open(filepath, mode='r') as f:
            for line in f:
                seqno = int(line)
                break
    else:
        seqno = 0
    return seqno

def saveSeqNo(filepath,seqno):
    with open(filepath, mode='w', encoding='utf-8') as f:
        f.write(str(seqno))


# Get key input from stdin without Enter key
#   This code comes from
#   https://qiita.com/tortuepin/items/9ede6ca603ddc74f91ba
def get_keyinput():
    #標準入力のファイルディスクリプタを取得
    fd = sys.stdin.fileno()

    #fdの端末属性をゲットする
    #oldとnewには同じものが入る。
    #newに変更を加えて、適応する
    #oldは、後で元に戻すため
    old = termios.tcgetattr(fd)
    new = termios.tcgetattr(fd)

    #new[3]はlflags
    #ICANON(カノニカルモードのフラグ)を外す
    new[3] &= ~termios.ICANON
    #ECHO(入力された文字を表示するか否かのフラグ)を外す
    new[3] &= ~termios.ECHO

    try:
        # 書き換えたnewをfdに適応する
        termios.tcsetattr(fd, termios.TCSANOW, new)
        # キーボードから入力を受ける。
        # lfalgsが書き換えられているので、エンターを押さなくても次に進む。echoもしない
        ch = sys.stdin.read(1)
        termios.tcflush(fd,termios.TCIFLUSH)
        return ch

    finally:
        # fdの属性を元に戻す
        # 具体的にはICANONとECHOが元に戻る
        termios.tcsetattr(fd, termios.TCSANOW, old)

# Let COZMO go forward
def cozmo_go_forward(robot: cozmo.robot.Robot):
    global DONKEY_COZMO_FORWARD_DIST
    global DONKEY_COZMO_SPEED
    robot.drive_straight(distance_mm(DONKEY_COZMO_FORWARD_DIST), speed_mmps(DONKEY_COZMO_SPEED)).wait_for_completed()

# Let COZMO turn left
def cozmo_left_turn(robot: cozmo.robot.Robot):
    global DONKEY_COZMO_TURN_ANGLE
    robot.turn_in_place(degrees(DONKEY_COZMO_TURN_ANGLE)).wait_for_completed()

# Let COZMO turn right
def cozmo_rigth_turn(robot: cozmo.robot.Robot):
    global DONKEY_COZMO_TURN_ANGLE
    robot.turn_in_place(degrees(-1.0 * DONKEY_COZMO_TURN_ANGLE)).wait_for_completed()

# Record COZMO camera image with action(go straight, left turn, right turn)
def cozmo_donkey_recorder(robot: cozmo.robot.Robot):
    global DONKEY_COZMO_FACE_INIT_ANGLE
    global DONKEY_COZMO_KEY_FORWARD
    global DONKEY_COZMO_KEY_LEFT_TURN
    global DONKEY_COZMO_KEY_RIGTH_TURN
    global DONKEY_COZMO_ACTION_FORWARD
    global DONKEY_COZMO_ACTION_LEFT_TURN
    global DONKEY_COZMO_ACTION_RIGHT_TURN
    global DONKEY_COZMO_DATADIR
    global DONKEY_COZMO_IMAGEPREFIX
    global DONKEY_COZMO_DATAFILE
    global DONKEY_COZMO_SEQNOFILE

    robot.camera.image_stream_enabled = True # Enable COZMO camera capture
    lift_action = robot.set_lift_height(0.0, in_parallel=False)
    lift_action.wait_for_completed()
    head_action = robot.set_head_angle(cozmo.util.Angle(degrees=DONKEY_COZMO_FACE_INIT_ANGLE),in_parallel=False)
    head_action.wait_for_completed()
    firstTime = True
    try:
        seqno = getSeqNo(DONKEY_COZMO_DATADIR + DONKEY_COZMO_SEQNOFILE)
        datf = open(DONKEY_COZMO_DATADIR + DONKEY_COZMO_DATAFILE,mode='a',encoding='utf-8')
        while True:
            duration_s = 0.1
            latest_image = robot.world.latest_image
            if latest_image != None:
                gray_image = latest_image.raw_image.convert('L')

                cv_image = np.asarray(gray_image)
                if firstTime:
                    height, width = cv_image.shape[:2]
                    print('*** Start captureing COZMO camera')
                    print('image height   = ',height)
                    print('image width    = ',width)
                    firstTime = False

                cv2.imshow('frame',cv_image)
                cv2.waitKey(1)

                ch = get_keyinput()
                current_action = None
                if ch == DONKEY_COZMO_KEY_FORWARD:
                    current_action = DONKEY_COZMO_ACTION_FORWARD
                    print('*** Go FORWARD ***')
                    cozmo_go_forward(robot)
                elif ch == DONKEY_COZMO_KEY_LEFT_TURN:
                    current_action = DONKEY_COZMO_ACTION_LEFT_TURN
                    print('*** Turn LEFT ***')
                    cozmo_left_turn(robot)
                elif ch == DONKEY_COZMO_KEY_RIGTH_TURN:
                    current_action = DONKEY_COZMO_ACTION_RIGHT_TURN
                    print('*** Turn RIGHT ***')
                    cozmo_rigth_turn(robot)
                elif ch == DONKEY_COZMO_KEY_HOLD:
                    current_action = DONKEY_COZMO_ACTION_HOLD
                    print('*** Hold ***')
                else:
                    current_action = None
                
                # save image and action data
                if current_action != None:
                    img_filename = DONKEY_COZMO_DATADIR + DONKEY_COZMO_IMAGEPREFIX + '_' + format(seqno,'05d') + '.png'
                    cv2.imwrite(img_filename,cv_image)
                    datf.write(img_filename + ',' + str(current_action) + '\n')
                    seqno = seqno + 1
#            time.sleep(duration_s)
    except KeyboardInterrupt:
        print('Keyboard Interrupt!!')
        print('Exit Cozmo SDK')
        saveSeqNo(DONKEY_COZMO_DATADIR + DONKEY_COZMO_SEQNOFILE,seqno)
        datf.close()
        cv2.destroyAllWindows()
        pass

cozmo.run_program(cozmo_donkey_recorder)

