# donkey_cozmo
## Copyright
Copyright (C) RC30-popo

## Overview
ANKI社のAIロボットCOZMO(日本ではタカラトミーが販売: https://www.takaratomy.co.jp/products/cozmo/)をDONKEY CAR(https://www.donkeycar.com/)と同じ制御方法で自律走行させるアプリケーションです。j

COZMOは顔部分にQVGA解像度のカメラが搭載されており、このカメラで撮影した画像でコースの走り方を学習させます。
COZMOはpythonからリモート制御するためのSDK(https://anki.com/ja-jp/cozmo/SDK.html)が公開されており、COZMO SDKとchainer,OpenCVを組み合わせてCOZMOのカメラ画像を取り込み、畳込みニューラルネットー枠で前進、左転回、右転回の動作を判断しながら走行します。
特定の条件で一時停止することを学習させることもできる様にしました。

走行の様子はyoutubeで公開しています。
https://youtu.be/27JVWMNAD2o
https://youtu.be/BFyrgkgBzEM
https://youtu.be/RztJSE1sJ7A

自律走行に使用したスクリプト(python)の内容は下記のブログのエントリーで説明しています。
(掲載されているスクリプトは少し古いものです)
https://rc30-popo.hatenablog.com/entry/2019/06/27/233757

## SW Environment
以下の環境で開発、動作確認しています。
OS: Ubuntu 16.04 LTS
Python: Python3.6/Anaconda 4.6.14
Chainer: 4.3.1
NumPy: 1.16.4
CuPy:
  CuPy Version          : 4.5.0
  CUDA Root             : /usr/local/cuda-9.2
  CUDA Build Version    : 9020
  CUDA Driver Version   : 10010
  CUDA Runtime Version  : 9020
  cuDNN Build Version   : 7201
  cuDNN Version         : 7201
  NCCL Build Version    : 2213
OpenCV: 3.1.0
COZMO SDK: 1.4.10

畳込ニューラルネットワークの学習と推論にNVIDIAのGPUを使用しています。

## HW
以下のものが必要です。
・上記のSW環境を走行させる母艦(NVIDIAのGPUが載ったPC)
・COZMO本体
・COZMOアプリを動かすスマートフォン(androidもしくはiPhone)
COZMO SDKでCOZMOをリモート制御するためには母艦PCにUSB経由でスマートフォンを接続、スマートフォン上でANKI社が公開しているCOZMOアプリを起動する必要があります。詳細はCOZMO SDKのドキュメントを参照して下さい。
http://cozmosdk.anki.com/docs/


## pythonスクリプト
4つのスクリプトから構成されています。
1つは畳み込みニューラルネットワークの定義、残り3つが、データ取得、学習、推論実行(自律走行)になります。

### cozmo_dnn3.py
畳込ニューラルネットワークの定義ファイルです。
他のスクリプトにimportされます。

### donky_cozmo_recorder3.py
COZMO SDKを使用し、リモートでCOZMOを走行させるスクリプトです。
COZMOのカメラ画像と、人間の操作を紐付けてデータファイルに記録していきます。

#### 定数
DONKEY_COZMO_DATADIR = 'data3/'
画像と操作情報を記録するためのフォルダ名を指定します。
このフォルダ下にカメラ画像のファイルとdonkey_cozmo.datというデータファイルを作成されます。
donkey_cozmo.datの内容は下記の様なもので、画像ファイル名と操作ラベル(0〜3)が並んだCSVファイルです。
操作ラベルは0:前進,1: 左10度転回,2:右10度転回,3:何もしない(停止)です。

data3/img_00550.png,0
data3/img_00551.png,2
data3/img_00552.png,0
data3/img_00553.png,0
data3/img_00554.png,0
data3/img_00555.png,3

#### 使い方
スマートフォンを母艦に接続し、COZMOアプリを起動、SDKモードでCOZMOと接続後に
python donky_cozmo_recorder3.py
で起動します。
COZMOからキャプチャしたカメラ画像が表示されます。COZMOを見ながら下記の操作でCOZMOを走行させます。

'l'キー: 前進2cm
','キー: 右転回10度
'.'キー: 左転回10度
'h'キー: その場で停止
その他キー: その場で停止、何も記録しない
CTRL-C: 記録を終了

その他キーと'h'キーの違いは'h'キーはその時のカメラ画像と、「何もしない(3)」というラベルをデータファイルに記録します。
その他キーは何も記録せずに次の画像をキャプチャします。今写っている映像に対して何も操作を紐付けたく無い場合に使用します。

### cozmo_dnn_train3.py
畳込ニューラルネットワークに記録した画像と操作ラベルの紐付けを学習させるスクリプトです。

#### 定数
DONKEY_COZMO_DATAFILE = 'data3/donkey_cozmo.dat'
donky_cozmo_recorder3.pyで作成したデータファイルのファイル名を指定します。
このファイルから画像とラベルを読み出し、畳込ニューラルネットワークに教師有り学習をさせます。
DONKEY_COZMO_MDLFILE = 'donkey_cozmo_mdl3.npz'
畳込ニューラルネットワークの学習結果を記録するファイルです。

n_epoch = 10
学習時のエポック数です。学習結果を見ながら適当に調整して下さい。

#### 使い方
定数パートを適宜調整したら、
python cozmo_dnn_train3.py
を実行するのみです。メモリ不足等でエラーが出なければDONKEY_COZMO_MDLFILEで指定したファイルに学習結果が書き込まれます。

### donky_cozmo_run3.py
COZMOを自律走行させるスクリプトです。

#### 定数
DONKEY_COZMO_MDLFILE = 'donkey_cozmo_mdl3.npz'
cozmo_dnn_train3.pyで出力した学習結果ファイル名を指定します。

#### 使い方
donky_cozmo_recorder3.py同様にCOZMOアプリをSDKモードで起動したスマートフォンをUSB接続した後、
python donky_cozmo_run3.py
で起動します。
donky_cozmo_recorder3.pyと異なり、COZMOはカメラ画像の解析結果に従って自動的に走行します。
CTRL+Cで終了します。





