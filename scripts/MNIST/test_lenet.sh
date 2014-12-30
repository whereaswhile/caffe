#!/usr/bin/env sh

CAFFERT=/mnt/disk2/wang308/workspace/code/caffe-fork/
CAFFE=${CAFFERT}build/tools/caffe 
netfile=lenet_train_test.prototxt
mdlfile=./mdl/lenet_iter_10000.caffemodel
logfile=./log/test.log

$CAFFE test -model $netfile -weights $mdlfile -gpu 1 2>$logfile

