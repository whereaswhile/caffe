#!/usr/bin/env sh

CAFFERT=/mnt/disk2/wang308/workspace/code/caffe-fork/
CAFFE=${CAFFERT}build/tools/caffe 
logfile=./log/train.log

$CAFFE train --solver=lenet_solver.prototxt -gpu 0 2>$logfile

