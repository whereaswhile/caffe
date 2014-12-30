#!/usr/bin/env sh
# This script converts the mnist data into lmdb/leveldb format,
# depending on the value assigned to $BACKEND.

CAFFEROOT=/mnt/disk2/wang308/workspace/code/caffe-fork/
DATA_TR=${CAFFEROOT}data/mnist/train-images-idx3-ubyte
LABEL_TR=${CAFFEROOT}data/mnist/train-labels-idx1-ubyte
DATA_TS=${CAFFEROOT}data/mnist/t10k-images-idx3-ubyte
LABEL_TS=${CAFFEROOT}data/mnist/t10k-labels-idx1-ubyte
BIN=${CAFFEROOT}build/examples/mnist/convert_mnist_data.bin

#format
BACKEND="lmdb"

#output folders
OUT_TR=./data/mnist_train_${BACKEND}
OUT_TS=./data/mnist_test_${BACKEND}

rm -rf $OUT_TR
rm -rf $OUT_TS

echo "Creating ${BACKEND}..."
$BIN $DATA_TR $LABEL_TR $OUT_TR --backend=${BACKEND}
$BIN $DATA_TS $LABEL_TS $OUT_TS --backend=${BACKEND}
echo "Done."

