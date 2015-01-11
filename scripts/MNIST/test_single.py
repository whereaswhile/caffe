#! /usr/bin/env python

import numpy as np
import sys
import caffe
import timeit
import time

CAFFERT = '/mnt/disk2/wang308/workspace/code/caffe-fork'
MODEL_FILE = 'lenet_train_test_deploy.prototxt'
PRETRAINED = './mdl/lenet_iter_10000.caffemodel'
IMAGE_FILE = CAFFERT+'/examples/images/cat.jpg'

net = caffe.Classifier(MODEL_FILE, PRETRAINED)
print 'Network {} loaded'.format(PRETRAINED)
print 'net inputs:', net.inputs
print 'net input dim:', net.image_dims, net.crop_dims
sz=net.image_dims

net.set_phase_test()
net.set_mode_gpu()

input_image = caffe.io.load_image(IMAGE_FILE)
input_oversampled = caffe.io.oversample([caffe.io.resize_image(input_image, net.image_dims)], net.crop_dims)
caffe_input = np.asarray([net.preprocess('data', in_)[0:1,:,:] for in_ in input_oversampled])
caffe_input = caffe_input[0:1]
t0=time.time()
f=net.forward_all(data=caffe_input, blobs=['pool2'])
print 'elaspe:', time.time()-t0
t0=time.time()
f2=net.forward_all2(data=caffe_input, blobs=['pool2'], end='pool2')
print 'elaspe:', time.time()-t0

for key in f:
    print key, f[key].shape
for key in f2:
    print key, f2[key].shape
print 'diff:\n', np.sum(f['pool2']-f2['pool2'])

