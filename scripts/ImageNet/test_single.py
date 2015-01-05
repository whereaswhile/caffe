#! /usr/bin/env python

import numpy as np
import sys
import caffe
import timeit

CAFFERT = '/mnt/disk2/wang308/workspace/code/caffe-fork'
MODEL_FILE = CAFFERT+'/models/bvlc_reference_caffenet/deploy.prototxt'
PRETRAINED = CAFFERT+'/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
IMAGE_FILE = CAFFERT+'/examples/images/cat.jpg'
MEAN_IMG = CAFFERT+'/python/caffe/imagenet/ilsvrc_2012_mean.npy'

mean_img = np.load(MEAN_IMG)
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                        mean=mean_img, channel_swap=(2,1,0), raw_scale=255, image_dims=(256,256))
print 'Network {} loaded'.format(PRETRAINED)


net.set_phase_test()
net.set_mode_gpu()
input_image = caffe.io.load_image(IMAGE_FILE)
# predict takes any number of images, and formats them for the Caffe net automatically
prediction = net.predict([input_image], oversample=True)
print 'prediction shape:', prediction[0].shape
print 'predicted class:', prediction[0].argmax()

#timeit.timeit('net.predict([input_image])', setup='from __main__ import net, input_image', number=10)
input_oversampled = caffe.io.oversample([caffe.io.resize_image(input_image, net.image_dims)], net.crop_dims)
# 'data' is the input blob name in the model definition, so we preprocess for that input.\n",
caffe_input = np.asarray([net.preprocess('data', in_) for in_ in input_oversampled])
timeit.timeit('net.forward(data=caffe_input)', setup='from __main__ import net, caffe_input', number=10)


