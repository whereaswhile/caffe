#! /usr/bin/env python

import numpy as np
import sys
import caffe
import timeit
sys.path.append('./routine')
import image_trans as imtrans

CAFFERT = '/mnt/disk2/wang308/workspace/code/caffe-fork'
MODEL_FILE = CAFFERT+'/models/VGG_ILSVRC_16_layers/VGG_ILSVRC_16_layers_deploy.prototxt'
PRETRAINED = CAFFERT+'/models/VGG_ILSVRC_16_layers/VGG_ILSVRC_16_layers.caffemodel'
IMAGE_FILE = CAFFERT+'/examples/images/cat.jpg'
MEAN_PIX = [103.939, 116.779, 123.68];
IMAGE_DIM = 256;
CROPPED_DIM = 224;

mean_img = np.zeros([3, IMAGE_DIM, IMAGE_DIM])
mean_img[0,:,:]=MEAN_PIX[0]
mean_img[1,:,:]=MEAN_PIX[1]
mean_img[2,:,:]=MEAN_PIX[2]
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                        mean=mean_img, channel_swap=(2,1,0), raw_scale=255, image_dims=(IMAGE_DIM, IMAGE_DIM))
print 'Network {} loaded'.format(PRETRAINED)

net.set_phase_test()
net.set_mode_gpu()
input_image = caffe.io.load_image(IMAGE_FILE)
# predict takes any number of images, and formats them for the Caffe net automatically
prediction = net.predict([input_image], oversample=True)
print 'prediction shape:', prediction[0].shape
print 'predicted classes:', np.where(prediction[0]>0.1)

input_oversampled = caffe.io.oversample([caffe.io.resize_image(input_image, net.image_dims)], net.crop_dims) #crop
caffe_input = np.asarray([net.preprocess('data', in_) for in_ in input_oversampled]) #preprocessing
pred2=net.forward(data=caffe_input) #calculate response
prob2=pred2['prob'][:,:,0,0]
prob2=np.mean(prob2, axis=0)
print 'predicted2 classes:', np.where(prob2>0.1)

imtrans.oversample()



