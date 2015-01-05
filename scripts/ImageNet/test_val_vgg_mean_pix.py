#! /usr/bin/env python

import numpy as np
import sys
import caffe
import timeit
import cPickle as pickle

CAFFERT = '/mnt/disk2/wang308/workspace/code/caffe-fork'
MODEL_FILE = CAFFERT+'/models/VGG_ILSVRC_16_layers/VGG_ILSVRC_16_layers_deploy.prototxt'
PRETRAINED = CAFFERT+'/models/VGG_ILSVRC_16_layers/VGG_ILSVRC_16_layers.caffemodel'
MEAN_PIX = [103.939, 116.779, 123.68]; #BGR
IMAGE_DIM = 256;
CROPPED_DIM = 224;
IMAGE_FILE = '/mnt/disk2/wang308/workspace/data/ImgNet/ILSVRC2012_img_val/ILSVRC2012_val_{:08d}.JPEG'
IMAGE_NUM = 50000
CLASS_NUM = 1000
LABEL_FILE = '/mnt/disk1/whan/ILSVRC2012/ilsvrc2012_val_labels.txt'
LABEL_CVRT = '/mnt/disk2/wang308/workspace/code/caffe-fork/data/ilsvrc12/cf2im.txt'

mean_img = np.zeros([3, IMAGE_DIM, IMAGE_DIM]) #channel order after swap
mean_img[0,:,:]=MEAN_PIX[0]
mean_img[1,:,:]=MEAN_PIX[1]
mean_img[2,:,:]=MEAN_PIX[2]
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                        mean=mean_img, channel_swap=(2,1,0), raw_scale=255, image_dims=(IMAGE_DIM,IMAGE_DIM))
print 'Network {} loaded'.format(PRETRAINED)
net.set_phase_test()
net.set_mode_gpu()

# predict
prob=np.zeros([IMAGE_NUM, CLASS_NUM])
gt = np.loadtxt(LABEL_FILE) #[0, 999]
gt = gt[0:IMAGE_NUM]
cf2im = np.loadtxt(LABEL_CVRT)
for i in range(IMAGE_NUM):
    fn = IMAGE_FILE.format(i+1)
    input_image = caffe.io.load_image(fn)
    probi = net.predict([input_image], oversample=True)
    prob[i] = probi
    #print 'prediction shape:', prediction[0].shape
    #print 'predicted class:', prediction[0].argmax()

    if i%100==0 or i==IMAGE_NUM-1:
        print 'image file:', fn
        # evaluate
        pred = np.argsort(prob[0:i+1], axis=1)
        pred = pred[:, ::-1]
        pred = cf2im[pred[:, 0:5]]-1 #convert to imagenet label [0, 999]
        diff=np.abs(np.reshape(np.repeat(gt[0:i+1], 5), [i+1, 5])-pred)
        acc1=np.mean(diff[:, 0]==0)
        acc5=np.mean(np.min(diff, axis=1)==0)
        print 'val {}: acc={}, {}'.format(i, acc1*100, acc5*100)
        sys.stdout.flush()
        pickle.dump({'prob': prob, 'pred':pred}, open('./save/val_res_vgg16.p', 'wb'))

