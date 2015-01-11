#! /usr/bin/env python

import numpy as np
import scipy.misc
import sys
import caffe
import timeit
import cPickle as pickle
sys.path.append('./routine')
import image_trans as imtrans

CAFFERT = '/mnt/disk2/wang308/workspace/code/caffe-fork'
MODEL_FILE = CAFFERT+'/models/VGG_ILSVRC_16_layers/VGG_ILSVRC_16_layers_deploy.prototxt'
PRETRAINED = CAFFERT+'/models/VGG_ILSVRC_16_layers/VGG_ILSVRC_16_layers.caffemodel'
MEAN_PIX = [103.939, 116.779, 123.68]; #BGR
IMAGE_DIM = 384;
CROPPED_DIM = 224;
CROP_NUM = 1; #centeral crop
CROP_FLP = 0;
IMAGE_FILE = '/mnt/disk2/wang308/workspace/data/ImgNet/ILSVRC2012_img_val/ILSVRC2012_val_{:08d}.JPEG'
IMAGE_NUM = 50000
FEA_DIM = 512

mean_img = np.zeros([3, CROPPED_DIM, CROPPED_DIM]) #channel order after swap
mean_img[0,:,:]=MEAN_PIX[0]
mean_img[1,:,:]=MEAN_PIX[1]
mean_img[2,:,:]=MEAN_PIX[2]
net = caffe.Classifier(MODEL_FILE, PRETRAINED, mean=mean_img, channel_swap=(2,1,0), raw_scale=255)
print 'Network {} loaded'.format(PRETRAINED)
net.set_phase_test()
net.set_mode_gpu()

# predict
fea=np.zeros([IMAGE_NUM, FEA_DIM])
for i in range(IMAGE_NUM):
    fn = IMAGE_FILE.format(i+1)
    if i%100==0:
        print 'image file:', fn
        sys.stdout.flush()

    input_image = caffe.io.load_image(fn)
    input_oversampled = imtrans.oversample([imtrans.resizeshort(input_image, IMAGE_DIM)], CROP_NUM,
                        CROPPED_DIM, mirror=CROP_FLP) #crop
    data_input = np.asarray([net.preprocess('data', in_) for in_ in input_oversampled]) #preprocessing
    fi=net.forward_all(data=data_input, blobs=['pool5']) #calculate feature response
    #print fi['pool5'].shape
    fi=fi['pool5'][0]
    fi=np.max(np.max(fi, axis=1), axis=1)
    fea[i] = fi

pickle.dump({'fea': fea}, open('./save/val_fea_vgg16.p', 'wb'))

