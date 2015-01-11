#! /usr/bin/env python

import numpy as np
import scipy.misc
import sys
import cPickle as pickle

CLUSTER_FILE='./save/cluster_B2000_K5.p'
PRED_FILE='save/val_res_vgg16_crop50_top10.p'
LABEL_FILE = '/mnt/disk1/whan/ILSVRC2012/ilsvrc2012_val_labels.txt'

d=pickle.load(open(CLUSTER_FILE, 'rb'))
#print np.histogram(d['cid'], bins=np.unique(d['cid']))
print [np.sum(d['cid']==_) for _ in range(d['cid'].max()+1)]
p=pickle.load(open(PRED_FILE, 'rb'))
print p['pred'].shape
gt = np.loadtxt(LABEL_FILE) #[0, 999]

diff=np.abs(np.reshape(np.repeat(gt, 5), [gt.shape[0], 5])-p['pred'][:,0:5])
acc1=np.mean(diff[:, 0]==0)
acc5=np.mean(np.min(diff, axis=1)==0)
print 'overall: acc={}, {}'.format(acc1*100, acc5*100)

for i in range(d['cid'].max()+1):
    diffi=diff[np.where(d['cid']==i)[0], :]
    acc1=np.mean(diffi[:, 0]==0)
    acc5=np.mean(np.min(diffi, axis=1)==0)
    print 'cluster {}: size={}, acc={:.2f}/{:.2f}'.format(i, diffi.shape[0], acc1*100, acc5*100)

