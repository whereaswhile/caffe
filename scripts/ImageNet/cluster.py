#! /usr/bin/env python

import numpy as np
import scipy.misc
import sys
import cPickle as pickle
from sklearn.cluster import MiniBatchKMeans

FEA_FILE='./save/val_fea_vgg16.p'
BATCH=2000
K=5

print 'loading {}...'.format(FEA_FILE)
d=pickle.load(open(FEA_FILE, 'rb'))

mbk = MiniBatchKMeans(init='k-means++', n_clusters=K, batch_size=BATCH,
                      n_init=10, max_no_improvement=50, verbose=1)
mbk.fit(d['fea'])
centers = mbk.cluster_centers_
labels = mbk.labels_

pickle.dump({'centers': centers, 'cid':labels}, open('./save/cluster_B{}_K{}.p'.format(BATCH, K), 'wb'))

