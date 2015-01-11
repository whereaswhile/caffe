#! /usr/bin/env python
# use caffe classification results without spatial distribution
#    jobn=int(sys.argv[1])
#    jobid=int(sys.argv[2])

from numpy import *
import sys
import os
import cPickle as pickle
import scipy.misc
import scipy.io as sio
sys.path.append('../../../common')
from w_util import readLines, rect_overlap, get_box, merge_box

OUT_RT='./fea/pred-val/'
BBX_RT='/mnt/disk2/wang308/workspace/code/cnn/ImgNet/fea/3k-256-1k-val/'
PROB_FILE='./save/val_res_vgg16_crop50_top10.p' #classification features, from test_val_imgprvd.py
BBX_DIRS=['3k6l-rnorm1/win0.8_crop25/bbxov-fc8/']
KM_RES_FILES=['3k6l-rnorm1/bbx_km/kmres_0.8_25.mat']
BBX_FILES=[BBX_RT + _ + "data_batch_{}" for _ in BBX_DIRS]
BBX_CONFIG_FILES=[BBX_RT + _ + "param.cfg" for _ in BBX_DIRS]
BBX_VIEW=len(BBX_FILES)
PRED_FILE1=OUT_RT + 'pred1_prob_bbxmv-{}.txt'.format(BBX_VIEW) #1 bbx/cls
PRED_FILE2=OUT_RT + 'pred2_prob_bbxmv-{}.txt'.format(BBX_VIEW) #multi-bbx/cls
CLASS_NUM=1000
BAT_SIZE=128
TOP_N=5
TOP_CAND=10
IMG_NUM=20000
NN_SIZE=224.0
SIZE_FILE='/mnt/disk2/wang308/workspace/data/ImgNet/data/bbx_val_gt.mat'
CLS_ONLY=0

# read configure files
param={}
param['bbx']=[]
for f in BBX_CONFIG_FILES:
    paramb={}
    plines=readLines(f)
    for l in plines:
        l=l.rstrip().split()
        paramb[l[0]]=l[1]
    assert(paramb['crop'][0]=='u')
    paramb['ncrop']=int(paramb['crop'][1:])
    paramb['scale']=float(paramb['scale'])
    param['bbx']+=[paramb]
del paramb, f, l
print param
print '%d images'  % IMG_NUM

prob_res=pickle.load(open(PROB_FILE, 'rb'))
prob_res['pred']=prob_res['pred'].astype('int32')

sz=sio.loadmat(SIZE_FILE)['imsize'][0]
sz=[(sz[i][0,0][0][0,0], sz[i][0,0][1][0,0]) for i in range(IMG_NUM)]

scr_rank=array([1, 0.95, 0.9, 0.9, 0.85, 0.7, 0.7, 0.6, 0.6, 0.6])
scr_rank=scr_rank/sum(scr_rank)

km=[]
for k in range(BBX_VIEW):
    km+=[sio.loadmat(BBX_RT+KM_RES_FILES[k])]

# initialize variables
d_bbx=[None]*BBX_VIEW
bbx_all=[]
base_view=[]
for k in range(BBX_VIEW):
    bbx_all+=[zeros([param['bbx'][k]['ncrop'], 4])]
    base_view+=[int(sqrt(param['bbx'][k]['ncrop']))]
if len(sys.argv)==1:
    jobn=1
    jobid=0
else:
    jobn=int(sys.argv[1])
    jobid=int(sys.argv[2])
istart=IMG_NUM/jobn*jobid
iend=min(IMG_NUM/jobn*(jobid+1), IMG_NUM)
print "processing job {}/{}, image [{}, {})".format(jobid, jobn, istart, iend)
PRED_FILE1=PRED_FILE1+'.job{}'.format(jobid)
PRED_FILE2=PRED_FILE2+'.job{}'.format(jobid)
# set batch counters to istart-1
b_bbx=zeros([BBX_VIEW, ], 'int')
bcnt_bbx=zeros([BBX_VIEW, ], 'int')
for k in range(BBX_VIEW):
    b_bbx[k]=(istart*param['bbx'][k]['ncrop'])/BAT_SIZE
    bcnt_bbx[k]=(istart*param['bbx'][k]['ncrop'])%BAT_SIZE
fid1=open(PRED_FILE1, 'w')
fid2=open(PRED_FILE2, 'w')

# main processing loop
for i in range(istart, iend):
    if i%100==0:
        print "processing image ", i
        sys.stdout.flush()

    # read bbx predictions
    for k in range(BBX_VIEW):
        for j in range(param['bbx'][k]['ncrop']):
            if ((i==istart and j==0) or bcnt_bbx[k]==BAT_SIZE): #load next batch
                fn=BBX_FILES[k].format(b_bbx[k])
                d_bbx[k]=pickle.load(open(fn, "rb"))
                if b_bbx[k]%1000==0:
                    print "loading...", fn
                    print "total #data: {}, dimension: {}".format(size(d_bbx[k]['data'])/d_bbx[k]['num_vis'], d_bbx[k]['num_vis'])
                bcnt_bbx[k]=bcnt_bbx[k]%BAT_SIZE
                b_bbx[k]+=1
            bbx_all[k][j]=d_bbx[k]['data'][bcnt_bbx[k]]
            bcnt_bbx[k]+=1

    # classification
    cls_pred=prob_res['pred'][i, 0:TOP_CAND]+1 #[1, 1000]

    # only classification
    if CLS_ONLY==1:
        for r in range(TOP_N):
            fid1.write('{} '.format(cls_pred[r]))
            fid1.write('50 50 400 400 ')
        fid1.write('\n')
        continue

    # bbx prediction
    s=sz[i]
    bpred = zeros([array([_['ncrop'] for _ in param['bbx']]).sum(), 4])
    bovlp = zeros([bpred.shape[0], ])
    bcnt = 0
    for k in range(BBX_VIEW):
        for j in range(param['bbx'][k]['ncrop']):
            borgj = get_box(s[0], s[1], param['bbx'][k]['scale'], 'u{}-{}'.format(base_view[k], j))
            dx = borgj[1]-1
            dy = borgj[0]-1
            l = borgj[2]-borgj[0]+1
            jx = base_view[k]/2 - j%base_view[k]
            jy = base_view[k]/2 - j/base_view[k]
            bpredj = (bbx_all[k][j]+array([jy*0, jx*0, jy*0, jx*0]))*l/NN_SIZE + array([dy, dx, dy, dx]) + 1
            bpredj[0] = max(1, bpredj[0])
            bpredj[1] = max(1, bpredj[1])
            bpredj[2] = min(s[1], bpredj[2])
            bpredj[3] = min(s[0], bpredj[3])
            bbx_all[k][j]=array([bpredj[1], bpredj[0], bpredj[3], bpredj[2]])
            bovlp[bcnt] = rect_overlap(borgj, bpredj)[0]
            bpred[bcnt] = bpredj.reshape((1, 4))
            bcnt +=1

    # kmeans
    kmpred=[]
    kmcnt=[]
    kmscl=[]
    bcnt=0
    for k in range(BBX_VIEW):
        bpredk=bpred[bcnt:bcnt+param['bbx'][k]['ncrop']]
        for j in range(max(km[k]['km_idx'][i])):
            mask=km[k]['km_idx'][i]==j+1
            if mean(mask)<=0.1:
                continue
            if 'pred' in km[k]:
                kmpred+=[mean(km[k]['pred'][0,0][3][i][mask], axis=0)]
            else:
                kmpred+=[mean(bpredk[mask], axis=0)]
            kmcnt+=[mean(mask)]
            kmscl+=[param['bbx'][k]['scale']]
        bcnt+=param['bbx'][k]['ncrop']
    bpred=(array(kmpred)+0.5).astype(int)

    # for each top-n prediction
    score=zeros([1, bpred.shape[0]]) #scores for all detection windows, for 1 class
    scr_pred1=zeros([TOP_CAND, ]) #scores for all candidate classes, 1 bbx/cls
    scr_pred2=zeros([TOP_CAND*bpred.shape[0], ]) #scores for all candidate classes, multi-bbx/cls
    maxk=zeros([TOP_CAND, ])
    for j in range(TOP_CAND):
        # bbx fusion based on class hypothesis
        score[0, :]=0
        for k in range(score.shape[1]):
            score[0, k]=pow(kmcnt[k], 0.5)*pow(kmscl[k], 1)
        maxk[j]=argmax(score)
        scr_pred1[j] = scr_rank[j]*score[0, maxk[j]]
        scr_pred2[j*score.shape[1]:(j+1)*score.shape[1]] = scr_rank[j]*score
        scr_pred2[j*score.shape[1]+maxk[j]] *= 5 #10
    #print scr_pred1
    #print scr_pred2

    # rerank predictions and write to prediction files
    rank_idx=argsort(scr_pred1)[-1:-1-TOP_N:-1]
    for r in rank_idx:
        fid1.write('{} '.format(cls_pred[r]))
        j=maxk[r]
        fid1.write('{} {} {} {} '.format(bpred[j, 1], bpred[j, 0], bpred[j, 3], bpred[j, 2]))
    fid1.write('\n')
    rank_idx=argsort(scr_pred2)[-1:-1-TOP_N:-1]
    for r in rank_idx:
        fid2.write('{} '.format(cls_pred[r/score.shape[1]]))
        j=r%score.shape[1]
        fid2.write('{} {} {} {} '.format(bpred[j, 1], bpred[j, 0], bpred[j, 3], bpred[j, 2]))
    fid2.write('\n')
fid1.close()
fid2.close()

# accuracy evaluation (matlab)
#cmd='matlab -r "run_eval(\'{}\'); exit;"'.format(PRED_FILE1)
#os.system(cmd)

