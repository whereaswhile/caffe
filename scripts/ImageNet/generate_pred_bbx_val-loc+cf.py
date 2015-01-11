#! /usr/bin/env python
# use caffe classification results together with the 6l probability with spatial distribution
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
PROB_RT='/mnt/disk2/wang308/workspace/code/cnn/ImgNet/fea/3k-256-1k-val/'
PROB_DIRS=['3k6l-15pert_4mdl', '3k6l-60pert_4mdl']
PROB_CLS_W=[0.55, 0.45]
PROB_MAP_W=[1, 0.1]
PROB_FILES=[PROB_RT + _ + "/probs/data_batch_{}" for _ in  PROB_DIRS] #classification features, from test_multi.sh
PROB_CONFIG_FILES=[PROB_RT + _ + "/probs/param.cfg" for _ in PROB_DIRS]
PROB_CF_RES='./save/val_res_vgg16_crop50_top10.p' #classification features, from test_val_imgprvd.py
BBX_DIRS=['3k6l-rnorm1/win0.8_crop25/bbxov-fc8/']
KM_RES_FILES=['3k6l-rnorm1/bbx_km/kmres_0.8_25.mat']
BBX_FILES=[BBX_RT + _ + "data_batch_{}" for _ in BBX_DIRS]
BBX_CONFIG_FILES=[BBX_RT + _ + "param.cfg" for _ in BBX_DIRS]
PROB_VIEW=len(PROB_FILES)
BBX_VIEW=len(BBX_FILES)
PRED_FILE1=OUT_RT + 'pred1_probmv-{}_bbxmv-{}.txt'.format(PROB_VIEW, BBX_VIEW) #1 bbx/cls
PRED_FILE2=OUT_RT + 'pred2_probmv-{}_bbxmv-{}.txt'.format(PROB_VIEW, BBX_VIEW) #multi-bbx/cls
CLASS_NUM=1000
BAT_SIZE=128
TOP_N=5
TOP_CAND=5
NN_SIZE=224.0
SIZE_FILE='/mnt/disk2/wang308/workspace/data/ImgNet/data/bbx_val_gt.mat'
if 0:
    GT_LBL_FILE='/mnt/disk1/whan/ILSVRC2012/ilsvrc2012_val_labels.txt'
    gt_label=readLines(GT_LBL_FILE)
    gt_label=[int(_.rstrip())+1 for _ in gt_label]
    print 'max gt label:', max(gt_label)
SAVE_MAT=0
CLS_ONLY=0

# read configure files
param={}
param['prob']=[]
for f in PROB_CONFIG_FILES:
    paramb={}
    plines=readLines(f)
    for l in plines:
        l=l.rstrip().split()
        paramb[l[0]]=l[1]
    paramb['scale']=[float(_) for _ in paramb['scale'].split('+')]
    paramb['crop']=paramb['crop'].split('+') #list of initial strings
    param['prob']+=[paramb]
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

pertcomb=[]
for i in range(PROB_VIEW): #each feature folder
    param['prob'][i]['ncrop']=0
    for crop in param['prob'][i]['crop']: #each crop
        if crop[0]=='f':
            flip=1
            crop=crop[1:]
        else:
            flip=0
        if crop[0]=='u':
            nc=int(crop[1:])
            crops=['u{}-{}'.format(int(sqrt(nc)), _) for _ in range(nc)]
        elif crop[0]=='g':
            nc=crop[1:].split('x')
            nc=array([int(_) for _ in nc]).prod()
            crops=[crop+'-{}'.format(_) for _ in range(nc)]
        else:
            crops=[crop]
        for c in crops: #each perturbation
            for s in param['prob'][i]['scale']:
                pertcomb+=[[c, flip, s]]
                param['prob'][i]['ncrop']+=1
                if c=='wh':
                    break
PERT_NUM=len(pertcomb)
IMG_NUM=int(param['prob'][0]['imgnum'])
#IMG_NUM=200
del crop, flip, c, s, crops
print param
print '%d images expanded with %d perturbation(s):' % (IMG_NUM, PERT_NUM)
print pertcomb

sz=sio.loadmat(SIZE_FILE)['imsize'][0]
sz=[(sz[i][0,0][0][0,0], sz[i][0,0][1][0,0]) for i in range(IMG_NUM)]

#scr_rank=1.0/array(range(100, 100+TOP_CAND))
#scr_rank=array([1, 0.8, 0.8, 0.7, 0.6, 0.5, 0.5, 0.4, 0.4, 0.3])
scr_rank=array([1, 0.7, 0.6, 0.6, 0.5, 0.5, 0.4, 0.4, 0.3, 0.3])
scr_rank=scr_rank/sum(scr_rank)

prob_res=pickle.load(open(PROB_CF_RES, 'rb'))
prob_res['pred']=prob_res['pred'].astype('int32')

if SAVE_MAT!=1:
    km=[]
    for k in range(BBX_VIEW):
        km+=[sio.loadmat(BBX_RT+KM_RES_FILES[k])]

# initialize variables
prob_all=[zeros([_['ncrop'], CLASS_NUM]) for _ in param['prob']]
d_prob=[None]*PROB_VIEW
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
b_prob=zeros([PROB_VIEW, ], 'int')
bcnt_prob=zeros([PROB_VIEW, ], 'int')
for k in range(PROB_VIEW):
    b_prob[k]=(istart*param['prob'][k]['ncrop'])/BAT_SIZE
    bcnt_prob[k]=(istart*param['prob'][k]['ncrop'])%BAT_SIZE
b_bbx=zeros([BBX_VIEW, ], 'int')
bcnt_bbx=zeros([BBX_VIEW, ], 'int')
for k in range(BBX_VIEW):
    b_bbx[k]=(istart*param['bbx'][k]['ncrop'])/BAT_SIZE
    bcnt_bbx[k]=(istart*param['bbx'][k]['ncrop'])%BAT_SIZE
fid1=open(PRED_FILE1, 'w')
fid2=open(PRED_FILE2, 'w')
save_pred=[]
save_prob=[]
save_clsbbx=[]

# main processing loop
for i in range(istart, iend):
    if i%100==0:
        print "processing image ", i
        sys.stdout.flush()

    # read probabilities
    for k in range(PROB_VIEW):
        for j in range(param['prob'][k]['ncrop']):
            if ((i==istart and j==0) or bcnt_prob[k]==BAT_SIZE): #load next batch
                fn=PROB_FILES[k].format(b_prob[k])
                if b_prob[k]%1000==0:
                    print "loading...", fn
                    #print "total #data: {}, dimension: {}".format(size(d['data'])/d['num_vis'], d['num_vis'])
                d_prob[k]=pickle.load(open(fn, "rb"))
                bcnt_prob[k]=bcnt_prob[k]%BAT_SIZE
                b_prob[k]+=1
            #if (j==0):
            #    cury=d_prob[k]['labels'][0, bcnt_prob[k]]
            prob_all[k][j]=d_prob[k]['data'][bcnt_prob[k]]
            #y=d_prob[k]['labels'][0, bcnt_prob[k]]
            #assert(y==cury)
            bcnt_prob[k]+=1

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
    #p=[mean(_, 0) for _ in prob_all]
    #p=dot(array(PROB_CLS_W).T, array(p))
    #cls_pred=argsort(p)[-1:-1-TOP_CAND:-1]+1
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
            #if (j%base_view[k]%15<5 or j%base_view[k]%15>9 or j/base_view[k]%15<5 or j/base_view[k]%15>9): # sample by half
            #    bovlp[bcnt] = -1
            #    bpredj[:] = 1
            #else:
            #    bovlp[bcnt] = 1
            bpred[bcnt] = bpredj.reshape((1, 4))
            bcnt +=1

    # prepare classification bbx
    bbx_cls_all=[None for _ in range(len(pertcomb))]
    for l in range(PERT_NUM):
        if pertcomb[l][0]=='wh':
            bbx_cls_all[l]=array([1, 1, s[1], s[0]])
        else: #u/g
            bbx_cls_all[l]=get_box(s[0], s[1], pertcomb[l][2], pertcomb[l][0])

    if SAVE_MAT==1: # save
        save_pred+=[bpred]
        continue

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
    resmap=zeros([s[1], s[0]])
    cntmap=zeros([s[1], s[0]])
    score=zeros([1, bpred.shape[0]]) #scores for all detection windows, for 1 class
    scr_pred1=zeros([TOP_CAND, ]) #scores for all candidate classes, 1 bbx/cls
    scr_pred2=zeros([TOP_CAND*bpred.shape[0], ]) #scores for all candidate classes, multi-bbx/cls
    maxk=zeros([TOP_CAND, ])
    for j in range(TOP_CAND):
        # bbx fusion based on class hypothesis
        cidx=cls_pred[j]-1
        p=concatenate(tuple(prob_all))
        #p=dot(array(PROB_MAP_W).T, array(p))
        p=p[:, cidx].flatten().tolist()
        resmap[:, :]=0
        cntmap[:, :]=0.001
        for l in range(PERT_NUM):
            if pertcomb[l][0]=='wh': # or pertcomb[l][0][0]=='u':
                continue
            bbx_cls=bbx_cls_all[l]
            if pertcomb[l][2]<=0.5:
                w=0.1 #0.2 #0.1
            else:
                w=1.0
            resmap[bbx_cls[0]:bbx_cls[2], bbx_cls[1]:bbx_cls[3]]+=p[l]*w
            cntmap[bbx_cls[0]:bbx_cls[2], bbx_cls[1]:bbx_cls[3]]+=1.0*w
        resmap=divide(resmap, cntmap)
        score[0, :]=0
        for k in range(score.shape[1]):
            score[0, k]=pow(kmcnt[k], 0.5)*mean(resmap[bpred[k][0]:bpred[k][2], bpred[k][1]:bpred[k][3]])*pow(kmscl[k], 1)
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

if SAVE_MAT==1: # save
    print "saving to mat..."
    mdic={'bbx_pred': save_pred, 'cls_prob': save_prob, 'cls_bbx': save_clsbbx, 'frames': i}
    sio.savemat(BBX_RT + 'bbx_pred.mat', mdic, do_compression=True)

# accuracy evaluation (matlab)
#cmd='matlab -r "run_eval(\'{}\'); exit;"'.format(PRED_FILE1)
#os.system(cmd)

