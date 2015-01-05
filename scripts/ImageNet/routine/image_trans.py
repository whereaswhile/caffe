import sys
import os
import numpy as np
import skimage.io
from scipy.ndimage import zoom
from skimage.transform import resize

def has_valid_dim(im):
    if len(im.shape)!=3:
        return 0
    elif im.shape[2]!=1 and im.shape[2]!=3:
        return 0
    else:
        return 1

def resizeshort(im, l, interp_order=1):
    """
    Resize the shorter side of image
    inputs
    im: (H x W x K) ndarray
    l: length of the shorter side after resizing
    interp_order: interpolation order, default is linear.
    """
    assert(has_valid_dim(im)==1)
    h, w, c = im.shape
    ratio = 1.0*l / min(w,h)
    new_dims =[max(l, int(h*ratio)), max(l, int(w*ratio))]
    im_min, im_max = im.min(), im.max()
    im_std = (im - im_min) / (im_max - im_min)
    im_res = resize(im_std, new_dims, order=interp_order)*(im_max - im_min)+im_min
    return im_res.astype(np.float32)

def oversample(imarr, crop_num, crop_dim, mirror=0):
    """
    input
    imarr: an array of N (H x W x K) ndarrays
    crop_num: number of crops on each side
    crop_dim: crop size is crop_dim x crop_dim
    output
    imcrops: (N*crop_num^2 x crop_dim x crop_dim x K)
    """
    assert(has_valid_dim(imarr[0])==1)
    h, w, c = imarr[0].shape
    l = crop_dim
    if crop_num==1:
        dxs = [int((w-l)/2)]
        dys = [int((h-l)/2)]
    else:
        dxs = [int((w-l)*_/(crop_num-1)) for _ in range(crop_num)]
        dys = [int((h-l)*_/(crop_num-1)) for _ in range(crop_num)]

    imcrops = [None]*len(imarr)*(crop_num*crop_num)
    curr = 0
    for im in imarr:
        for dx in dxs:
            for dy in dys:
                imcrops[curr] = im[dy:dy+l, dx:dx+l, :]
                curr += 1
    if mirror==1:
        imflip = [_[:,::-1,:] for _ in imcrops]
        imcrops += imflip
    return imcrops

