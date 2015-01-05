%run in terminal before starting Matlab
%export LD_PRELOAD=$LD_PRELOAD:/usr/lib/gcc/x86_64-pc-linux-gnu/4.8.3/libstdc++.so.6:/usr/lib64/libfreetype.so.6
%export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/composerxe-2013.1.106/mkl/lib/intel64:/opt/cuda/lib64

% caffe path
addpath('/mnt/disk2/wang308/workspace/code/caffe-fork/matlab/caffe/');
addpath('./routine/');

% define
USE_GPU=1;
CAFFE_PATH='/mnt/disk2/wang308/workspace/code/caffe-fork';
MDL_DEF_FILE = fullfile(CAFFE_PATH, 'models/VGG_ILSVRC_16_layers/VGG_ILSVRC_16_layers_deploy.prototxt');
MDL_WGH_FILE = fullfile(CAFFE_PATH, 'models/VGG_ILSVRC_16_layers/VGG_ILSVRC_16_layers.caffemodel');
IMG_FILE = fullfile(CAFFE_PATH, 'examples/images/cat.jpg');

% init caffe network (spews logging info)
matcaffe_init(USE_GPU, MDL_DEF_FILE, MDL_WGH_FILE);
return;


% prepare input
im = imread(IMG_FILE); %color image as uint8 HxWx3
tic;
% input_data is Width x Height x Channel x Num
% if you have multiple images, cat them with cat(4, ...)
input_data = {prepare_image_ilsvrc(im)};
toc;

% do forward pass to get scores
tic;
scores = caffe('forward', input_data);
toc;

scores = scores{1};
size(scores)
scores = squeeze(scores);
scores = mean(scores,2);
[maxs, maxc] = max(scores);

fprintf('prediction: cls=%d, scr=%.4f\n', maxc, maxs);

