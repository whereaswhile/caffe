function images = prepare_image_ilsvrc(im)

% Data coming in from matlab needs to be in the order 
%   [width, height, channels, images]
% where width is the fastest dimension.
% Here is the rough matlab for putting image data into the correct
% format:
%   % convert from uint8 to single
%   im = single(im);
%   % reshape to a fixed size (e.g., 227x227)
%   im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
%   % permute from RGB to BGR and subtract the data mean (already in BGR)
%   im = im(:,:,[3 2 1]) - data_mean;
%   % flip width and height to make width the fastest dimension
%   im = permute(im, [2 1 3]);

% Five things to be aware of:
%   caffe uses row-major order
%   matlab uses column-major order
%   caffe uses BGR color channel order
%   matlab uses RGB color channel order
%   images need to have the data mean subtracted

d = load('/mnt/disk2/wang308/workspace/code/caffe-fork/matlab/caffe/ilsvrc_2012_mean.mat');
IMAGE_MEAN = d.image_mean;
IMAGE_DIM = 256;
CROPPED_DIM = 227;

% resize to fixed input size
im = single(im);
im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
% permute from RGB to BGR (IMAGE_MEAN is already BGR)
im = im(:,:,[3 2 1]) - IMAGE_MEAN;

% oversample (4 corners, center, and their x-axis flips)
images = zeros(CROPPED_DIM, CROPPED_DIM, 3, 10, 'single');
indices = [0 IMAGE_DIM-CROPPED_DIM] + 1;
curr = 1;
for i = indices
  for j = indices
    images(:, :, :, curr) = ...
        permute(im(i:i+CROPPED_DIM-1, j:j+CROPPED_DIM-1, :), [2 1 3]); %from height first to width first
    images(:, :, :, curr+5) = images(end:-1:1, :, :, curr); %reflection around x-axis
    curr = curr + 1;
  end
end
center = floor(indices(2) / 2)+1;
images(:,:,:,5) =  permute(im(center:center+CROPPED_DIM-1,center:center+CROPPED_DIM-1,:), [2 1 3]);
images(:,:,:,10) = images(end:-1:1, :, :, curr);

