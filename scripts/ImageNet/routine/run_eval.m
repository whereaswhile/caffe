function run_eval(pred_file, list_file)
%% pred_file: predictions of label and bbx in submission format
%% list_file: image list index in validation, from 1. Default starting from 1.

addpath('/mnt/disk2/wang308/workspace/data/ImgNet/evaluation/');

fprintf('evaluating %s...\n', pred_file)
if ~exist('list_file', 'var')
	list_file='';
end

fprintf('CLASSIFICATION WITH LOCALIZATION TASK\n');

data_dir = '/mnt/disk2/wang308/workspace/data/ImgNet/data/';
meta_file = fullfile(data_dir, 'meta_clsloc.mat');
ground_truth_file = fullfile(data_dir, 'ILSVRC2014_clsloc_validation_ground_truth_subset20000.txt'); %class label
blacklist_file = fullfile(data_dir, 'ILSVRC2014_clsloc_validation_blacklist.txt');
ground_truth_dir='/mnt/disk2/wang308/workspace/data/ImgNet/bbx_val'; %containing bbox html files
num_predictions_per_image=5;
optional_cache_file = './tmp/cache.mat';

val_files = dir(sprintf('%s/*.xml',ground_truth_dir));
num_val_files = numel(val_files);
if num_val_files ~= 50000
    fprintf('That does not seem to be the correct directory.\n');
	return;
end

error_cls = zeros(num_predictions_per_image,1);
error_loc = zeros(num_predictions_per_image,1);

for i=[1, num_predictions_per_image]
	fprintf('evaluating top %d...\n', i);
	[error_cls(i) error_loc(i)] = eval_clsloc(pred_file, list_file, ground_truth_file, ground_truth_dir, ...
		                                       meta_file, i, blacklist_file, optional_cache_file);
end

disp('# guesses vs clsloc error vs cls-only error');
disp([(1:num_predictions_per_image)',error_loc,error_cls]);

