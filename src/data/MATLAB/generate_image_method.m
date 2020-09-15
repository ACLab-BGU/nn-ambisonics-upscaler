restoredefaultpath
addpath('spherical harmonics/');
% addpath('/Users/tomshlomo/Datasets/TSP');

% [num_of_reflections, source_type] = ndgrid([inf, 10, 0], ["whitenoise", "speech"]);
num_of_reflections = 10;
source_type = "whitenoise";
num_of_files = 10;
sig_length = 1;
target_sh_order = 5;

for i=1:numel(num_of_reflections)
    folder_name = sprintf("%s_%d_reflections", source_type(i), num_of_reflections(i));
    folder_path = fullfile(get_raw_data_folder_path(), folder_name);
    mkdir(folder_path);
    make_image_method_data(num_of_files, 1, "number_of_reflections", num_of_reflections(i), "folder_path", folder_path, "source_type", source_type(i),"duration",sig_length, "target_sh_order",target_sh_order);
end
