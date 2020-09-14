restoredefaultpath
addpath('spherical harmonics/');
% addpath('/Users/tomshlomo/Datasets/TSP');

% [num_of_reflections, source_type] = ndgrid([inf, 10, 0], ["whitenoise", "speech"]);
num_of_reflections = 10;
source_type = "whitenoise";
num_of_filers = 10;

for i=1:numel(num_of_reflections)
    folder_name = sprintf("%s_%d_reflections", source_type(i), num_of_reflections(i));
    folder_path = fullfile(get_raw_data_folder_path(), folder_name);
    mkdir(folder_path);
    make_image_method_data(num_of_filers, 1, "number_of_reflections", num_of_reflections(i), "source_type", source_type(i));
end
