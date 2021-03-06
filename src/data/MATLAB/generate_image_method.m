restoredefaultpath
addpath('spherical harmonics/');
% addpath('/Users/tomshlomo/Datasets/TSP');

% [num_of_reflections, source_type] = ndgrid([inf, 10, 0], ["whitenoise", "speech"]);
num_of_reflections = 10;
source_type = "whitenoise";
num_of_files = 10;
sig_length = 2;
target_sh_order = 4;
nfft = 512;
stft_win_size = 512;
stft_hop_size = 256;
compact_cov = false;

for i=1:numel(num_of_reflections)
    folder_name = sprintf("%s_%d_reflections", source_type(i), num_of_reflections(i));
    folder_path = fullfile(get_raw_data_folder_path(), folder_name);
    mkdir(folder_path);
    make_image_method_data(num_of_files, 1, "number_of_reflections", num_of_reflections(i), "folder_path", folder_path, "source_type",...
        source_type(i),"duration",sig_length, "target_sh_order",target_sh_order,...
        "nfft",nfft,"stft_hop_size",stft_hop_size,"stft_win_size",stft_win_size, "compact_cov",compact_cov);
end
