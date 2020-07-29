function file_path = make_image_method_data(num_of_files,start_index)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

arguments
    num_of_files
    start_index = 1
end

SELECTED_FREQ = 500;
folder_path = fullfile(get_raw_data_folder_path(), "image-method");
file_path = strings(num_of_files,1);
wb = wbar();
for i=start_index:num_of_files
    %%
    [~, anm_target, fs] = simulator(i, "roomIndex", 2);

    %%
    [R, freq] = calculate_narrowband_scm(anm_target, fs);
    [~, j] = min(abs(freq-SELECTED_FREQ));
    R = R(:,:,j);
    
    v = [real(R(:)).'; imag(R(:)).'];
    file_path(i) = fullfile(folder_path, sprintf("%08d.bin", i)); 
    fid = fopen(file_path(i), 'w');
    fwrite(fid, v(:), 'double');
    wbar(i,num_of_files,wb);
end

end