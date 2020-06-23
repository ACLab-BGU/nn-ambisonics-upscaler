function make_free_field_data(num_of_files, length_sec, N)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

rng(1);
folder_path = fullfile(get_raw_data_folder_path(), "free-field");
doa = zeros(num_of_files, 2);
file_path = strings(num_of_files,1);
fs = zeros(num_of_files, 1);
for i=1:num_of_files
    [anm, fs(i), doa(i,:)] = free_field_simulator(length_sec, N);
    [R, freq] = calculate_narrowband_scm(anm, fs(i));
    file_path(i) = fullfile(folder_path, sprintf("%04d.bin", i)); 
    fid = fopen(file_path(i), 'w');
    fwrite(fid, R);
end
info = table(file_path, fs, doa(:,1), doa(:,2), 'VariableNames', ["file_path","fs","elevation","azimuth"]);
writetable(info, fullfile(folder_path, "info.csv"));
writematrix(freq, fullfile(folder_path, "frequencies.txt"));

end

