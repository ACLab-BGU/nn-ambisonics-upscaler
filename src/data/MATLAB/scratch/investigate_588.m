clear;
folder = "/Users/tomshlomo/PycharmProjects/nn-ambisonics-upscaler2/data/whitenoise_10_reflections/train";
files = dir(folder+"/*.mat");
nfiles = length(files);
FREQ = 1000;
N_in = 3;
N_out = 4;
Q_in = (N_in+1)^2;
Q_out = (N_out+1)^2;
loss_if_only_R_in_is_learned = zeros(nfiles,1);
loss_if_only_R_in_is_not_learned = zeros(nfiles,1);
for i=1:nfiles
    i/nfiles
    s = load(fullfile(files(i).folder, files(i).name));
    [~, j] = min(abs(s.freq - FREQ));
    R = inv_vec_hermitian_matrix(s.R(:,j), true);
    R_in = R(1:Q_in, 1:Q_in);
    R_out = R(1:Q_out, 1:Q_out);
    R_out_norm = norm(R_out, 'fro');
    R_out = R_out/R_out_norm;
    R_in = R_in/R_out_norm;
    
    [~, s] = svd(R_out);
    one_rankness(i) = s(1);
    residual_if_only_R_in_is_learned = R_out;
    residual_if_only_R_in_is_learned(1:Q_in, 1:Q_in) = 0;
    loss_if_only_R_in_is_learned(i) = norm(residual_if_only_R_in_is_learned)^2;
    
    residual_if_only_R_in_is_not_learned = blkdiag(R_in, zeros(Q_out-Q_in));
    loss_if_only_R_in_is_not_learned(i) = norm(residual_if_only_R_in_is_not_learned)^2;
    
end

figure;
histogram(loss_if_only_R_in_is_learned);
hold on;
histogram(loss_if_only_R_in_is_not_learned);

figure;
histogram(one_rankness);

mean(loss_if_only_R_in_is_learned)
mean(loss_if_only_R_in_is_not_learned)
