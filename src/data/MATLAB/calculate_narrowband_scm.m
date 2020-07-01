function [R, freq] = calculate_narrowband_scm(anm, fs, opts)
arguments
    anm (:,:) double
    fs (1,1) double {mustBeNonnegative}
    opts.stft_window = hann(1024);
    opts.stft_hop = 512;
    opts.NFFT = 1024;
end

Q = size(anm, 2);
[anm_stft, freq] = stft(anm, opts.stft_window, opts.stft_hop, opts.NFFT, fs);
F = floor(opts.NFFT/2+1);
anm_stft = anm_stft(1:F, :, :);
freq = freq(1:F);
T = size(anm_stft,2);
R = zeros(Q, Q, F);
for f=1:F
    x = reshape(anm_stft(f, :, :), T, Q);
    R(:,:,f) = x'*x/T;
end

end

