function file_path = make_image_method_data(num_of_files,start_index, opts)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

arguments
    num_of_files
    start_index = 1
    opts.output_signals (1,1) logical = true
    opts.output_type (1,1) string {mustBeMember(opts.output_type, ["covs", "signals"])} = "signals"
    opts.number_of_reflections (1,1) double = inf
%     opts.output_freqs = "all" % relevant only for output_type = "covs".
    opts.folder_path (1,1) string = fullfile(get_raw_data_folder_path(), "image-method")
    opts.decimation_factor = 3
    opts.R_dtype = "single"
    opts.anm_dtype = "int16"
    opts.nfft (1,1) double = 512
    opts.stft_win_size (1,1) double = 512
    opts.stft_hop_size (1,1) double = 256
    opts.source_type = "whitenoise"
    opts.duration = 3 % sec
    opts.signal_order_to_save = 3
    opts.real_sh (1,1) logical = true
    opts.target_sh_order = 6
    opts.vec_covs (1,1) logical = true
    opts.compact_cov (1,1) logical = true
end

file_path = strings(num_of_files,1);
wb = wbar('dros');
for i=start_index:num_of_files
    %%
    [~, anm, fs] = simulator(i, ...
        "T", opts.duration+2, ...
        "roomIndex", 2, ...
        "decimation_factor", opts.decimation_factor, ...
        "maxWaves", opts.number_of_reflections+1, ...
        "source_type", opts.source_type, ...
        "real_sh", opts.real_sh, ...
        "target_sh_order", opts.target_sh_order);
    Q = size(anm, 2);
    anm = anm((ceil(1*fs)+1):(size(anm,1)-ceil(1*fs)),:);

    %% calculate matrices
    [R, freq] = calculate_narrowband_scm(anm, fs, "NFFT", opts.nfft, "stft_window", hann(opts.stft_win_size),"stft_hop",opts.stft_hop_size);
    F = size(R, 3);
    if opts.compact_cov
        R_complex = R;
        if opts.vec_covs
            R = zeros( Q^2, F );
            for f=1:F
                R(:,f) = vec_hermitian_matrix(R_complex(:,:,f), 1);
            end
        else
            R = zeros(Q, Q, 2, F);
            R(:,:,1,:) = real(R_complex);
            R(:,:,2,:) = imag(R_complex);
        end
    else
        R = permute(R,[3,1,2]);
    end
    [R, R_scaling] = cast_to(R, opts.R_dtype, 1);
    seed = i; %#ok<NASGU>
    nfft = opts.nfft; %#ok<NASGU>
    N_R = opts.target_sh_order;
    is_real_sh = opts.real_sh;
    num_of_reflections = opts.number_of_reflections;
    vars = {"R", "fs", "freq", "seed", "nfft", "R_scaling", "N_R", "is_real_sh", "num_of_reflections"}; %#ok<CLARRSTR>
    
    %% signals
    if opts.output_signals
        anm = anm(:, 1:(opts.signal_order_to_save+1)^2);
        if ~opts.real_sh
            anm = cat(3, real(anm), imag(anm));
        end
        [anm, anm_scaling] = cast_to(anm, opts.anm_dtype, 0.99); %#ok<*ASGLU>
        N_signals = opts.signal_order_to_save;
        vars = [vars, {"anm", "anm_scaling", "N_signals"}]; %#ok<AGROW>
    end
    
    %% file path
    file_path(i) = fullfile(opts.folder_path, sprintf("%08d.mat", i));
    save(file_path(i), vars{:});
    wbar(i,num_of_files,wb);
end

    function [v, scaling] = cast_to(v, type, p)
        switch type
            case "int16"
                max_val = 2^15;
            case "double"
                max_val = inf;
            case "single"
                max_val = inf;
            otherwise
                error();
        end
        if ~isinf(max_val) && p<1
            cut = prctile(abs(v), p*100, 'all');
            scaling = max_val/cut;
        else
            scaling = 1;
        end
        v = cast(v*scaling, type);
    end
end