function [p, anm_target, fs, reflectionsInfo, sceneInfo, s] = simulator(seed, opts)
arguments
    seed
    opts.roomIndex = 2
    opts.maxWaves = inf;
    opts.array_type = "anm"
    opts.array_sh_order = 3 % relevant only for array_type == "anm"
    opts.target_sh_order = 6;
    opts.source_type (1,1) string {mustBeMember(opts.source_type, ["speech", "whitenoise"])} = "whitenoise"
    opts.dnr = inf;
    opts.T = 5;
    opts.no_signals (1,1) logical = false
    opts.plotFlag (1,1) logical = false
    opts.angle_dependence = true;
    opts.max_delay_of_early_reflections = 20e-3
end
rng(seed);

switch opts.roomIndex
    case 1 % small
        roomDim = [5,4,2.5];
%         R = 0.95;
    case 2 % medim
        roomDim = [7,5,3];
%         R = 0.95;
    case 3 % large
        roomDim = [10,7,3.5];
%         R = 0.95;
%     case 4 % large
%         roomDim = [9,7,3.5];
%         R = 0.97;
    otherwise
        error();
end

if opts.angle_dependence
    R = 0.95;
else
    R = 0.9;
end


switch opts.source_type
    case "speech"
        [s, fs] = tsp.glued(opts.T, -1);
    case "whitenoise"
        fs = 48000;
        s = randn(round(fs*opts.T),1);
    otherwise
        error();
end
[arrayPos, sourcePos] = rand_pos(roomDim);

%% get responce
reflectionsInfo = image_method.calc_parametric_rir(roomDim, sourcePos, arrayPos, R, ...
    "maxwaves", opts.maxWaves, "zerofirstdelay", true, "angledependence", opts.angle_dependence);
if ~opts.no_signals
    [h_target, reflectionsInfo.delay, roomParams] = rir_from_parametric(fs, ...
        reflectionsInfo.delay, ...
        reflectionsInfo.amp, ...
        reflectionsInfo.omega, ...
        "array_type", "anm",...
        "N", opts.target_sh_order, ...
        "bpfFlag", false); 
    
    [h_array, reflectionsInfo.delay, roomParams] = rir_from_parametric(fs, ...
        reflectionsInfo.delay, ...
        reflectionsInfo.amp, ...
        reflectionsInfo.omega, ...
        "array_type", opts.array_type,...
        "N", opts.array_sh_order, ...
        "bpfFlag", false);

    sceneInfo.T60 = roomParams.T60;
    sceneInfo.DRR = roomParams.DRR;
    sceneInfo.SNR = RoomParams.dnr_drr_to_snr(opts.dnr, roomParams.DRR);
    fprintf("T60: %.2f sec\n", roomParams.T60);
    fprintf("DRR: %.1f dB\n", roomParams.DRR);
    fprintf("DNR: %.1f dB\n", opts.dnr);
    snr = RoomParams.dnr_drr_to_snr(opts.dnr, roomParams.DRR);
    fprintf("SNR: %.1f dB\n", snr);
end

%% calc some sceneInfos
sceneInfo.dist = norm(sourcePos-arrayPos);
sceneInfo.num_of_early_reflections = nnz(reflectionsInfo.delay < opts.max_delay_of_early_reflections )-1;

fprintf("Distance: %.2f meters\n", sceneInfo.dist);
fprintf("Num of early reflections: %d\n", sceneInfo.num_of_early_reflections);

%%
if opts.no_signals
    p = [];
    fs = [];
    return
end

%% plot scene, print room parameters
if opts.plotFlag
    K = find(reflectionsInfo.delay < opts.taumax, 1, 'last')-1;
    scene_plot(roomDim, arrayPos, sourcePos, reflectionsInfo.relativePos(2:K+1, :)+arrayPos, 3);
    figure("name", "RIR early");
    stem(reflectionsInfo.delay*1e3, reflectionsInfo.amp);
    xlim([0 opts.taumax*2*1e3]);
    
    figure("name", "RIR all");
    semilogy(reflectionsInfo.delay, abs(reflectionsInfo.amp), '.');
    xline(sceneInfo.T60, "color", "r");
    xlabel("Time [sec]");
end

%% convolve with responce
if size(h_array,1)==1
    p = s*h_array;
    anm_target  = s*h_target;
else
    p = fftfilt(h_array, s, 2^15); % 2^15 seems to be the fastest on my mac
    anm_target = fftfilt(h_target, s, 2^15); % 2^15 seems to be the fastest on my mac
end
p = p./std(p, [], "all");
anm_target = anm_target./std(anm_target(:,1));

%% add noise
noise = randn(size(p))* 10^(-snr/20);
p = p+noise;

% % decimate (discard high frequencies)
% [p, fs] = decimate_cols(p, fs, fmax*2*1.05);


end