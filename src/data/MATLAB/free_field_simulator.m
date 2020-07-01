function [anm, fs, doa] = free_field_simulator( length_sec, N, opts )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
arguments
    length_sec (1,1) double {mustBeNonnegative}
    N (1,1) double {mustBeNonnegative, mustBeInteger}
    opts.doa (1,2) double = rand_on_sphere(1);
    opts.speaker_index (1,1) double {mustBeInteger, mustBePositive} = 1;
end

[s, fs] = tsp.glued(length_sec, opts.speaker_index);
Yh = conj(shmat(N, opts.doa, true, false));
anm = s.*Yh;
doa = opts.doa;

end

