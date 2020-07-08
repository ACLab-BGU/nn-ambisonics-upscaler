function [h, parametric, roomParams] = calc_rir(fs, roomDim, sourcePos, arrayPos, R, calc_parametric_rir_name_value_pairs, varargin)
% for details about name-value argument see
% image_method.calc_parametric_rir.m 
% and
% rir_from_parametric.m.

%% calc paramteric info
parametric = image_method.calc_parametric_rir(roomDim, sourcePos, arrayPos,R, calc_parametric_rir_name_value_pairs{:});

%% convert to rir signal
[h, parametric.delay, roomParams] = rir_from_parametric(fs, parametric.delay, parametric.amp, parametric.omega, varargin{:});

end