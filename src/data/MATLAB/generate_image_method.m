restoredefaultpath
addpath('spherical harmonics/');

%%
SELECTED_FREQ = 500;

%%
[p, anm_target, fs, reflectionsInfo, sceneInfo, s] = simulator(1);

%%
[R_p, freq] = calculate_narrowband_scm(p, fs);
R_anm_target = calculate_narrowband_scm(anm_target, fs);
[~, i] = min(abs(freq-SELECTED_FREQ));
