function fnm = sh_freq_complement(fnm, nfft, f_dim, sh_dim)


% input check
assert(ndims(fnm)<=3, 'This function 2D and 3D arrays only');
assert(f_dim <=3 && sh_dim <= 3 && f_dim ~= sh_dim, 'f_dim and sh_dim must be different and between 1 and 3');

% permute matrix
extra_dim = 1:3;
extra_dim([f_dim sh_dim]) = [];
permute_order = [sh_dim f_dim extra_dim];
fnm = permute(fnm, permute_order);

% get n,m  of each index
i = (1:size(fnm,1))';
[n,m] = i2nm(i);

% get first half
if mod(nfft, 2)==0
    fnm_c = fnm(:, end-1:-1:2, :);
else
    fnm_c = fnm(:, end  :-1:2, :);
end

% add missing frequecies
fnm_c = conj(fnm_c);
fnm_c = fnm_c( nm2i(n, -m), : , :) .* (-1).^m;
fnm = cat(2, fnm, fnm_c);
assert(nfft==size(fnm,2), 'nfft does not match size(fnm, f_din)');

% permute to original dimension permutation
fnm = ipermute(fnm, permute_order);

end

