function D = diagnd(A, diagdim, dim2add)
%UNTITLED13 Summary of this function goes here
%   Detailed explanation goes here
error("in development, not finished");
N = size(A,diagdim);
permuteVec = [diagdim setdiff(1:ndims(A), diagdim)];
A = permute(A, permuteVec);

original_size = size(A);
original_ndims = ndims(A);
A = reshape(A, N, []);
D = zeros([size(A) N]);
for i=1:size(A,2)
    D(:, i, :) = diag(A(:, i));
end
D = reshape(D, [original_size N]);
D = ipermute(D, [permuteVec original_ndims+1]);

end

