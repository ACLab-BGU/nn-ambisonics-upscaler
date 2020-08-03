function v = vec_hermitian_matrix(mat, is_complex)
arguments
    mat (:,:)
    is_complex (1,1) logical
end
N = size(mat,1);
M = size(mat,2);
assert(N==M);
[n,m] = ndgrid((1:N)', (1:M));
if ~is_complex
    lower = n>=m;
    v = mat(lower(:));
else
    diag = n==m;
    lower = n>m;
    v = [real(mat(diag(:))); real(mat(lower(:))); imag(mat(lower(:)))];
end

end