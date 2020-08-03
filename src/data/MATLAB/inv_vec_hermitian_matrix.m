function [mat] = inv_vec_hermitian_matrix(v, is_complex)
arguments
    v (:,1)
    is_complex (1,1) logical
end
K = size(v,1);
if is_complex
    N = sqrt(K);
else
    N = sqrt(1+2*K)-1;
end
N = round(N);
mat = zeros(N);
[n,m] = ndgrid((1:N)', (1:N));
if ~is_complex
    lower = n>=m;
    mat(lower(:)) = v;
else
    diag = n==m;
    lower = n>m;
    mat(diag(:)) = v(1:N);
    mat(lower(:)) = v( (N+1) : (N+0.5*N*(N-1)) ) + 1i * v(N+0.5*N*(N-1)+1:end);
end

mat_h = mat';
mat_h(n==m) = 0;
mat = mat + mat_h;

end