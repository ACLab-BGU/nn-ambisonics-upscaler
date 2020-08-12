function fnm = sft_complex_to_real(fnm, dim)

Q = size(fnm, dim);
N = sqrt(Q)-1;
T = complex2realSHMtx(N); % Q x Q
switch dim
    case 1
        fnm = T*fnm;
    case 2
        fnm = fnm*T.'; % = (T*fnm.').'
    otherwise
        error("sorry, not implemented :(");
end

end