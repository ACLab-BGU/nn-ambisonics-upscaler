function Y = shmat_fast(N, dirs, isComplex, transposeFlag)
%GETSH Get Spherical harmonics up to order N
%
%   N:  maximum order of harmonics
%   dirs:   [azimuth_1 inclination_1; ...; azimuth_K inclination_K] angles
%           in rads for each evaluation point, where inclination is the
%           polar angle from zenith: inclination = pi/2-elevation
%   basisType:  'complex' or 'real' spherical harmonics (default:
%   'complex')
%   transposeFlag
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Archontis Politis, 10/10/2013
%   archontis.politis@aalto.fi
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Modified by Tom Shlomo, Dec 2019
% - Removed last transposition that was quite expensive.
% - Added option for transposed output (useful for steering matrices)
% - Added defaults
% - changes dirs to be [theta phi], where theta is angle with the z-axis
%   (top to bottom).
% - cached some variables for fast calulcation
% - changed calculation of normalization factors to be without factorials,
%   to improve numeric stability. This is important when N>30
%   (for double precision).

if nargin<4 || isempty(transposeFlag)
    transposeFlag = false;
end
if nargin<3
    isComplex = true;
end
assert(isComplex, "Only complex SH are supported currently");
Q = size(dirs, 1);
L = (N+1)^2;

persistent N_cache mnonneg_ind mnonneg mneg_ind mneg_phase j leg z2N
if isempty(N_cache) || N~=N_cache
    [nvec,mvec] = i2nm((1:L)', true);
%     Nnm = sqrt( (2*nvec+1).*factorial(nvec-mvec) ./ (4*pi*factorial(nvec+mvec)) );
    Nnm = zeros(L,1);
    for i=1:L
        if mvec(i)>=0
            Nnm(i) = sqrt( (2*nvec(i)+1) /( (4*pi) * prod(nvec(i)+(-mvec(i)+1:mvec(i))) ));
        else
            Nnm(i) = sqrt( (2*nvec(i)+1) * prod(nvec(i)+(mvec(i)+1:-mvec(i))) / (4*pi) );
        end
    end
    z2N = 0:N;
    mnonneg_ind = mvec>=0;
    mnonneg = mvec(mnonneg_ind);
    
%     half_mnonneg = mvec(mnonneg)
    mneg_ind = mvec<0;
%     Nnm = Nnm(mnonneg);
    mneg_phase = (-1).^(mvec(mneg_ind)).';
    [j,~] = find(-mvec==mvec' & nvec==nvec');
    j = j(mneg_ind);
    N_cache = N;
    leg = zeros(L, N+1);
    for n=0:N
        c = LegendrePoly(n).';
        c = [zeros(1,N+1-length(c)) c]; %#ok<AGROW>
        for m=0:n
            leg(nm2i(n,m),:) = c;
            c =  polyder(c);
            c = [zeros(1, N+1-length(c)) c]; %#ok<AGROW>
        end
    end
    leg = leg .* Nnm .* (-1).^mvec;
    leg = leg(mnonneg_ind,:);
    leg = leg.';
    leg = flipud(leg);
end

%% Tom: some speedups
if transposeFlag
    V = (cos(dirs(:,1)).') .^ (z2N.') ;
    U = (leg.' * V);
    
    if N>=2
        cos_sq = V(3,:);
    elseif N==1
        cos_sq = (V(1,:).^2);
    end
    M = sqrt(1-cos_sq).^(z2N.') .* exp(1i*z2N.'.*dirs(:,2).');
    U = U.*M(mnonneg+1,:);
    Y = zeros(L,Q);
    Y(mnonneg_ind,:) = U;
    Y(mneg_ind,:) = conj(Y(j,:)) .* mneg_phase.';
else
    V = cos(dirs(:,1)).^z2N;
    U = V*leg;
    
    if N>=2
        cos_sq = V(:,3);
    else
        cos_sq = V(:,1).^2;
    end
    M = sqrt(1-cos_sq).^z2N .* exp(1i*z2N.*dirs(:,2));
    U = U.*M(:, mnonneg+1);
    Y = zeros(Q, L);
    Y(:,mnonneg_ind) = U;
    Y(:,mneg_ind) = conj(Y(:,j)) .* mneg_phase;
end
% 
% %%
% Y = zeros(L, Q);
% for n=0:N
%     % vector of unnormalised associated Legendre functions of current order
%     Y(i(:,n+1),:) = legendre(n, cos_theta);
% end
% 
% % Lnm(m<0) = Lnm(m>0);
% % if ~transposeFlag
% %     Y = Y.';
% % end
% % Y(m>=0) = Nnm(m>=0) .* Y(m>=0) .* Exp(mvec(m>=0));
% % Y(m<0) = conj(Y(
% 
% if transposeFlag
%     Exp = exp(1i*(0:N).'.*dirs(:,2).');
%     Y(mnonneg_ind,:) = Y(mnonneg_ind,:) .* Nnm .* Exp(k,:);
%     Y(mneg_ind,:) = conj(Y(j,:)) .* mneg_phase;
% else
%     Exp = exp(1i*(0:N).*dirs(:,2));
%     Y = Y.';
%     Y(:,mnonneg_ind) = Y(:,mnonneg_ind) .* Nnm.' .* Exp(:, k);
%     Y(:,mneg_ind) = conj(Y(:,j)) .* mneg_phase.';
% end

end


% LegendrePoly.m by David Terr, Raytheon, 5-10-04
% Given nonnegative integer n, compute the 
% Legendre polynomial P_n. Return the result as a vector whose mth
% element is the coefficient of x^(n+1-m).
% polyval(LegendrePoly(n),x) evaluates P_n(x).
function pk = LegendrePoly(n)
if n==0 
    pk = 1;
elseif n==1
    pk = [1 0]';
else
    
    pkm2 = zeros(n+1,1);
    pkm2(n+1) = 1;
    pkm1 = zeros(n+1,1);
    pkm1(n) = 1;
    for k=2:n
        
        pk = zeros(n+1,1);
        for e=n-k+1:2:n
            pk(e) = (2*k-1)*pkm1(e+1) + (1-k)*pkm2(e);
        end
        
        pk(n+1) = pk(n+1) + (1-k)*pkm2(n+1);
        pk = pk/k;
        
        if k<n
            pkm2 = pkm1;
            pkm1 = pk;
        end
        
    end
    
end
end