function x = decimate_cols(x, decimation_factor)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

x_bu = x;
Q = size(x,2);
x = zeros(ceil(size(x_bu,1)/decimation_factor), Q);
for q=1:Q
    x(:,q) = decimate(x_bu(:,q), decimation_factor);
end

end

