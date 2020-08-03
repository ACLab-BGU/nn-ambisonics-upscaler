function [  ] = close_all_waitbars( displayFlag )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

if nargin==0
    displayFlag=true;
end
h = findall(0,'tag', 'TMWWaitbar');
close(h);
if displayFlag
    disp([num2str(length(h)) ' waitbars were closed']);
end
end
