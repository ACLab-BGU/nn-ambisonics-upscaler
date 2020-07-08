function h = plot(th, ph, varargin)

if (nargin<2 || isempty(ph)) && size(th,2)>=2
    ph = th(:,2);
    th = th(:,1);
end
[x,y] = hammer.project(th, ph);
h = plot(x, y, varargin{:});
hammer.axes(h.Parent);

end