function [a_loc,s_loc] = rand_pos(roomDim, opts)

arguments
    roomDim (1,3) double
    opts.min_source_array_dist = 1
    opts.min_dist_from_walls = 1
end

while 1
    s_loc(1) = rand_between([0 roomDim(1)] + opts.min_dist_from_walls*[1 -1]);
    s_loc(2) = rand_between([0 roomDim(2)] + opts.min_dist_from_walls*[1 -1]);
    s_loc(3) = rand_between([0 roomDim(3)] + opts.min_dist_from_walls*[1 -1]);
    
    a_loc(1) = rand_between([0 roomDim(1)] + opts.min_dist_from_walls*[1 -1]);
    a_loc(2) = rand_between([0 roomDim(2)] + opts.min_dist_from_walls*[1 -1]);
    a_loc(3) = rand_between([0 roomDim(3)] + opts.min_dist_from_walls*[1 -1]);
    
    source_to_array_dist = norm(a_loc-s_loc);
    if source_to_array_dist>opts.min_source_array_dist
        break
    end
end

end

