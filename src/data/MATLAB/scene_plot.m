function scene_plot(room, array, directSource, reflections, flag)

%% plot Room
if flag==3
    n=2;
    tiledlayout(n,1, "Padding", "compact", "TileSpacing", "compact");
    nexttile();
else
    n=1;
end

if flag==1 || flag==3
%     nexttile();
    h=plot3(array(:,1), array(:,2), array(:,3), '.', 'MarkerSize', 20, 'LineWidth', 4);
    hold on;
    
    plot3(directSource(:,1), directSource(:,2), directSource(:,3), '.', 'MarkerSize', 18, 'LineWidth', 4);
    
    if ~isempty(reflections)
        plot3(reflections(:,1), reflections(:,2), reflections(:,3), '.', 'MarkerSize', 18, 'LineWidth', 4);
        text(reflections(:,1), reflections(:,2), reflections(:,3), "  "+(1:size(reflections,1))', "HorizontalAlignment", "left");
        legend('Microphone Array', 'Direct Sound', 'Early Reflections', 'Location', 'eastoutside');
        
    else
        legend('Microphone Array', 'Direct Sound', 'Location', 'eastoutside');
        
    end
    plotcube(room, [0 0 0], 0.1, [0 1 0]);
    h_patch = findobj(h.Parent, "type", "patch");
    for hh=h_patch(2:end)'
        hasbehavior(hh, 'legend', false)
    end
    h_patch(1).DisplayName = "Room";
    view(0,90);
    xlabel('$x$ (m)');
    ylabel('$y$ (m)');
    zlabel('$z$ (m)');
end

if flag==2 || flag==3
    if n==2
        nexttile();
    end
    [omega(:,1), omega(:,2)] = c2s( [directSource; reflections] - array );
    hammer.plot(omega(1,:), [], '.', 'MarkerSize', 25);
    hold on;
    hammer.plot(omega(2:end, :), [], '.', 'MarkerSize', 18);
    hammer.text(omega(2:end, :), [], [], " "+(1:size(reflections,1))', 'FontWeight', 'Bold');
    hammer.room(room, array);
    mylegend("Direct", "Early reflections");
end

end

