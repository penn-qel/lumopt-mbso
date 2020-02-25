% Histograms showing the spread of parameters for the non-static
% metasurfaces

function F = gradient_history(date, letter)

dir = strcat('../Optimizations/', date, '/', letter, '/');
load(strcat(dir, 'history.mat'))

shape = size(params);

iterations = shape(1);
N = shape(2);

videoout = strcat(dir, 'history');
v = VideoWriter(videoout, 'MPEG-4');
v.FrameRate = 2;
open(v);

height = double(height);

%Create vector to define how gradients are color mapped
maxgrad = max(max(abs(grad)));
colorvector = -maxgrad:(2*maxgrad/N):maxgrad;

F(iterations) = struct('cdata',[],'colormap',[]);
for i = 1:iterations
    clf
    offsetgrad = grad(i,1:N/2);
    widthgrad = grad(i,N/2+1:end);
    offsets = params(i,1:N/2);
    x = init_pos + offsets;
    widths = params(i,N/2+1:end);
    
    %Distribute gradients into bins defined by uniformly spaced edges
    Y = discretize(widthgrad, colorvector);
    Z = discretize(offsetgrad, colorvector);
    
    %Create colormap based on number of objects
    colorVec = redblue(N);
    
    for j = 1:length(widths)
        centerpos = x(j);
        width = widths(j);
        rectangle('Position', [centerpos-width/2 0 width height], 'FaceColor', colorVec(Y(j),:), 'LineStyle', 'none');
    end
    minpos = (min(init_pos) - 1.1*max(max(params)));
    maxpos = (max(init_pos) + 1.1*max(max(params)));
    figwidth = maxpos-minpos;
    figheight = 1.5*height;
    aspect = figwidth/figheight;
    
    text(0, height*1.3, strcat('Iteration #', num2str(i)));
    
    colormap(colorVec);
    caxis([-maxgrad maxgrad])
    colorbar;

    axis([minpos, maxpos, 0, figheight])
    yticks([])
    set(gca, 'YTickLabel', {' '})
    set(gca, 'ycolor', 'none')
    pixelheight = 20;
    set(gcf,'Position',[100,100,100+pixelheight*aspect, 100+pixelheight])
    F(i) = getframe(gcf);
    writeVideo(v, F(i));
end
close(v);
end