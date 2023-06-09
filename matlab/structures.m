close all;
clear;

load('../Optimizations/20200129/A/params.mat')
x1 = x;
widths1 = widths;

figure(1);
for i = 1:length(widths)
    centerpos1 = x1(i)*1e6;
    width1 = widths1(i)*1e6;
    rectangle('Position', [centerpos1-width1/2 0 width1 height*1e6], 'FaceColor', 'k', 'LineStyle', 'none');
end

minpos = (min(x1) - max(widths)).*1e6;
maxpos = (max(x1) + max(widths)).*1e6;
figwidth = maxpos-minpos;
figheight = height.*1e6;
aspect = figwidth/figheight;

axis([minpos, maxpos, 0, figheight])
yticks([])
set(gca, 'YTickLabel', {' '})
set(gca, 'ycolor', 'none')
pixelheight = 20;
set(gcf,'Position',[100,100,100+pixelheight*aspect, 100+pixelheight])