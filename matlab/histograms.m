% Histograms showing the spread of parameters for the non-static
% metasurfaces

close all;
clear;

load('../Optimizations/20200129/A/params.mat')
init_widths = (-15:0.2:15).*1e-6;
Nbins = 20;

offsets = x - init_widths;
N = length(offsets);

gaps = zeros(1,N-1);
for i=1:(N-1)
    gaps(i) = x(i+1) - x(i) - 0.5*widths(i) - 0.5*widths(i+1);
end

figure;
subplot(1,3,1)
histogram(widths.*1e9, Nbins);
xlabel('Pillar widths (nm)')
ylabel('Count')

subplot(1,3,2)
histogram(offsets.*1e9, Nbins);
xlabel('Offsets from initial grid (nm)')
ylabel('Count')

subplot(1,3,3)
histogram(gaps.*1e9, Nbins);
xlabel('Gaps between pillars (nm)')
ylabel('Count')

set(gcf,'Position',[100,100,1200 300])
