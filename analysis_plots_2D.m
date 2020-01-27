%%Initialize
clear;
close all;


%Path to file
%Should have variables Efield, Efocus, Esurf
%Each is a Lumerical struct containing whole field, focal plane, and
%bottom surface, respectively
load('../Optimizations/20191121/A/results.mat');

%% Call cross_section_fields function

y = exist('Efield');
if y == 1
    cross_section_fields_2D(Efield);
else
    disp('Efield struct not found')
end

% %% Call bottom_surface_fields function
% 
% y = exist('Esurf');
% if y == 1
%     bottom_surface_fields(Esurf);
% else
%     disp('Esurf struct not found')
% end

%% Call focus_fields function

y = exist('Efocus');
if y == 1
    focus_fields_2D(Efocus);
else
    disp('Efocus struct not found')
end

%% Call focusing_efficiency function

y = exist('Efocus');
z = exist('Hfocus');
x = exist('sp');
if y == 1 & x == 1 & z == 1
    n = exist('NA');
    if n == 1
        [FWHMs, efficiency, transmission] = focusing_efficiency(Efocus, Hfocus, sp, [0 -1 0], NA);
    else
        [FWHMs, efficiency, transmission] = focusing_efficiency(Efocus, Hfocus, sp, [0 -1 0]);
    end
    %disp(FWHMs*1e9);
    %disp(efficiency);
    %disp(transmission);
else
    disp('Efocus struct not found')
end