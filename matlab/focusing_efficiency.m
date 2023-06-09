%Calculates the focusing efficiency at each wavelength
%Focusing efficiency: fraction of light that passes through circular
%aperture with radius equal to three times FWHM spot size

function [FWHMs, efficiency, transmission] = focusing_efficiency(Efocus, Hfocus, sp, norm, NA)

x = Efocus.x;
y = Efocus.y;
z = Efocus.z;
lambda = Efocus.lambda;

%Poynting vector
Efield = Efocus.E;
Hfield = Hfocus.H;
P = cross(Efield, conj(Hfield));

%Dot through monitor normal
n = zeros(size(Efield));
n(:,1,:) = norm(1);
n(:,2,:) = norm(2);
n(:,3,:) = norm(3);
T = 0.5*dot(real(P),n,2);

Nx = length(x);
Ny = length(y);
Nz = length(z);
Nlambda = length(lambda);

FWHMs = zeros(1, Nlambda);
efficiency = zeros(1,Nlambda);
transmission = zeros(1, Nlambda);
%Iterate over each wavelength
for i = 1:Nlambda
    Tlambda = reshape(T(:,1,i), [Nx Ny]);
    Tx = Tlambda(:,ceil(Ny/2));
    FWHMs(i) = myfwhm(x, Tx);
    
    %Cutoff aperature
    %targetrad = 3/2*FWHMs(i);
    targetrad = (3/2)*lambda(i)/(2*NA);
    %targetrad = 0.5e-6;
    Taperture = Tlambda;
    for j = 1:Nx
        if Ny > 1
            for k = 1:Ny
                if sqrt(x(j)^2 + y(k)^2) > targetrad
                    Taperture(j,k) = 0;
                end
            end
        else
            if sqrt(x(j)^2) > targetrad
                Taperture(j) = 0;
            end
        end
    end
    if Ny > 1
        efficiency(i) = trapz(x,trapz(y,Taperture))/sp(i);
        transmission(i) = trapz(x,trapz(y,Tlambda))/sp(i); 
    else
        efficiency(i) = trapz(x,Taperture)/sp(i);
        transmission(i) = trapz(x,Tlambda)/sp(i);  
    end
end
figure;
hold on;
plot(lambda*1e9, FWHMs*1e9)
if exist('NA', 'var')
    plot(lambda*1e9, lambda*1e9/(2*NA))
    legend('Simulated', 'Diffraction limit')
end
xlabel('\lambda (nm)')
ylabel('FWHM (nm)')
title('FWHM of focal spot')

figure;
plot(lambda*1e9, transmission, lambda*1e9, efficiency)
xlabel('\lambda (nm)')
legend('Total transmission', 'Focusing efficiency')
title('Efficiency')
axis([min(lambda)*1e9, max(lambda)*1e9, 0, 1])

end
