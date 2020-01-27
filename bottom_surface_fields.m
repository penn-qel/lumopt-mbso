%Makes plots at bottom surface of structure. Takes in Lumerical struct.
function bottom_surface_fields(Esurf)

x = Esurf.x;
y = Esurf.y;
z = Esurf.z;
Efield = Esurf.E;
lambda = Esurf.lambda;


Nx = length(x);
Ny = length(y);
Nz = length(z);
Nlambda = length(lambda);


wavelengthindex = ceil(length(lambda)/2);
E = Efield(:,:,wavelengthindex);


Ex = reshape(E(:,1), [Nx Ny Nz]);
Ey = reshape(E(:,2), [Nx Ny Nz]);
Ez = reshape(E(:,3), [Nx Ny Nz]);

I = abs(Ex).^2 + abs(Ey).^2 + abs(Ez).^2;

figure;
subplot(2,3,1);
pcolor(x.*1E6, y.*1E6,abs(reshape(Ex(:,:,1), [Nx Ny])'));
shading flat;
colorbar
colormap('jet')
xlabel('x (um)')
ylabel('y (um)')
title('|E_x|')

subplot(2,3,2);
pcolor(x.*1E6, y.*1E6,abs(reshape(Ey(:,:,1), [Nx Ny])'));
shading flat;
colorbar
colormap('jet')
xlabel('x (um)')
ylabel('y (um)')
title('|E_y|')

subplot(2,3,3);
pcolor(x.*1E6, y.*1E6,abs(reshape(Ez(:,:,1), [Nx Ny])'));
shading flat;
colorbar
colormap('jet')
xlabel('x (um)')
ylabel('y (um)')
title('|E_z|')

center = ceil(Nx/2);

Phix = angle(reshape(Ex(:, :, 1), [Nx Ny])');
%Phix = Phix + 2*pi - Phix(center,center);
subplot(2,3,4);
pcolor(x.*1E6, y.*1E6,wrapTo2Pi(Phix));
shading flat;
colorbar
colormap('jet')
xlabel('x (um)')
ylabel('y (um)')
title('\angle E_x')

Phiy = angle(reshape(Ey(:, :, 1), [Nx Ny])');
%Phiy = Phiy + 2*pi - Phiy(center,center);
subplot(2,3,5);
pcolor(x.*1E6, y.*1E6,wrapTo2Pi(Phiy));
shading flat;
colorbar
colormap('jet')
xlabel('x (um)')
ylabel('y (um)')
title('\angle E_y')

Phiz = angle(reshape(Ez(:, :, 1), [Nx Ny])');
%Phiz = Phiz + 2*pi - Phiz(center,center);
subplot(2,3,6);
pcolor(x.*1E6, y.*1E6,wrapTo2Pi(Phiz));
shading flat;
colorbar
colormap('jet')
xlabel('x (um)')
ylabel('y (um)')
title('\angle E_z')
suptitle('Magnitude and phase below surface at center wavelength')

figure;
hold on;
nplotsx = 3;
nplotsy = ceil(Nlambda/nplotsx);
for i = 1:Nlambda
    subplot(nplotsy, nplotsx, i); 
    Elambda = Efield(:,:,i);
    I = abs(reshape(Elambda(:,1), [Nx Ny Nz])).^2 + abs(reshape(Elambda(:,2), [Nx Ny Nz])).^2 + abs(reshape(Elambda(:,3), [Nx Ny Nz])).^2;
    pcolor(x.*1E6, y.*1E6,reshape(I(:,:,:), [Nx Ny])');
    shading flat;
    colorbar
    colormap('jet')
    xlabel('x (um)')
    ylabel('y (um)')
    title(['|E|^2 at ', num2str(lambda(i)*1e9), ' nm'])
end
suptitle('Intensity at bottom surface')

end

