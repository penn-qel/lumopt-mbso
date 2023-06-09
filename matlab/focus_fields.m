%Makes plots at focal plane. Takes in Lumerical struct.
function focus_fields(Esurf)

x = Esurf.x;
y = Esurf.y;
z = Esurf.z;
Efield = Esurf.E;
lambda = Esurf.lambda;


Nx = length(x);
Ny = length(y);
Nz = length(z);
Nlambda = length(lambda);


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
suptitle('Intensity at focal plane')

end
