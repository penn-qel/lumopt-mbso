% Makes 2D plot of cross-section of data. Takes in Lumerical struct

function cross_section_fields(E)
x = E.x;
y = E.y;
z = E.z;
Efield = E.E;
lambda = E.lambda;

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
    pcolor(x.*1E6, z.*1E6,reshape(I(:,ceil(Ny/2),:), [Nx Nz])');
    shading flat;
    colorbar
    colormap('jet')
    xlabel('x (um)')
    ylabel('z (um)')
    title(['|E|^2 at ', num2str(lambda(i)*1e9), ' nm'])
end
suptitle('Cross-sectional field profile')
end