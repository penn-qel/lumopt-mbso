function transmission = collection_efficiency(Eout, Hout, dp);

NA = 0.1;
thetamax = asin(NA); %Angle in radians corresponds to NA of 0.1
thetamaxdeg = thetamax*180/pi;

epsilon0 = 8.85e-12;
mu0 = 4e-7*pi;
Z = sqrt(mu0/epsilon0);

E = Eout.E;
H = Hout.H;
lambda = Eout.lambda;
x = Eout.x;
y = Eout.y;
z = Eout.z;

Nlambda = length(lambda);


transmission = zeros(11,1);
for i = 1:length(lambda)
    N = length(x);
    Efield = E(:,:,i);
    Hfield = H(:,:,i);
    Espectral = fftshift(fft(Efield));
    Hspectral = fftshift(fft(Hfield));
    k0 = 2*pi/lambda(i);
    p = -floor(N/2):floor(N/2); %Indices for fft
    dx = (max(x) - min(x))/(N-1); %Sampling period
    Fs = 1/dx; %Sampling frequency
    kx = Fs/N*p;
    
    
    theta = asin(abs(kx)/k0);
    kxmax = k0*sin(thetamax);
    Efiltered = Espectral;
    Hfiltered = Hspectral;
    %Remove spatial frequencies corresponding to high angles
    for j = 1:3
        Efiltered(:,j) = Espectral(:,j).*(abs(kx) <= kxmax)';
        Hfiltered(:,j) = Hspectral(:,j).*(abs(kx) <= kxmax)';
    end
    
    Efieldfiltered = ifft(fftshift(Efiltered));
    Hfieldfiltered = ifft(fftshift(Hfiltered));
    S = 0.5*real(cross(Efieldfiltered, conj(Hfieldfiltered)));
    Power = vecnorm(S,2,2);
    transmission(i) = trapz(x, Power)/dp(i);
end   
plot(lambda, transmission)
xlabel('\lambda (nm)')
ylabel('Transmission into NA < 0.1')
