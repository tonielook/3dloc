function [u1] = ADMM_poisson_neg(img, psf,b,mu,nu,thd)
%% modified from ADMM_decov.m
% ADMM for KL-L1 model

%% initial variable setting
[ny, nx, nz] = size(psf);

y = img(:)';
T = zeros(1, nz); T(nz) = 1;
A = psf; % use A matrix to represent the 3D PSF, fA is fast Fourier transform of A
% u0 is the 3D image space, x is the 3D recovered space, u1 is the sparse version of x
% fu0, fx, fu1 are the corresponding Fourier transform.
% Because most of the calculation is done in Fourier domain, corresponding
% variable in space domain may not be required.
fx = zeros(size(psf));
feta0 = fx; feta1 = fx;

%% For ADMM_docov

N = 800;% 1000 iterations
TT = (T'*T+mu*eye(nz));
fA = fftn(A);
invall = abs(fA).^2+nu;
% thd = max(img(:))/max(psf(:))*0.1/5;% thresdhold to calculate u1 from x %0.09
% thd = lambda/nu;
u0_2d = TT\(T'*y);

u0 = reshape(u0_2d', ny, nx, nz);
fu0 = fftn(u0);
fu1 = zeros(size(u0));
%% ADMM iterations
% Calculations are mostly conducted in Fourier domain
for k = 1:N
    ftemp_1 = conj(fA).*(fu0-feta0);
    fx = (ftemp_1+nu*(fu1-feta1))./invall;% update x, equation (7)
    fAx = fA.*fx;
    feta0 = feta0-(fu0-fAx);% update eta0, equation (8)
    feta1 = feta1-(fu1-fx);% update eta1, equation (9)
    Ax_eta_2d = ifftn(fAx+feta0);
    if nz > 1 
        Ax_eta_2d = reshape(XYZ_rot(Ax_eta_2d, [3,1,2]), nz, ny*nx);
    else
        Ax_eta_2d = Ax_eta_2d(:)';
    end
    TT1 = T'*ones(1,ny*nx); TT1 =  mu* Ax_eta_2d - (1+mu*b)*TT1;
    u0_2d = (TT1 + sqrt(TT1.^2+4*mu*T'*(y-b+b*mu*Ax_eta_2d(end,:))))./(2*mu);
    u0 = reshape(u0_2d', ny, nx, nz);% update u0, equation (5)
    fu0 = fftn(u0);
    x_eta1 = ifftn(fx+feta1);
    u1 = max(x_eta1-thd, 0);% update u1, equation (6) % for ADMM_deconv.m
    fu1 = fftn(u1);

end

% Get final output
u1 = fftshift(u1);
u1 = ifftshift(u1,3);
